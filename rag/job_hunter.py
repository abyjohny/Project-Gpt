"""
job_hunter.py - Job hunting engine for Project GPT.
====================================================

Searches for jobs, analyses CVs, matches jobs to skills,
drafts cover emails, and sends applications with CV attached.

Data directories:
    data/cv/          uploaded CVs
    data/jobs/        job history JSON (applications.json, config.json)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests
from ddgs import DDGS

from rag import config, email_client, gemini_client, ollama_client

logger = logging.getLogger("job_hunter")

CV_DIR = config.PROJECT_ROOT / "data" / "cv"
JOBS_DIR = config.PROJECT_ROOT / "data" / "jobs"
APPLICATIONS_FILE = JOBS_DIR / "applications.json"
CONFIG_FILE = JOBS_DIR / "config.json"
PRESETS_FILE = JOBS_DIR / "presets.json"
RUNNER_FILE = JOBS_DIR / "runner.json"
DAILY_RESULTS_FILE = JOBS_DIR / "daily_results.json"

EXPERIENCE_LEVELS = [
    "Entry Level",
    "Junior",
    "Mid-Level",
    "Senior",
    "Lead",
    "Manager",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

FILTERED_EMAIL_PREFIXES = {
    "noreply", "no-reply", "no_reply", "donotreply", "do-not-reply",
    "support", "help", "info", "admin", "webmaster", "postmaster",
    "mailer-daemon", "abuse", "spam", "notifications", "newsletter",
    "marketing", "sales", "billing", "feedback", "unsubscribe",
}

# Sites that are job aggregators (not direct company postings)
AGGREGATOR_DOMAINS = {
    "linkedin.com", "indeed.com", "glassdoor.com", "monster.com",
    "naukri.com", "ziprecruiter.com", "simplyhired.com", "dice.com",
    "careerbuilder.com", "jooble.org", "jobrapido.com", "adzuna.com",
    "reed.co.uk", "totaljobs.com", "cwjobs.co.uk", "seek.com.au",
    "shine.com", "foundit.in", "instahyre.com", "hirist.com",
    "timesjobs.com", "apna.co", "cutshort.io",
}


# ------------------------------------------------------------------
# CV text extraction
# ------------------------------------------------------------------

def extract_cv_text(filepath: Path) -> str:
    """Read a CV file (PDF, DOCX, TXT) and return its raw text."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".pdf":
        import pymupdf
        text_parts: list[str] = []
        with pymupdf.open(str(filepath)) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts).strip()

    if suffix in (".docx", ".doc"):
        import docx
        doc = docx.Document(str(filepath))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    # .txt or other plain-text
    return filepath.read_text(encoding="utf-8", errors="replace").strip()


# ------------------------------------------------------------------
# Job search (DuckDuckGo)
# ------------------------------------------------------------------

def search_jobs(
    role: str,
    country: str,
    state: str = "",
    date_from: str = "",
    date_to: str = "",
    experience: str = "",
) -> list[dict]:
    """Search DuckDuckGo for job postings. Returns deduplicated list of dicts."""
    location = f"{state} {country}".strip() if state else country
    exp_tag = f" {experience}" if experience else ""

    # Build several query variations for broader coverage
    queries = [
        f"{role}{exp_tag} jobs {location} site:linkedin.com OR site:indeed.com",
        f"{role}{exp_tag} hiring {location} site:linkedin.com OR site:glassdoor.com",
        f"{role}{exp_tag} careers {location}",
    ]
    if date_from:
        queries = [f"{q} after:{date_from}" for q in queries]
    if date_to:
        queries = [f"{q} before:{date_to}" for q in queries]

    seen_urls: set[str] = set()
    results: list[dict] = []

    for query in queries:
        try:
            raw = DDGS().text(query, max_results=10) or []
        except Exception as exc:
            logger.warning("DuckDuckGo query failed: %s â€” %s", query, exc)
            continue

        for r in raw:
            url = r.get("href", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = r.get("title", "").strip()
            snippet = r.get("body", "").strip()

            # Guess source from URL
            source = "web"
            for site in ("linkedin.com", "indeed.com", "glassdoor.com", "monster.com"):
                if site in url:
                    source = site.split(".")[0]
                    break

            # Try to extract company from title  ("Role at Company - Site")
            company = ""
            for sep in (" at ", " - ", " | "):
                if sep in title:
                    parts = title.split(sep)
                    if len(parts) >= 2:
                        company = parts[1].strip()
                    break

            results.append({
                "title": title,
                "company": company,
                "location": location,
                "url": url,
                "snippet": snippet,
                "source": source,
                "posting_type": "aggregator" if any(d in url for d in AGGREGATOR_DOMAINS) else "direct",
                "has_company": bool(company and company.lower() not in (
                    "unknown", "", "n/a", "various", "multiple", "confidential",
                    "not specified", "undisclosed",
                )),
            })

    return results


# ------------------------------------------------------------------
# Scrape contact emails from a job posting page
# ------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Collects readable text from HTML, skipping scripts/styles."""
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.texts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.texts.append(data)


def scrape_contact_emails(url: str) -> list[str]:
    """Fetch a job posting page and extract email addresses."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": USER_AGENT})
        if resp.status_code != 200:
            return []
    except Exception:
        return []

    parser = _TextExtractor()
    try:
        parser.feed(resp.text[:2_000_000])
        parser.close()
    except Exception:
        pass

    full_text = " ".join(parser.texts)
    # Also scan raw HTML for mailto: links etc.
    full_text += " " + resp.text

    raw_emails = set(EMAIL_RE.findall(full_text))

    # Filter out non-HR / common automated addresses
    filtered: list[str] = []
    for email in raw_emails:
        local = email.split("@")[0].lower()
        if local in FILTERED_EMAIL_PREFIXES:
            continue
        # Skip image-like false positives
        if email.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
            continue
        filtered.append(email)

    return sorted(set(filtered))


# ------------------------------------------------------------------
# Ollama helpers
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# LLM helpers (Gemini preferred, Ollama fallback)
# ------------------------------------------------------------------

def _ask_llm(system: str, user: str) -> str:
    """Send a chat request to the best available LLM.
    Uses Gemini API if configured (faster, works when hosted).
    Falls back to local Ollama if Gemini is not set up.
    """
    # Try Gemini first (faster, works remotely)
    if gemini_client.is_configured():
        try:
            return gemini_client.ask(system, user)
        except Exception as exc:
            logger.warning("Gemini API call failed: %s â€” falling back to Ollama", exc)

    # Fallback to local Ollama
    return _ask_ollama(system, user)


def _ask_ollama(system: str, user: str) -> str:
    """Send a chat request to Ollama and return the assistant's reply."""
    payload = {
        "model": config.CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": config.CONTEXT_WINDOW},
    }
    data = ollama_client._post("/api/chat", payload).json()
    return data.get("message", {}).get("content", "")


# ------------------------------------------------------------------
# CV analysis (with cache)
# ------------------------------------------------------------------

_cv_cache: dict = {"path": None, "mtime": None, "analysis": None}


def analyze_cv(cv_text: str, cv_path: str | Path | None = None) -> dict:
    """Use Ollama to extract skills, experience, and education from CV text.
    Caches the result keyed by file path + mtime so repeated calls are instant.
    """
    # Check cache
    if cv_path:
        p = Path(cv_path)
        if p.exists():
            mtime = p.stat().st_mtime
            if _cv_cache["path"] == str(p) and _cv_cache["mtime"] == mtime and _cv_cache["analysis"]:
                logger.info("CV analysis cache hit for %s", p.name)
                return _cv_cache["analysis"]

    system = (
        "You are a CV/resume analyser. Extract structured information from the CV text. "
        "Return ONLY valid JSON with these keys: "
        '"skills" (list of strings), "experience" (list of objects with "title", "company", "duration", "description"), '
        '"education" (list of objects with "degree", "institution", "year"), '
        '"summary" (string, 2-3 sentence professional summary). '
        "No markdown fences, just raw JSON."
    )
    answer = _ask_llm(system, cv_text[:6000])  # cap to fit context

    # Try to parse JSON from the response
    try:
        result = json.loads(answer)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown fences
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", answer, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if result is None:
        result = {
            "skills": [],
            "experience": [],
            "education": [],
            "summary": answer[:500],
            "raw": answer,
        }

    # Update cache
    if cv_path:
        p = Path(cv_path)
        _cv_cache["path"] = str(p)
        _cv_cache["mtime"] = p.stat().st_mtime if p.exists() else None
        _cv_cache["analysis"] = result

    return result


# ------------------------------------------------------------------
# Job matching (BATCHED â€” single Ollama call for all jobs)
# ------------------------------------------------------------------

def _parse_json_lenient(text: str):
    """Try to parse JSON from text, handling markdown fences and arrays."""
    text = text.strip()
    # Remove markdown fences
    fenced = re.search(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON array or object
    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def match_jobs(cv_analysis: dict, jobs: list[dict]) -> list[dict]:
    """Score ALL jobs in a SINGLE Ollama call. Much faster than one-per-job."""
    if not jobs:
        return []

    skills = ", ".join(cv_analysis.get("skills", []))[:300]
    experience_titles = ", ".join(
        e.get("title", "") for e in cv_analysis.get("experience", [])
    )[:200]
    summary = cv_analysis.get("summary", "")[:200]

    # Build a compact job list
    job_lines = []
    for i, job in enumerate(jobs):
        job_lines.append(
            f"{i+1}. \"{job.get('title', '')[:60]}\" at \"{job.get('company', '')[:40]}\" "
            f"({job.get('location', '')[:30]}): {job.get('snippet', '')[:80]}"
        )
    jobs_block = "\n".join(job_lines)

    system = (
        "You are a job-matching assistant. Score how well a candidate matches EACH job "
        "from 0 (no match) to 100 (perfect match). "
        "Return ONLY a valid JSON array: [{\"id\":1,\"score\":75,\"reason\":\"one sentence\"}, ...]. "
        "One entry per job. No markdown fences. No extra text."
    )
    user_msg = (
        f"CANDIDATE: Skills: {skills}. Experience: {experience_titles}. Summary: {summary}\n\n"
        f"JOBS:\n{jobs_block}"
    )

    try:
        answer = _ask_llm(system, user_msg)
        parsed = _parse_json_lenient(answer)

        if isinstance(parsed, list):
            # Map scores back to jobs
            score_map = {}
            for entry in parsed:
                idx = entry.get("id", 0) - 1  # 1-indexed to 0-indexed
                if 0 <= idx < len(jobs):
                    score_map[idx] = {
                        "score": int(entry.get("score", 0)),
                        "reason": entry.get("reason", ""),
                    }
            scored = []
            for i, job in enumerate(jobs):
                job_copy = {**job}
                if i in score_map:
                    job_copy["match_score"] = score_map[i]["score"]
                    job_copy["match_reason"] = score_map[i]["reason"]
                else:
                    job_copy["match_score"] = 0
                    job_copy["match_reason"] = "Not scored"
                scored.append(job_copy)
        elif isinstance(parsed, dict):
            # Single job response â€” wrap it
            scored = [{**jobs[0], "match_score": int(parsed.get("score", 0)),
                       "match_reason": parsed.get("reason", "")}]
            scored += [{**j, "match_score": 0, "match_reason": "Not scored"} for j in jobs[1:]]
        else:
            raise ValueError("Unexpected response format")

    except Exception as exc:
        logger.warning("Batch match failed: %s â€” falling back to zero scores", exc)
        scored = [{**j, "match_score": 0, "match_reason": "Scoring failed"} for j in jobs]

    scored.sort(key=lambda j: j.get("match_score", 0), reverse=True)
    return scored


# ------------------------------------------------------------------
# Email drafting (BATCHED â€” single Ollama call for all jobs)
# ------------------------------------------------------------------

def draft_email(job: dict, cv_analysis: dict, sender_email: str) -> dict:
    """Draft a cover email for a single job. Used as fallback."""
    return draft_emails_batch([job], cv_analysis, sender_email)[0]


def draft_emails_batch(jobs: list[dict], cv_analysis: dict, sender_email: str) -> list[dict]:
    """Draft cover emails for ALL jobs in a SINGLE Ollama call."""
    if not jobs:
        return []

    skills = ", ".join(cv_analysis.get("skills", []))[:300]
    summary = cv_analysis.get("summary", "")[:200]

    # Build compact job list
    job_lines = []
    for i, job in enumerate(jobs):
        job_lines.append(
            f"{i+1}. Title: \"{job.get('title', '')[:60]}\" | "
            f"Company: \"{job.get('company', '')[:40]}\" | "
            f"URL: {job.get('url', '')[:60]}"
        )
    jobs_block = "\n".join(job_lines)

    system = (
        "You draft professional cover emails for job applications. "
        "Given a list of jobs and a candidate profile, draft a SHORT email for EACH job. "
        "Return ONLY a valid JSON array: "
        "[{\"id\":1,\"subject\":\"...\",\"body\":\"...\"}, ...]. "
        "RULES for each email body: "
        "- Write ONLY 3-5 sentences: greeting, why you are a good fit, call to action, sign-off. "
        "- Do NOT include any job listing metadata (URL, portal link, experience range, salary/compensation, location type). "
        "- Do NOT include portfolio/LinkedIn URLs. "
        "- Do NOT repeat the job title or company in a structured format. "
        "- Keep it conversational and professional like a real human wrote it. "
        "- Subject should be simple like 'Application for [Role] at [Company]'. "
        "No markdown fences. No extra text."
    )
    user_msg = (
        f"MY PROFILE: Skills: {skills}. Summary: {summary}. Email: {sender_email}\n\n"
        f"JOBS:\n{jobs_block}\n\n"
        f"Draft {len(jobs)} emails, one per job."
    )

    try:
        answer = _ask_llm(system, user_msg)
        parsed = _parse_json_lenient(answer)

        drafts = []
        if isinstance(parsed, list):
            draft_map = {}
            for entry in parsed:
                idx = entry.get("id", 0) - 1
                if 0 <= idx < len(jobs):
                    draft_map[idx] = {
                        "subject": entry.get("subject", f"Application for {jobs[idx].get('title', 'Position')}"),
                        "body": entry.get("body", ""),
                    }
            for i, job in enumerate(jobs):
                contact = job.get("contact_email", "")
                if i in draft_map:
                    drafts.append({"to": contact, **draft_map[i]})
                else:
                    drafts.append({
                        "to": contact,
                        "subject": f"Application for {job.get('title', 'Position')}",
                        "body": f"Dear Hiring Manager,\n\nI am writing to express my interest in the {job.get('title', '')} position at {job.get('company', 'your company')}.\n\nMy skills include {skills[:100]}.\n\nBest regards",
                    })
        else:
            # Fallback: generate generic drafts
            for job in jobs:
                drafts.append({
                    "to": job.get("contact_email", ""),
                    "subject": f"Application for {job.get('title', 'Position')}",
                    "body": f"Dear Hiring Manager,\n\nI am writing to express my interest in the {job.get('title', '')} position at {job.get('company', 'your company')}.\n\nMy skills include {skills[:100]}.\n\nBest regards",
                })

    except Exception as exc:
        logger.warning("Batch draft failed: %s â€” using generic drafts", exc)
        drafts = []
        for job in jobs:
            drafts.append({
                "to": job.get("contact_email", ""),
                "subject": f"Application for {job.get('title', 'Position')}",
                "body": f"Dear Hiring Manager,\n\nI am writing to express my interest in the {job.get('title', '')} position.\n\nBest regards",
            })

    return drafts
    return {
        "to": contact_email,
        "subject": parsed.get("subject", f"Application for {job.get('title', 'Position')}"),
        "body": parsed.get("body", answer[:2000]),
    }


# ------------------------------------------------------------------
# Send application
# ------------------------------------------------------------------

def send_application(to: str, subject: str, body: str, cv_path: Path) -> dict:
    """Send a job application email with the CV attached."""
    return email_client.send_email_with_attachment(
        to=to,
        subject=subject,
        body=body,
        attachment_path=str(cv_path),
        attachment_name=cv_path.name,
    )


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------

def _ensure_dirs() -> None:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def save_application(job: dict, email_draft: dict) -> None:
    """Append a sent application to the history file."""
    _ensure_dirs()
    apps = load_applications()
    apps.append({
        "job": job,
        "email": email_draft,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    })
    APPLICATIONS_FILE.write_text(json.dumps(apps, indent=2, ensure_ascii=False), encoding="utf-8")


def load_applications() -> list[dict]:
    """Read the application history."""
    if not APPLICATIONS_FILE.exists():
        return []
    try:
        return json.loads(APPLICATIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_config(cfg: dict) -> None:
    """Save search preferences."""
    _ensure_dirs()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config() -> dict:
    """Load search preferences."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ------------------------------------------------------------------
# Analytics
# ------------------------------------------------------------------

def get_analytics() -> dict:
    """Compute analytics from the application history."""
    apps = load_applications()
    total = len(apps)
    if total == 0:
        return {
            "total": 0, "sent": 0, "failed": 0,
            "this_week": 0, "this_month": 0,
            "by_company": [], "by_role": [],
            "timeline": [],
        }

    sent = sum(1 for a in apps if a.get("send_status") != "failed")
    failed = total - sent

    # Time-based counts
    now = datetime.now(timezone.utc)
    this_week = 0
    this_month = 0
    for a in apps:
        sent_at = a.get("sent_at", "")
        if not sent_at:
            continue
        try:
            dt = datetime.fromisoformat(sent_at)
            delta = now - dt
            if delta.days <= 7:
                this_week += 1
            if delta.days <= 30:
                this_month += 1
        except (ValueError, TypeError):
            pass

    # Group by company
    company_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for a in apps:
        job = a.get("job", {})
        company = job.get("company", "Unknown")
        title = job.get("title", "Unknown")
        company_counts[company] = company_counts.get(company, 0) + 1
        role_counts[title] = role_counts.get(title, 0) + 1

    by_company = sorted(
        [{"name": k, "count": v} for k, v in company_counts.items()],
        key=lambda x: x["count"], reverse=True,
    )[:10]

    by_role = sorted(
        [{"name": k, "count": v} for k, v in role_counts.items()],
        key=lambda x: x["count"], reverse=True,
    )[:10]

    # Daily timeline (last 30 days)
    day_counts: dict[str, int] = {}
    for a in apps:
        sent_at = a.get("sent_at", "")
        if not sent_at:
            continue
        try:
            dt = datetime.fromisoformat(sent_at)
            day = dt.strftime("%Y-%m-%d")
            day_counts[day] = day_counts.get(day, 0) + 1
        except (ValueError, TypeError):
            pass
    timeline = sorted(
        [{"date": k, "count": v} for k, v in day_counts.items()],
        key=lambda x: x["date"],
    )[-30:]

    return {
        "total": total,
        "sent": sent,
        "failed": failed,
        "this_week": this_week,
        "this_month": this_month,
        "by_company": by_company,
        "by_role": by_role,
        "timeline": timeline,
    }


# ------------------------------------------------------------------
# Presets CRUD
# ------------------------------------------------------------------

def load_presets() -> list[dict]:
    """Load saved search presets."""
    if not PRESETS_FILE.exists():
        return []
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_presets(presets: list[dict]) -> None:
    """Write presets list to disk."""
    _ensure_dirs()
    PRESETS_FILE.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")


def add_preset(preset: dict) -> dict:
    """Add a new preset; returns it with an id and created_at."""
    import uuid
    preset["id"] = uuid.uuid4().hex[:12]
    preset.setdefault("active_for_runner", False)
    preset["created_at"] = datetime.now(timezone.utc).isoformat()
    presets = load_presets()
    presets.append(preset)
    save_presets(presets)
    return preset


def update_preset(preset_id: str, updates: dict) -> dict | None:
    """Update a preset by id. Returns the updated preset or None."""
    presets = load_presets()
    for p in presets:
        if p.get("id") == preset_id:
            p.update(updates)
            save_presets(presets)
            return p
    return None


def delete_preset(preset_id: str) -> bool:
    """Delete a preset by id. Returns True if found and deleted."""
    presets = load_presets()
    new = [p for p in presets if p.get("id") != preset_id]
    if len(new) == len(presets):
        return False
    save_presets(new)
    return True


# ------------------------------------------------------------------
# Daily Runner config & logic
# ------------------------------------------------------------------

def load_runner_config() -> dict:
    """Load daily runner configuration."""
    defaults = {
        "enabled": False,
        "run_time": "09:00",
        "min_match_score": 50,
        "to_email": "abyjohny25@gmail.com",
        "last_run": None,
        "last_run_date": None,
    }
    if not RUNNER_FILE.exists():
        return defaults
    try:
        data = json.loads(RUNNER_FILE.read_text(encoding="utf-8"))
        defaults.update(data)
        return defaults
    except (json.JSONDecodeError, OSError):
        return defaults


def save_runner_config(cfg: dict) -> None:
    """Save daily runner configuration."""
    _ensure_dirs()
    RUNNER_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def run_daily_scan() -> dict:
    """Run a full scan for all active presets. Returns results summary."""
    presets = load_presets()
    active = [p for p in presets if p.get("active_for_runner")]
    if not active:
        return {"status": "no_active_presets", "presets_run": 0, "results": []}

    # Get CV text if available
    cv_text = None
    cv_analysis = None
    cv_files = []
    if CV_DIR.exists():
        cv_files = sorted(
            [f for f in CV_DIR.iterdir() if f.suffix.lower() in (".pdf", ".docx", ".txt")],
            key=lambda f: f.stat().st_mtime, reverse=True,
        )
    if cv_files:
        try:
            cv_text = extract_cv_text(cv_files[0])
            cv_analysis = analyze_cv(cv_text)
        except Exception as exc:
            logger.warning("Daily scan: failed to analyze CV: %s", exc)

    all_results: list[dict] = []
    for preset in active:
        try:
            jobs = search_jobs(
                role=preset.get("role", ""),
                country=preset.get("country", ""),
                state=preset.get("state", ""),
                date_from=preset.get("date_from", ""),
                date_to=preset.get("date_to", ""),
                experience=preset.get("experience", ""),
            )
            # Match against CV if available
            if cv_analysis and jobs:
                jobs = match_jobs(cv_analysis, jobs)

            # Mark which jobs are eligible for auto-application
            # (only direct postings with verified company details)
            for j in jobs:
                j["auto_eligible"] = (
                    j.get("posting_type") == "direct"
                    and j.get("has_company", False)
                )

            all_results.append({
                "preset_name": preset.get("name", preset.get("role", "Unknown")),
                "preset_id": preset.get("id", ""),
                "role": preset.get("role", ""),
                "location": f"{preset.get('state', '')} {preset.get('country', '')}".strip(),
                "experience": preset.get("experience", ""),
                "jobs": jobs,
                "job_count": len(jobs),
            })
        except Exception as exc:
            logger.warning("Daily scan: preset %s failed: %s", preset.get("name", "?"), exc)
            all_results.append({
                "preset_name": preset.get("name", "Unknown"),
                "preset_id": preset.get("id", ""),
                "error": str(exc),
                "jobs": [],
                "job_count": 0,
            })

    summary = {
        "status": "completed",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "presets_run": len(active),
        "total_jobs": sum(r["job_count"] for r in all_results),
        "results": all_results,
    }

    # Save results
    _ensure_dirs()
    DAILY_RESULTS_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Update runner last_run
    runner_cfg = load_runner_config()
    runner_cfg["last_run"] = summary["run_at"]
    runner_cfg["last_run_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    save_runner_config(runner_cfg)

    return summary


def load_daily_results() -> dict:
    """Load the latest daily scan results."""
    if not DAILY_RESULTS_FILE.exists():
        return {}
    try:
        return json.loads(DAILY_RESULTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def send_daily_summary(results: dict, to_email: str) -> dict:
    """Send an HTML summary email of the daily scan results."""
    total_jobs = results.get("total_jobs", 0)
    presets_run = results.get("presets_run", 0)
    run_at = results.get("run_at", "")

    # Build HTML email body
    html_parts = [
        "<h2 style='color:#FF9B6B;'>ðŸ” AJP Daily Job Scan Results</h2>",
        f"<p><strong>Scan time:</strong> {run_at}<br>",
        f"<strong>Presets scanned:</strong> {presets_run}<br>",
        f"<strong>Total jobs found:</strong> {total_jobs}</p>",
        "<hr style='border:1px solid #333;'>",
    ]

    for preset_result in results.get("results", []):
        name = preset_result.get("preset_name", "Unknown")
        jobs = preset_result.get("jobs", [])
        html_parts.append(f"<h3>ðŸ“‹ {name} ({len(jobs)} jobs)</h3>")
        if preset_result.get("error"):
            html_parts.append(f"<p style='color:red;'>Error: {preset_result['error']}</p>")
            continue
        if not jobs:
            html_parts.append("<p style='color:#888;'>No jobs found for this preset.</p>")
            continue
        html_parts.append("<table style='width:100%;border-collapse:collapse;'>")
        html_parts.append(
            "<tr style='background:#1a1a2e;color:#eee;'>"
            "<th style='padding:8px;text-align:left;'>Title</th>"
            "<th style='padding:8px;text-align:left;'>Company</th>"
            "<th style='padding:8px;text-align:center;'>Match</th>"
            "</tr>"
        )
        for j in jobs[:20]:  # top 20
            score = j.get("match_score")
            score_str = f"{score}%" if score is not None else "â€”"
            color = "#5be3b0" if (score or 0) >= 70 else "#ffc86b" if (score or 0) >= 40 else "#ff6b8b"
            url = j.get("url", "#")
            html_parts.append(
                f"<tr style='border-bottom:1px solid #333;'>"
                f"<td style='padding:8px;'><a href='{url}' style='color:#FF9B6B;'>{j.get('title', 'Untitled')}</a></td>"
                f"<td style='padding:8px;color:#aaa;'>{j.get('company', 'Unknown')}</td>"
                f"<td style='padding:8px;text-align:center;color:{color};font-weight:bold;'>{score_str}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")

    html_parts.append(
        "<br><p style='color:#888;font-size:12px;'>Review and send applications at "
        "<a href='http://127.0.0.1:8080/jobs' style='color:#FF9B6B;'>AJP Job Hunter</a></p>"
    )

    subject = f"AJP Daily Job Scan: {total_jobs} jobs found ({presets_run} presets)"
    body = "\n".join(html_parts)

    try:
        result = email_client.send_email(to=to_email, subject=subject, body=body)
        return {"status": "sent", "result": result}
    except Exception as exc:
        logger.warning("Failed to send daily summary email: %s", exc)
        return {"status": "failed", "error": str(exc)}
