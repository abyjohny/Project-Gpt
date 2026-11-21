"""
job_engine.py - Job hunting engine (standalone, Gemini-only).
=============================================================

Searches for jobs, analyses CVs, matches jobs to skills,
drafts cover emails. No Ollama dependency.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests
from ddgs import DDGS

from gemini_client import ask as gemini_ask
from users import (
    get_user_cv_dir, get_user_cv_path, load_user_gemini_key,
    load_user_presets, load_user_runner, save_user_runner,
    save_user_daily_results, save_user_presets,
)

logger = logging.getLogger("job_engine")

EXPERIENCE_LEVELS = [
    "Entry Level", "Junior", "Mid-Level", "Senior", "Lead", "Manager",
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

# ------------------------------------------------------------------
# Strict Whitelist & Blacklist for Strict Job Search
# ------------------------------------------------------------------

VERIFIED_JOB_PLATFORMS = {
    # Major Global & Regional Job Boards / Aggregators
    "linkedin.com", "indeed.com", "naukri.com", "glassdoor.com", "glassdoor.co.in",
    "foundit.in", "monster.com", "shine.com", "instahyre.com", "hirist.com",
    "hirist.tech", "cutshort.io", "apna.co", "timesjobs.com", "wellfound.com",
    "angel.co", "workatastartup.com", "builtin.com", "dice.com", "ziprecruiter.com",
    "simplyhired.com", "careerbuilder.com", "jooble.org", "adzuna.com", "adzuna.in",
    "reed.co.uk", "totaljobs.com", "cwjobs.co.uk", "seek.com.au", "remoteok.com",
    "weworkremotely.com", "otta.com", "levels.fyi", "flexjobs.com", "hired.com",
    "catchwork.jobs", "jobrapido.com",

    # Applicant Tracking Systems (ATS / Direct Application Portals)
    "greenhouse.io", "lever.co", "myworkdayjobs.com", "ashbyhq.com",
    "bamboohr.com", "smartrecruiters.com", "workable.com", "recruitee.com",
    "breezy.hr", "jobvite.com", "taleo.net", "jobs2web.com", "successfactors.com",
    "rippling-ats.com", "teamtailor.com", "pinpointhq.com", "personio.com",
    "personio.de", "applytojob.com",
}

BLOCKED_NON_JOB_DOMAINS = {
    "wikipedia.org", "w3schools.com", "geeksforgeeks.org", "tutorialspoint.com",
    "python.org", "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    "youtube.com", "reddit.com", "quora.com", "medium.com", "substack.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "stackoverflow.com", "stackexchange.com", "dev.to", "hashnode.com",
    "coursera.org", "udemy.com", "edx.org", "freecodecamp.org", "javatpoint.com",
    "docs.python.org", "pypi.org", "grokipedia.com", "britannica.com",
    "investopedia.com", "khanacademy.org", "codecademy.com", "datacamp.com",
    "simplilearn.com", "programiz.com", "baeldung.com", "a-z-animals.com", "iere.org",
    "timesofindia.indiatimes.com", "ndtv.com", "thehindu.com", "hindustantimes.com",
    "news18.com", "indiatoday.in", "indianexpress.com", "techcrunch.com", "theverge.com",
}

AD_AND_TRACKER_DOMAINS = {
    "bing.com", "googleadservices.com", "doubleclick.net", "adclick", "clickserve",
    "advertising.com", "outbrain.com", "taboola.com",
}

BLOCKED_TITLE_PATTERNS = {
    "(programming language)", "tutorial", "tutorials", "documentation",
    "download python", "what is", "cheat sheet", "cheatsheet", "roadmap",
    "interview questions", "learn python", "guide for beginners", "syntax",
    "compiler", "interpreter", "ide online", "ide, editor", "introduction to",
}

CAREER_PATH_INDICATORS = (
    "/careers", "/career", "/jobs", "/job/", "/openings", "/opening",
    "/vacancies", "/vacancy", "/positions", "/position", "/join-us",
    "/work-with-us", "/apply",
)

JOB_INTENT_KEYWORDS = {
    "job", "jobs", "hiring", "vacancy", "vacancies", "career", "careers",
    "opening", "openings", "role", "roles", "position", "positions",
    "recruitment", "apply", "developer", "engineer", "salary", "full-time",
    "part-time", "contract", "remote", "fresher", "intern", "experienced",
    "walkin", "walk-in", "work from home", "wfh", "talent", "opportunity",
}


def classify_and_filter_job(url: str, title: str, snippet: str) -> dict | None:
    """
    STRICT job search filter:
    Returns job metadata dict ONLY if the URL is a verified job board, ATS, or direct company career page.
    Returns None if the URL is an encyclopedia, blog, ad redirect, tutorial, or non-job page.
    """
    if not url or not url.startswith("http"):
        return None

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    combined_text = f"{title.lower()} {snippet.lower()}"

    # 1. Block ad trackers, click redirects, and affiliate wrappers
    if any(ad in host or ad in path for ad in AD_AND_TRACKER_DOMAINS):
        return None

    # 2. Block known non-job domains (encyclopedias, tutorials, news, social)
    if any(blocked in host for blocked in BLOCKED_NON_JOB_DOMAINS):
        return None

    # 3. Block tutorial/documentation/compiler titles
    if any(w in title.lower() for w in BLOCKED_TITLE_PATTERNS):
        return None

    # 4. Check if it's a Verified Job Platform or ATS
    matched_platform = None
    for p in VERIFIED_JOB_PLATFORMS:
        if p in host:
            matched_platform = p
            break

    if matched_platform:
        is_ats = any(ats in host for ats in (
            "greenhouse.io", "lever.co", "myworkdayjobs.com", "ashbyhq.com",
            "bamboohr.com", "smartrecruiters.com", "workable.com", "recruitee.com",
            "breezy.hr", "jobvite.com", "taleo.net", "jobs2web.com", "successfactors.com",
            "rippling-ats.com", "teamtailor.com", "pinpointhq.com", "personio.com",
            "applytojob.com",
        ))
        source_name = matched_platform.split(".")[0].capitalize()
        return {
            "source": f"{source_name} (ATS)" if is_ats else source_name,
            "posting_type": "direct" if is_ats else "aggregator",
        }

    # 5. Check if it's an authentic Company Career Page
    is_career_subdomain = host.startswith("careers.") or host.startswith("jobs.")
    is_career_path = any(p in path for p in CAREER_PATH_INDICATORS)

    if (is_career_subdomain or is_career_path) and any(k in combined_text for k in JOB_INTENT_KEYWORDS):
        clean_domain = host.replace("www.", "").replace("careers.", "").replace("jobs.", "")
        company_name = clean_domain.split(".")[0].capitalize()
        return {
            "source": f"{company_name} Careers",
            "posting_type": "direct",
        }

    # If it does not belong to a verified job site or company career page: REJECT
    return None


def extract_company_name(url: str, title: str) -> str:
    """Extract clean company name from ATS URLs, career subdomains, or title separators."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/").split("/")

    # Check ATS path/subdomain structures
    if "greenhouse.io" in host and path:
        return path[0].replace("-", " ").title()
    if "lever.co" in host and path:
        return path[0].replace("-", " ").title()
    if "ashbyhq.com" in host and path:
        return path[0].replace("-", " ").title()
    if "workable.com" in host and path:
        return path[0].replace("-", " ").title()
    if "myworkdayjobs.com" in host:
        return host.split(".")[0].replace("-", " ").title()

    # Title parsing: "Role at Company", "Role - Company", "Role | Company"
    for sep in (" at ", " | ", " - "):
        if sep in title:
            parts = title.split(sep)
            if len(parts) >= 2:
                candidate = parts[1].strip()
                if candidate.lower() not in (
                    "linkedin", "indeed", "glassdoor", "naukri", "greenhouse software",
                    "lever", "workday", "ashby", "jobs", "careers",
                ):
                    return candidate
            if "hiring" in parts[0].lower():
                cand = parts[0].replace("hiring", "").strip()
                if cand:
                    return cand

    # Company careers subdomain
    if host.startswith("careers.") or host.startswith("jobs."):
        clean = host.replace("careers.", "").replace("jobs.", "").replace("www.", "")
        return clean.split(".")[0].title()

    return ""


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
    """Search DuckDuckGo for job postings. Returns strictly verified job postings only."""
    location = f"{state} {country}".strip() if state else country
    exp_tag = f" {experience}" if experience else ""

    # Normalize role: if user entered only a technology/skill (e.g. 'python', 'react'), ensure job intent
    role_clean = role.strip()
    role_lower = role_clean.lower()
    role_nouns = (
        "developer", "engineer", "programmer", "architect", "lead", "specialist",
        "analyst", "manager", "designer", "consultant", "scientist", "tester", "intern",
    )
    if not any(noun in role_lower for noun in role_nouns) and "job" not in role_lower:
        search_role = f"{role_clean} Developer"
    else:
        search_role = role_clean

    queries = [
        # Verified Job Boards
        f"{search_role}{exp_tag} jobs in {location} site:linkedin.com/jobs",
        f"{search_role}{exp_tag} jobs {location} site:indeed.com",
        f"{search_role}{exp_tag} jobs {location} site:naukri.com" if "india" in location.lower() else f"{search_role}{exp_tag} jobs {location} site:glassdoor.com",
        # ATS & Direct Application Portals
        f"{search_role}{exp_tag} {location} site:boards.greenhouse.io",
        f"{search_role}{exp_tag} {location} site:jobs.lever.co",
        f"{search_role}{exp_tag} {location} site:myworkdayjobs.com",
        f"{search_role}{exp_tag} {location} site:jobs.ashbyhq.com",
        # Broad job openings
        f"{search_role}{exp_tag} jobs {location}",
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
            logger.debug("DuckDuckGo query returned no items: %s (%s)", query, exc)
            continue

        for r in raw:
            url = r.get("href", "")
            if not url or url in seen_urls:
                continue

            title = r.get("title", "").strip()
            snippet = r.get("body", "").strip()

            # STRICT FILTER: strictly accept ONLY verified job boards, ATS, or company career pages
            job_meta = classify_and_filter_job(url, title, snippet)
            if not job_meta:
                continue

            seen_urls.add(url)
            company = extract_company_name(url, title)

            results.append({
                "title": title,
                "company": company,
                "location": location,
                "url": url,
                "snippet": snippet,
                "source": job_meta["source"],
                "posting_type": job_meta["posting_type"],
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

    full_text = " ".join(parser.texts) + " " + resp.text
    raw_emails = set(EMAIL_RE.findall(full_text))

    filtered: list[str] = []
    for email in raw_emails:
        local = email.split("@")[0].lower()
        if local in FILTERED_EMAIL_PREFIXES:
            continue
        if email.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
            continue
        filtered.append(email)

    return sorted(set(filtered))


# ------------------------------------------------------------------
# LLM helper (Gemini only)
# ------------------------------------------------------------------

def _ask_llm(system: str, user: str, api_key: str | None = None) -> str:
    """Send a chat request to Gemini."""
    return gemini_ask(system, user, api_key=api_key)


# ------------------------------------------------------------------
# CV analysis (with per-user cache in memory)
# ------------------------------------------------------------------

_cv_cache: dict = {}  # keyed by user_id


def analyze_cv(cv_text: str, cv_path: str | Path | None = None,
               user_id: str = "", api_key: str | None = None) -> dict:
    """Use Gemini to extract skills, experience, and education from CV text."""
    if user_id and cv_path:
        p = Path(cv_path)
        if p.exists():
            mtime = p.stat().st_mtime
            cached = _cv_cache.get(user_id)
            if cached and cached.get("path") == str(p) and cached.get("mtime") == mtime:
                logger.info("CV analysis cache hit for user %s", user_id)
                return cached["analysis"]

    system = (
        "You are a CV/resume analyser. Extract structured information from the CV text. "
        "Return ONLY valid JSON with these keys: "
        '"skills" (list of strings), "experience" (list of objects with "title", "company", "duration", "description"), '
        '"education" (list of objects with "degree", "institution", "year"), '
        '"summary" (string, 2-3 sentence professional summary). '
        "No markdown fences, just raw JSON."
    )
    answer = _ask_llm(system, cv_text[:6000], api_key)

    try:
        result = json.loads(answer)
    except json.JSONDecodeError:
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
            "skills": [], "experience": [], "education": [],
            "summary": answer[:500], "raw": answer,
        }

    if user_id and cv_path:
        p = Path(cv_path)
        _cv_cache[user_id] = {
            "path": str(p),
            "mtime": p.stat().st_mtime if p.exists() else None,
            "analysis": result,
        }

    return result


# ------------------------------------------------------------------
# JSON parsing helper
# ------------------------------------------------------------------

def _parse_json_lenient(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


# ------------------------------------------------------------------
# Job matching (BATCHED)
# ------------------------------------------------------------------

def match_jobs(cv_analysis: dict, jobs: list[dict],
               api_key: str | None = None) -> list[dict]:
    """Score ALL jobs in a SINGLE Gemini call."""
    if not jobs:
        return []

    skills = ", ".join(cv_analysis.get("skills", []))[:300]
    experience_titles = ", ".join(
        e.get("title", "") for e in cv_analysis.get("experience", [])
    )[:200]
    summary = cv_analysis.get("summary", "")[:200]

    job_lines = []
    for i, job in enumerate(jobs):
        job_lines.append(
            f'{i+1}. "{job.get("title", "")[:60]}" at "{job.get("company", "")[:40]}" '
            f'({job.get("location", "")[:30]}): {job.get("snippet", "")[:80]}'
        )
    jobs_block = "\n".join(job_lines)

    system = (
        "You are a job-matching assistant. Score how well a candidate matches EACH job "
        "from 0 (no match) to 100 (perfect match). "
        'Return ONLY a valid JSON array: [{"id":1,"score":75,"reason":"one sentence"}, ...]. '
        "One entry per job. No markdown fences. No extra text."
    )
    user_msg = (
        f"CANDIDATE: Skills: {skills}. Experience: {experience_titles}. Summary: {summary}\n\n"
        f"JOBS:\n{jobs_block}"
    )

    try:
        answer = _ask_llm(system, user_msg, api_key)
        parsed = _parse_json_lenient(answer)

        if isinstance(parsed, list):
            score_map = {}
            for entry in parsed:
                idx = entry.get("id", 0) - 1
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
        else:
            scored = [{**j, "match_score": 0, "match_reason": "Scoring failed"} for j in jobs]

    except Exception as exc:
        logger.warning("Batch match failed: %s — falling back to zero scores", exc)
        scored = [{**j, "match_score": 0, "match_reason": "Scoring failed"} for j in jobs]

    scored.sort(key=lambda j: j.get("match_score", 0), reverse=True)
    return scored


# ------------------------------------------------------------------
# Email drafting (BATCHED)
# ------------------------------------------------------------------

def draft_emails_batch(jobs: list[dict], cv_analysis: dict,
                       sender_email: str, api_key: str | None = None) -> list[dict]:
    """Draft cover emails for ALL jobs in a SINGLE Gemini call."""
    if not jobs:
        return []

    skills = ", ".join(cv_analysis.get("skills", []))[:300]
    summary = cv_analysis.get("summary", "")[:200]

    job_lines = []
    for i, job in enumerate(jobs):
        job_lines.append(
            f'{i+1}. Title: "{job.get("title", "")[:60]}" | '
            f'Company: "{job.get("company", "")[:40]}" | '
            f'URL: {job.get("url", "")[:60]}'
        )
    jobs_block = "\n".join(job_lines)

    system = (
        "You draft professional cover emails for job applications. "
        "Given a list of jobs and a candidate profile, draft a SHORT email for EACH job. "
        "Return ONLY a valid JSON array: "
        '[{"id":1,"subject":"...","body":"..."}, ...]. '
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
        answer = _ask_llm(system, user_msg, api_key)
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
            for job in jobs:
                drafts.append({
                    "to": job.get("contact_email", ""),
                    "subject": f"Application for {job.get('title', 'Position')}",
                    "body": f"Dear Hiring Manager,\n\nI am writing to express my interest in the {job.get('title', '')} position at {job.get('company', 'your company')}.\n\nMy skills include {skills[:100]}.\n\nBest regards",
                })

    except Exception as exc:
        logger.warning("Batch draft failed: %s — using generic drafts", exc)
        drafts = []
        for job in jobs:
            drafts.append({
                "to": job.get("contact_email", ""),
                "subject": f"Application for {job.get('title', 'Position')}",
                "body": f"Dear Hiring Manager,\n\nI am writing to express my interest in the {job.get('title', '')} position.\n\nBest regards",
            })

    return drafts


# ------------------------------------------------------------------
# Presets helpers
# ------------------------------------------------------------------

def add_preset(user_id: str, preset: dict) -> dict:
    preset["id"] = uuid.uuid4().hex[:12]
    preset.setdefault("active_for_runner", False)
    preset["created_at"] = datetime.now(timezone.utc).isoformat()
    presets = load_user_presets(user_id)
    presets.append(preset)
    save_user_presets(user_id, presets)
    return preset


def update_preset(user_id: str, preset_id: str, updates: dict) -> dict | None:
    presets = load_user_presets(user_id)
    for p in presets:
        if p.get("id") == preset_id:
            p.update(updates)
            save_user_presets(user_id, presets)
            return p
    return None


def delete_preset(user_id: str, preset_id: str) -> bool:
    presets = load_user_presets(user_id)
    new = [p for p in presets if p.get("id") != preset_id]
    if len(new) == len(presets):
        return False
    save_user_presets(user_id, new)
    return True


# ------------------------------------------------------------------
# Analytics
# ------------------------------------------------------------------

def get_analytics(user_id: str) -> dict:
    from users import load_user_applications
    apps = load_user_applications(user_id)
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

    company_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    day_counts: dict[str, int] = {}
    for a in apps:
        job = a.get("job", {})
        company = job.get("company", "Unknown")
        title = job.get("title", "Unknown")
        company_counts[company] = company_counts.get(company, 0) + 1
        role_counts[title] = role_counts.get(title, 0) + 1
        sent_at = a.get("sent_at", "")
        if sent_at:
            try:
                dt = datetime.fromisoformat(sent_at)
                day = dt.strftime("%Y-%m-%d")
                day_counts[day] = day_counts.get(day, 0) + 1
            except (ValueError, TypeError):
                pass

    by_company = sorted(
        [{"name": k, "count": v} for k, v in company_counts.items()],
        key=lambda x: x["count"], reverse=True,
    )[:10]
    by_role = sorted(
        [{"name": k, "count": v} for k, v in role_counts.items()],
        key=lambda x: x["count"], reverse=True,
    )[:10]
    timeline = sorted(
        [{"date": k, "count": v} for k, v in day_counts.items()],
        key=lambda x: x["date"],
    )[-30:]

    return {
        "total": total, "sent": sent, "failed": failed,
        "this_week": this_week, "this_month": this_month,
        "by_company": by_company, "by_role": by_role,
        "timeline": timeline,
    }


# ------------------------------------------------------------------
# Daily scan for a user
# ------------------------------------------------------------------

def run_daily_scan_for_user(user_id: str) -> dict:
    """Run a full scan for all active presets for a given user."""
    presets = load_user_presets(user_id)
    active = [p for p in presets if p.get("active_for_runner")]
    if not active:
        return {"status": "no_active_presets", "presets_run": 0, "results": []}

    api_key = load_user_gemini_key(user_id)

    cv_path = get_user_cv_path(user_id)
    cv_analysis = None
    if cv_path:
        try:
            cv_text = extract_cv_text(cv_path)
            cv_analysis = analyze_cv(cv_text, cv_path, user_id=user_id, api_key=api_key)
        except Exception as exc:
            logger.warning("Daily scan: failed to analyze CV for user %s: %s", user_id, exc)

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
            if cv_analysis and jobs:
                jobs = match_jobs(cv_analysis, jobs, api_key=api_key)

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

    save_user_daily_results(user_id, summary)

    runner_cfg = load_user_runner(user_id)
    runner_cfg["last_run"] = summary["run_at"]
    runner_cfg["last_run_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    save_user_runner(user_id, runner_cfg)

    return summary
