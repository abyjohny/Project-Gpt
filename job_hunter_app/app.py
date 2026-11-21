"""
app.py - Standalone Multi-Tenant Job Hunter SaaS.
==================================================

A lightweight FastAPI server for the Job Hunter service.
Users sign in with Google OAuth, connect their Gmail, and get
isolated job hunting with CV analysis, email drafting, and sending.

No Ollama, no PyTorch, no ChromaDB — Gemini API only.

Run:   python app.py
       then open http://127.0.0.1:8080
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auth
import job_engine
import email_sender
import users
from gemini_client import is_configured as gemini_is_configured, GEMINI_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"
HOST = os.environ.get("HOST", "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))

SESSION_COOKIE = "jh_session"


# ------------------------------------------------------------------
# Auth middleware helper
# ------------------------------------------------------------------

def get_current_user(request: Request) -> dict | None:
    """Extract user from session cookie. Returns None if not logged in."""
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return None
    return auth.validate_session_cookie(cookie)


def require_user(request: Request) -> dict:
    """Get user or raise 401."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated. Please sign in.")
    return user


# ------------------------------------------------------------------
# Daily runner background loop (multi-user)
# ------------------------------------------------------------------

def _daily_runner_loop() -> None:
    """Check every 60 seconds if it's time to run the daily job scan for each user."""
    logger.info("Multi-user daily runner loop started.")
    while True:
        try:
            time.sleep(60)
            user_ids = users.list_users_with_runner()
            if not user_ids:
                continue

            now = datetime.now()
            current_time = now.strftime("%H:%M")
            today = now.strftime("%Y-%m-%d")

            for user_id in user_ids:
                try:
                    cfg = users.load_user_runner(user_id)
                    if not cfg.get("enabled"):
                        continue
                    run_time = cfg.get("run_time", "09:00")
                    last_run_date = cfg.get("last_run_date")

                    if current_time == run_time and last_run_date != today:
                        logger.info("Daily runner triggered for user %s at %s", user_id, current_time)
                        try:
                            results = job_engine.run_daily_scan_for_user(user_id)
                            to_email = cfg.get("to_email", "")
                            if results.get("total_jobs", 0) > 0 and to_email:
                                # Send summary email
                                _send_daily_summary_email(user_id, results, to_email)
                            logger.info("Daily scan for user %s: %d jobs found",
                                       user_id, results.get("total_jobs", 0))
                        except Exception:
                            logger.error("Daily scan failed for user %s:\n%s",
                                        user_id, traceback.format_exc())
                except Exception:
                    logger.error("Runner check failed for user %s:\n%s",
                                user_id, traceback.format_exc())
        except Exception:
            logger.error("Daily runner loop error:\n%s", traceback.format_exc())


def _send_daily_summary_email(user_id: str, results: dict, to_email: str) -> dict:
    """Send an HTML summary email of the daily scan results."""
    total_jobs = results.get("total_jobs", 0)
    presets_run = results.get("presets_run", 0)
    run_at = results.get("run_at", "")

    html_parts = [
        "<div style='font-family:sans-serif;max-width:680px;margin:0 auto;color:#0F172A;'>",
        "<h2 style='color:#2563EB;margin-bottom:8px;'>Job Hunter — Daily Scan Results</h2>",
        f"<p style='color:#475569;margin-top:0;'>Scan time: {run_at} | Presets: {presets_run} | Jobs found: <strong>{total_jobs}</strong></p>",
        "<hr style='border:none;border-top:1px solid #E2E8F0;margin:16px 0;'>",
    ]

    for preset_result in results.get("results", []):
        name = preset_result.get("preset_name", "Unknown")
        jobs = preset_result.get("jobs", [])
        html_parts.append(f"<h3 style='color:#1E293B;margin-top:20px;'>{name} <span style='font-size:12px;color:#64748B;'>({len(jobs)} jobs)</span></h3>")
        if preset_result.get("error"):
            html_parts.append(f"<p style='color:#DC2626;'>Error: {preset_result['error']}</p>")
            continue
        if not jobs:
            html_parts.append("<p style='color:#64748B;'>No jobs found for this preset.</p>")
            continue
        html_parts.append("<table style='width:100%;border-collapse:collapse;margin-top:10px;font-size:14px;'>")
        html_parts.append(
            "<tr style='background:#F1F5F9;color:#334155;text-align:left;'>"
            "<th style='padding:8px 12px;border-bottom:1px solid #E2E8F0;'>Title</th>"
            "<th style='padding:8px 12px;border-bottom:1px solid #E2E8F0;'>Company</th>"
            "<th style='padding:8px 12px;border-bottom:1px solid #E2E8F0;text-align:center;'>Match</th>"
            "</tr>"
        )
        for j in jobs[:20]:
            score = j.get("match_score")
            score_str = f"{score}%" if score is not None else "—"
            color = "#16A34A" if (score or 0) >= 70 else "#D97706" if (score or 0) >= 40 else "#DC2626"
            url = j.get("url", "#")
            html_parts.append(
                f"<tr style='border-bottom:1px solid #F1F5F9;'>"
                f"<td style='padding:8px 12px;'><a href='{url}' style='color:#2563EB;text-decoration:none;'>{j.get('title', 'Untitled')}</a></td>"
                f"<td style='padding:8px 12px;color:#475569;'>{j.get('company', 'Unknown')}</td>"
                f"<td style='padding:8px 12px;text-align:center;color:{color};font-weight:bold;'>{score_str}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")
    html_parts.append("</div>")

    subject = f"Job Hunter Daily Scan: {total_jobs} jobs found ({presets_run} presets)"
    body = "\n".join(html_parts)

    try:
        result = email_sender.send_email(user_id, to_email, subject, body)
        return {"status": "sent", "result": result}
    except Exception as exc:
        logger.warning("Failed to send daily summary for user %s: %s", user_id, exc)
        return {"status": "failed", "error": str(exc)}


# ------------------------------------------------------------------
# App lifecycle
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_daily_runner_loop, daemon=True, name="daily-runner").start()
    yield


app = FastAPI(title="Job Hunter", lifespan=lifespan)

# Mount static files
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ------------------------------------------------------------------
# Landing page / Dashboard
# ------------------------------------------------------------------

def is_local_dev(request: Request) -> bool:
    """Return True only if running locally on localhost/127.0.0.1, not on Render."""
    if os.environ.get("RENDER"):
        return False
    host = request.headers.get("host", "").split(":")[0].lower()
    return host in ("127.0.0.1", "localhost", "0.0.0.0")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    user = get_current_user(request)
    if user:
        # Serve the dashboard
        dashboard_path = STATIC_DIR / "dashboard.html"
        if dashboard_path.exists():
            html = dashboard_path.read_text(encoding="utf-8")
            # Inject user data
            html = html.replace("{{USER_NAME}}", user.get("name", ""))
            html = html.replace("{{USER_EMAIL}}", user.get("email", ""))
            html = html.replace("{{USER_PICTURE}}", user.get("picture", ""))
            return html

    # Serve the landing page
    landing_path = STATIC_DIR / "landing.html"
    if landing_path.exists():
        html = landing_path.read_text(encoding="utf-8")
        if is_local_dev(request):
            dev_block = """
                <div class="auth-divider">
                    <span>or local test</span>
                </div>
                <div class="dev-card">
                    <div class="dev-card-header">
                        <span class="dev-badge"><i class="ph ph-terminal"></i> Local Dev Access</span>
                        <span class="dev-sub">Local environment only</span>
                    </div>
                    <div class="dev-info">
                        <div class="dev-user">Instant Test Account</div>
                        <div class="dev-sub">Auto-configures isolated workspace &amp; Gemini engine</div>
                    </div>
                    <a href="/auth/dev-login" class="btn-primary">
                        <span>Launch Workspace Demo</span>
                        <i class="ph ph-arrow-right"></i>
                    </a>
                </div>
            """
            html = html.replace("{{DEV_LOGIN_SECTION}}", dev_block)
        else:
            html = html.replace("{{DEV_LOGIN_SECTION}}", "")
        return html

    return "<h1>Job Hunter</h1><p><a href='/auth/login'>Sign in with Google</a></p>"


# ------------------------------------------------------------------
# Auth endpoints
# ------------------------------------------------------------------

@app.get("/auth/login")
async def auth_login(request: Request):
    if not auth.is_oauth_configured():
        if is_local_dev(request):
            return HTMLResponse(
                "<h2>Google OAuth not configured yet</h2>"
                "<p>To use Google Sign-In, set <code>GOOGLE_CLIENT_ID</code> and <code>GOOGLE_CLIENT_SECRET</code> in <code>.env</code>.</p>"
                "<p><a href='/auth/dev-login' style='display:inline-block;padding:10px 20px;background:#2563EB;color:#fff;text-decoration:none;border-radius:8px;font-weight:600;'>⚡ Enter Local Dev Test Mode</a></p>"
                "<p><a href='/'>← Back to Home</a></p>",
                status_code=200,
            )
        else:
            raise HTTPException(500, "Google OAuth is not configured on this server. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in the Render dashboard.")
    url = auth.get_auth_url(request)
    return RedirectResponse(url)


@app.get("/auth/dev-login")
async def auth_dev_login(request: Request):
    """Local dev login bypass for instant testing without OAuth configuration."""
    if not is_local_dev(request) and not os.environ.get("ALLOW_DEV_LOGIN"):
        raise HTTPException(403, "Dev login is disabled in cloud production.")

    user_id = "local_dev_user"
    email = "abyjohny25@gmail.com"
    name = "Aby Johny (Dev)"
    picture = "https://lh3.googleusercontent.com/a/default-user=s96-c"

    # Save profile
    users.save_user_profile(user_id, {
        "id": user_id,
        "email": email,
        "name": name,
        "picture": picture,
    })

    # Set runner email default if not set
    runner_cfg = users.load_user_runner(user_id)
    if not runner_cfg.get("to_email"):
        runner_cfg["to_email"] = email
        users.save_user_runner(user_id, runner_cfg)

    # Create session cookie
    session_data = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "picture": picture,
    }
    cookie = auth.create_session_cookie(session_data)

    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        cookie,
        max_age=auth.SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = "", error: str = ""):
    if error:
        return HTMLResponse(f"<h1>Authentication failed</h1><p>{error}</p><p><a href='/'>Go back</a></p>")
    if not code:
        return HTMLResponse("<h1>Missing authorization code</h1><p><a href='/'>Go back</a></p>")

    try:
        # Exchange code for tokens
        tokens = auth.exchange_code(code, request)
        access_token = tokens.get("access_token")
        if not access_token:
            raise ValueError("No access token received")

        # Get user info
        user_info = auth.get_user_info(access_token)
        user_id = user_info.get("id", "")
        email = user_info.get("email", "")
        name = user_info.get("name", "")
        picture = user_info.get("picture", "")

        if not user_id:
            raise ValueError("Could not get user ID from Google")

        # Save tokens and profile
        auth.save_user_tokens(user_id, tokens)
        users.save_user_profile(user_id, {
            "id": user_id,
            "email": email,
            "name": name,
            "picture": picture,
        })

        # Set runner email default if not set
        runner_cfg = users.load_user_runner(user_id)
        if not runner_cfg.get("to_email") and email:
            runner_cfg["to_email"] = email
            users.save_user_runner(user_id, runner_cfg)

        # Create session cookie
        session_data = {
            "user_id": user_id,
            "email": email,
            "name": name,
            "picture": picture,
        }
        cookie = auth.create_session_cookie(session_data)

        response = RedirectResponse("/", status_code=302)
        response.set_cookie(
            SESSION_COOKIE,
            cookie,
            max_age=auth.SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=not auth.BASE_URL.startswith("http://127.0.0.1"),
        )
        return response

    except Exception as exc:
        logger.error("OAuth callback failed: %s", exc)
        return HTMLResponse(
            f"<h1>Authentication failed</h1><p>{exc}</p><p><a href='/'>Try again</a></p>",
            status_code=500,
        )


@app.get("/auth/logout")
async def auth_logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/api/auth/me")
async def auth_me(request: Request):
    user = get_current_user(request)
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, **user}


# ------------------------------------------------------------------
# Job Hunter API (all require authentication)
# ------------------------------------------------------------------

class JobSearchRequest(BaseModel):
    role: str = Field(min_length=1, max_length=200)
    country: str = Field(min_length=1, max_length=100)
    state: str = ""
    date_from: str = ""
    date_to: str = ""
    experience: str = ""


@app.post("/api/jobs/search")
async def search_jobs(body: JobSearchRequest, request: Request):
    user = require_user(request)
    try:
        results = await asyncio.to_thread(
            job_engine.search_jobs, body.role, body.country,
            body.state, body.date_from, body.date_to, body.experience)
        return {"jobs": results}
    except Exception as exc:
        raise HTTPException(500, f"Job search failed: {exc}") from exc


@app.post("/api/jobs/cv")
async def upload_cv(request: Request, file: UploadFile = File(...)):
    user = require_user(request)
    user_id = user["user_id"]
    filename = Path(file.filename or "").name.strip()
    if not filename:
        raise HTTPException(400, "Invalid file name.")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt", ".doc"):
        raise HTTPException(415, "Only PDF, DOCX, DOC, and TXT files are supported for CVs.")
    cv_dir = users.get_user_cv_dir(user_id)
    save_path = cv_dir / filename
    content = await file.read()
    save_path.write_bytes(content)
    try:
        cv_text = await asyncio.to_thread(job_engine.extract_cv_text, save_path)
    except Exception as exc:
        raise HTTPException(500, f"Failed to read CV: {exc}") from exc
    return {"status": "ok", "filename": filename, "path": str(save_path), "preview": cv_text[:500]}


@app.get("/api/jobs/cv")
def get_current_cv(request: Request):
    user = require_user(request)
    cv_path = users.get_user_cv_path(user["user_id"])
    if not cv_path:
        return {"cv": None}
    return {"cv": {
        "filename": cv_path.name,
        "path": str(cv_path),
        "size": cv_path.stat().st_size,
        "modified": cv_path.stat().st_mtime,
    }}


class AnalyzeRequest(BaseModel):
    jobs: list[dict]


@app.post("/api/jobs/analyze")
async def analyze_jobs(body: AnalyzeRequest, request: Request):
    user = require_user(request)
    user_id = user["user_id"]
    api_key = users.load_user_gemini_key(user_id) or os.environ.get("GEMINI_API_KEY")
    cv_path = users.get_user_cv_path(user_id)
    if not cv_path:
        raise HTTPException(400, "No CV uploaded. Please upload your CV first.")
    cv_text = await asyncio.to_thread(job_engine.extract_cv_text, cv_path)
    cv_analysis = await asyncio.to_thread(
        job_engine.analyze_cv, cv_text, cv_path, user_id=user_id, api_key=api_key)
    matched = await asyncio.to_thread(
        job_engine.match_jobs, cv_analysis, body.jobs, api_key=api_key)
    return {"jobs": matched, "cv_analysis": cv_analysis}


class DraftRequest(BaseModel):
    jobs: list[dict]


@app.post("/api/jobs/draft")
async def draft_emails(body: DraftRequest, request: Request):
    user = require_user(request)
    user_id = user["user_id"]
    api_key = users.load_user_gemini_key(user_id) or os.environ.get("GEMINI_API_KEY")
    cv_path = users.get_user_cv_path(user_id)
    if not cv_path:
        raise HTTPException(400, "No CV uploaded.")
    cv_text = await asyncio.to_thread(job_engine.extract_cv_text, cv_path)
    cv_analysis = await asyncio.to_thread(
        job_engine.analyze_cv, cv_text, cv_path, user_id=user_id, api_key=api_key)
    draft_results = await asyncio.to_thread(
        job_engine.draft_emails_batch, body.jobs, cv_analysis, user["email"], api_key=api_key)
    drafts = [{**job, "draft": draft_results[i]} for i, job in enumerate(body.jobs)]
    return {"drafts": drafts}


class SendRequest(BaseModel):
    drafts: list[dict]


@app.post("/api/jobs/send")
async def send_applications(body: SendRequest, request: Request):
    user = require_user(request)
    user_id = user["user_id"]
    cv_path = users.get_user_cv_path(user_id)
    if not cv_path:
        raise HTTPException(400, "No CV uploaded.")
    results = []
    for item in body.drafts:
        draft = item.get("draft", {})
        to = draft.get("to") or item.get("contact_email", "")
        if not to:
            results.append({**item, "send_status": "skipped", "reason": "No recipient email"})
            continue
        try:
            result = await asyncio.to_thread(
                email_sender.send_email_with_attachment,
                user_id, to, draft.get("subject", ""), draft.get("body", ""),
                str(cv_path), cv_path.name)
            users.save_user_application(user_id, item, draft)
            results.append({**item, "send_status": "sent", "send_result": result})
        except Exception as exc:
            results.append({**item, "send_status": "failed", "reason": str(exc)})
    return {"results": results}


@app.get("/api/jobs/history")
def job_history(request: Request):
    user = require_user(request)
    return {"applications": users.load_user_applications(user["user_id"])}


@app.get("/api/jobs/analytics")
def job_analytics(request: Request):
    user = require_user(request)
    return {"analytics": job_engine.get_analytics(user["user_id"])}


# --- Config ---

@app.get("/api/jobs/config")
def get_job_config(request: Request):
    user = require_user(request)
    return {"config": users.load_user_config(user["user_id"])}


class JobConfigRequest(BaseModel):
    config: dict


@app.post("/api/jobs/config")
def save_job_config(body: JobConfigRequest, request: Request):
    user = require_user(request)
    users.save_user_config(user["user_id"], body.config)
    return {"status": "saved"}


# --- Presets ---

@app.get("/api/jobs/presets")
def list_presets(request: Request):
    user = require_user(request)
    return {"presets": users.load_user_presets(user["user_id"])}


class PresetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    role: str = ""
    country: str = ""
    state: str = ""
    experience: str = ""
    date_from: str = ""
    date_to: str = ""
    active_for_runner: bool = False


@app.post("/api/jobs/presets")
def create_preset(body: PresetRequest, request: Request):
    user = require_user(request)
    preset = job_engine.add_preset(user["user_id"], body.model_dump())
    return {"preset": preset}


class PresetUpdateRequest(BaseModel):
    name: str | None = None
    role: str | None = None
    country: str | None = None
    state: str | None = None
    experience: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    active_for_runner: bool | None = None


@app.put("/api/jobs/presets/{preset_id}")
def update_preset_endpoint(preset_id: str, body: PresetUpdateRequest, request: Request):
    user = require_user(request)
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    result = job_engine.update_preset(user["user_id"], preset_id, updates)
    if result is None:
        raise HTTPException(404, "Preset not found")
    return {"preset": result}


@app.delete("/api/jobs/presets/{preset_id}")
def delete_preset_endpoint(preset_id: str, request: Request):
    user = require_user(request)
    if not job_engine.delete_preset(user["user_id"], preset_id):
        raise HTTPException(404, "Preset not found")
    return {"status": "deleted"}


# --- Runner ---

@app.get("/api/jobs/runner")
def get_runner_config(request: Request):
    user = require_user(request)
    return {"runner": users.load_user_runner(user["user_id"])}


class RunnerConfigRequest(BaseModel):
    enabled: bool | None = None
    run_time: str | None = None
    min_match_score: int | None = None
    to_email: str | None = None


@app.post("/api/jobs/runner")
def update_runner_config(body: RunnerConfigRequest, request: Request):
    user = require_user(request)
    user_id = user["user_id"]
    cfg = users.load_user_runner(user_id)
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    cfg.update(updates)
    users.save_user_runner(user_id, cfg)
    return {"runner": cfg}


@app.post("/api/jobs/runner/trigger")
async def trigger_daily_scan(request: Request):
    user = require_user(request)
    user_id = user["user_id"]
    try:
        results = await asyncio.to_thread(job_engine.run_daily_scan_for_user, user_id)
        runner_cfg = users.load_user_runner(user_id)
        email_result = None
        if results.get("total_jobs", 0) > 0 and runner_cfg.get("to_email"):
            try:
                email_result = await asyncio.to_thread(
                    _send_daily_summary_email, user_id, results, runner_cfg["to_email"])
            except Exception as exc:
                email_result = {"status": "failed", "error": str(exc)}
        return {"results": results, "email": email_result}
    except Exception as exc:
        raise HTTPException(500, f"Daily scan failed: {exc}") from exc


@app.get("/api/jobs/runner/results")
def get_daily_results(request: Request):
    user = require_user(request)
    return {"results": users.load_user_daily_results(user["user_id"])}


# --- Gemini API Key ---

@app.get("/api/jobs/gemini")
def get_gemini_status(request: Request):
    user = require_user(request)
    user_key = users.load_user_gemini_key(user["user_id"])
    global_key = os.environ.get("GEMINI_API_KEY", "")
    configured = bool(user_key or global_key)
    return {"configured": configured, "model": GEMINI_MODEL,
            "has_personal_key": bool(user_key)}


class GeminiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1, max_length=200)


@app.post("/api/jobs/gemini")
def save_gemini_key(body: GeminiKeyRequest, request: Request):
    user = require_user(request)
    users.save_user_gemini_key(user["user_id"], body.api_key)
    return {"status": "saved", "configured": True}


# --- Scrape emails ---

@app.post("/api/jobs/scrape-emails")
async def scrape_emails(body: dict, request: Request):
    require_user(request)
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, "URL required")
    emails = await asyncio.to_thread(job_engine.scrape_contact_emails, url)
    return {"emails": emails}


# ------------------------------------------------------------------
# Health check (for Render keep-alive pings)
# ------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Job Hunter", "gemini_configured": gemini_is_configured()}


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
