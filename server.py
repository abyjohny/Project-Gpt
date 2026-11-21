"""
server.py - PHASE 4: The web UI's backend (FastAPI).
====================================================

The browser (web/index.html + app.js) talks to this file over plain HTTP:

    GET    /                        the web page itself
    GET    /api/health              is Ollama up? which models? how many chunks?
    GET    /api/stats               live system stats (GPU memory, speed, ...) for Settings
    GET    /api/documents           what is in the knowledge base
    POST   /api/upload              add a file -> load, chunk, embed, store (same as ingest.py)
    GET    /api/sessions            saved chats for the sidebar
    GET    /api/sessions/{id}       one chat, with every message and its sources
    DELETE /api/sessions/{id}       delete a chat
    POST   /api/chat                ask a question -> the answer STREAMS back (Server-Sent Events)
    GET    /api/agent/folders       folders Agent mode may read (the sandbox)
    POST   /api/agent/folders       allow a folder        DELETE /api/agent/folders  remove one

HOW STREAMING WORKS (Server-Sent Events, "SSE")
    The answer is not sent in one piece. The response stays open and we push small
    named events down the wire as soon as they are ready:

        event: status   data: {"text": "Searching your documents..."}
        event: sources  data: [ {n: 1, title: "NHCX_Spec.pdf", ...}, ... ]
        event: token    data: {"text": "The claim"}
        event: token    data: {"text": " deadline is"}
        ...
        event: done     data: {"tokens_per_second": 38.2, ...}

    The heavy work (search, re-rank, Qwen) is ordinary blocking Python, so it runs in a
    worker thread. The thread drops events into a queue; the async side forwards them.
    If the browser presses "Stop" (closes the connection) we set a flag and the worker
    stops asking Qwen for more words.

Run:   python server.py      then open  http://127.0.0.1:8080
Learn more: docs/learning/16_web_ui.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ingest
from rag import (agent, agent_tools, config, email_agent, generator, intent, job_hunter, keyword_search,
                 gemini_client, ollama_client, pipeline, reranker, vector_store, web_search)
from rag.sessions import SessionStore, result_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")
for noisy in ("primp", "ddgs", "httpx"):     # web search libraries log every request (incl. the query)
    logging.getLogger(noisy).setLevel(logging.WARNING)

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024     # 200 MB per file
MAX_QUESTION_CHARS = 4000

sessions = SessionStore()
state = {
    "warming_up": True,            # keyword index + re-ranker still loading
    "warm_error": None,
    "last_generation": None,       # speed of the most recent answer, for the stats panel
    "ingesting": None,             # name of the file being ingested right now
}
_ingest_lock = threading.Lock()    # one ingestion at a time (it is GPU-heavy)


# ----------------------------------------------------------------
# Startup: prepare the slow parts in the background (like chat.py does up-front)
# ----------------------------------------------------------------
def _warm_up() -> None:
    try:
        keyword_search.get_index()
        if config.USE_RERANKER and reranker.is_available():
            reranker.warm_up()
    except Exception as exc:     # never crash the server because of a warm-up problem
        logger.warning("Warm-up failed: %s", exc)
        state["warm_error"] = str(exc)
    finally:
        state["warming_up"] = False
        logger.info("Warm-up finished (re-ranker on %s).", reranker.device_name())


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True, name="warm-up").start()
    threading.Thread(target=_daily_runner_loop, daemon=True, name="daily-runner").start()
    yield


# ----------------------------------------------------------------
# Daily runner background loop
# ----------------------------------------------------------------
def _daily_runner_loop() -> None:
    """Check every 60 seconds if it's time to run the daily job scan."""
    import traceback
    logger.info("Daily runner loop started.")
    while True:
        try:
            time.sleep(60)
            cfg = job_hunter.load_runner_config()
            if not cfg.get("enabled"):
                continue
            run_time = cfg.get("run_time", "09:00")
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            today = now.strftime("%Y-%m-%d")
            last_run_date = cfg.get("last_run_date")
            if current_time == run_time and last_run_date != today:
                logger.info("Daily runner triggered at %s", current_time)
                try:
                    results = job_hunter.run_daily_scan()
                    to_email = cfg.get("to_email", "abyjohny25@gmail.com")
                    if results.get("total_jobs", 0) > 0 and to_email:
                        job_hunter.send_daily_summary(results, to_email)
                        logger.info("Daily summary email sent to %s", to_email)
                    else:
                        logger.info("Daily scan complete: %d jobs found", results.get("total_jobs", 0))
                except Exception:
                    logger.error("Daily scan failed:\n%s", traceback.format_exc())
        except Exception:
            logger.error("Daily runner loop error:\n%s", traceback.format_exc())


app = FastAPI(title="AJP - Local RAG", lifespan=lifespan)
WEB_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Frontend not found.</h1>"


# ----------------------------------------------------------------
# Health + live stats
# ----------------------------------------------------------------
def _ollama_models() -> list[str]:
    try:
        return ollama_client.list_models()
    except Exception:
        return []


@app.get("/api/health")
def health():
    running = ollama_client.is_running()
    models = _ollama_models() if running else []
    return {
        "ollama_running": running,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chat_model_ready": any(m == config.CHAT_MODEL or m.startswith(config.CHAT_MODEL + ":") for m in models),
        "chunks": vector_store.count(),
        "warming_up": state["warming_up"],
    }


_gpu_cache: dict = {"at": 0.0, "data": None}


def _gpu_stats() -> dict | None:
    """Ask nvidia-smi for GPU memory/usage. Cached for 1 s; None if there is no NVIDIA GPU."""
    if time.time() - _gpu_cache["at"] < 1.0:
        return _gpu_cache["data"]
    data = None
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, creationflags=flags,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, used, total, util, temp = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            data = {"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total),
                    "utilization": int(util), "temperature": int(temp)}
    except (OSError, ValueError, subprocess.SubprocessError):
        data = None
    _gpu_cache.update(at=time.time(), data=data)
    return data


def _loaded_models() -> list[dict]:
    """Which models Ollama currently holds in memory (like `ollama ps`)."""
    try:
        import requests
        response = requests.get(f"{config.OLLAMA_URL}/api/ps", timeout=3)
        return [{"name": m.get("name"), "size_mb": round(m.get("size", 0) / 1e6),
                 "vram_mb": round(m.get("size_vram", 0) / 1e6)} for m in response.json().get("models", [])]
    except Exception:
        return []


@app.get("/api/stats")
def stats():
    running = ollama_client.is_running()
    sources = vector_store.list_sources()
    return {
        "ollama_running": running,
        "gpu": _gpu_stats(),
        "loaded_models": _loaded_models() if running else [],
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chunks": sum(sources.values()),
        "files": len(sources),
        "retrieval_mode": config.RETRIEVAL_MODE,
        "reranker": ("loading" if state["warming_up"] else reranker.device_name())
                    if config.USE_RERANKER and reranker.is_available() else "off",
        "top_k": config.TOP_K,
        "context_window": config.CONTEXT_WINDOW,
        "last_generation": state["last_generation"],
        "ingesting": state["ingesting"],
    }


# ----------------------------------------------------------------
# Documents
# ----------------------------------------------------------------
@app.get("/api/documents")
def documents():
    sources = vector_store.list_sources()
    return {"documents": [{"name": n, "chunks": c} for n, c in sorted(sources.items(), key=lambda i: i[0].lower())],
            "supported": sorted(config.SUPPORTED_EXTENSIONS)}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Save one file into data/documents and ingest it (same steps as `python ingest.py <file>`)."""
    # Path(...).name drops any folder part, so "../../server.py" becomes "server.py"
    # and can only ever land inside data/documents.
    filename = Path(file.filename or "").name.strip()
    if not filename or filename.startswith("."):
        raise HTTPException(400, "Invalid file name.")
    suffix = Path(filename).suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"'{suffix or filename}' files are not supported. "
                                 f"Supported: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}")
    if not ollama_client.is_running():
        raise HTTPException(503, "Ollama is not running. Start the Ollama app and try again.")

    save_path = config.DOCUMENTS_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_name(save_path.name + ".part")
    size = 0
    try:
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                buffer.write(chunk)
        temp_path.replace(save_path)
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to save file: {exc}") from exc

    def run_ingest() -> int:
        with _ingest_lock:
            state["ingesting"] = filename
            try:
                added = ingest.ingest([save_path.resolve()])
                keyword_search.get_index(force_rebuild=True)   # re-ingesting a file can keep the same chunk count
                return added
            finally:
                state["ingesting"] = None

    try:
        added = await asyncio.to_thread(run_ingest)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(500, f"Failed to index {filename}: {exc}") from exc

    if not added:
        raise HTTPException(422, f"No readable text found in {filename} (blank file, or a scan that OCR could not read).")
    return {"status": "success", "filename": filename, "chunks": added,
            "message": f"Indexed {filename} ({added} chunks)"}


# ----------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------
@app.get("/api/sessions")
def list_sessions():
    return {"sessions": sessions.list()}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Chat not found.")
    return session.to_dict()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not sessions.delete(session_id):
        raise HTTPException(404, "Chat not found.")
    return {"status": "deleted"}


# ----------------------------------------------------------------
# Agent mode: allowed folders (the sandbox)
# ----------------------------------------------------------------
class FolderRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


@app.get("/api/agent/folders")
def agent_folders():
    return {"folders": agent_tools.list_folders()}


@app.post("/api/agent/folders")
def add_agent_folder(body: FolderRequest):
    try:
        folder = agent_tools.add_folder(body.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"folder": folder, "folders": agent_tools.list_folders()}


@app.delete("/api/agent/folders")
def remove_agent_folder(body: FolderRequest):
    if not agent_tools.remove_folder(body.path):
        raise HTTPException(404, "Folder is not in the list.")
    return {"folders": agent_tools.list_folders()}


# ----------------------------------------------------------------
# Job Hunter
# ----------------------------------------------------------------

@app.get("/jobs", response_class=HTMLResponse)
async def serve_jobs_page():
    jobs_path = WEB_DIR / "jobs.html"
    if jobs_path.exists():
        return jobs_path.read_text(encoding="utf-8")
    return "<h1>Job Hunter page not found.</h1>"


class JobSearchRequest(BaseModel):
    role: str = Field(min_length=1, max_length=200)
    country: str = Field(min_length=1, max_length=100)
    state: str = ""
    date_from: str = ""
    date_to: str = ""
    experience: str = ""


@app.post("/api/jobs/search")
async def search_jobs(body: JobSearchRequest):
    try:
        results = await asyncio.to_thread(
            job_hunter.search_jobs, body.role, body.country, body.state, body.date_from, body.date_to, body.experience)
        return {"jobs": results}
    except Exception as exc:
        raise HTTPException(500, f"Job search failed: {exc}") from exc


@app.post("/api/jobs/cv")
async def upload_cv(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name.strip()
    if not filename:
        raise HTTPException(400, "Invalid file name.")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt", ".doc"):
        raise HTTPException(415, "Only PDF, DOCX, DOC, and TXT files are supported for CVs.")
    cv_dir = config.PROJECT_ROOT / "data" / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    save_path = cv_dir / filename
    content = await file.read()
    save_path.write_bytes(content)
    try:
        cv_text = await asyncio.to_thread(job_hunter.extract_cv_text, save_path)
    except Exception as exc:
        raise HTTPException(500, f"Failed to read CV: {exc}") from exc
    return {"status": "ok", "filename": filename, "path": str(save_path), "preview": cv_text[:500]}


@app.get("/api/jobs/cv")
def get_current_cv():
    cv_dir = config.PROJECT_ROOT / "data" / "cv"
    if not cv_dir.exists():
        return {"cv": None}
    files = sorted(cv_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    files = [f for f in files if f.suffix.lower() in (".pdf", ".docx", ".txt")]
    if not files:
        return {"cv": None}
    f = files[0]
    return {"cv": {"filename": f.name, "path": str(f), "size": f.stat().st_size, "modified": f.stat().st_mtime}}


class AnalyzeRequest(BaseModel):
    jobs: list[dict]


@app.post("/api/jobs/analyze")
async def analyze_jobs(body: AnalyzeRequest):
    cv_dir = config.PROJECT_ROOT / "data" / "cv"
    files = sorted(cv_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True) if cv_dir.exists() else []
    files = [f for f in files if f.suffix.lower() in (".pdf", ".docx", ".txt")]
    if not files:
        raise HTTPException(400, "No CV uploaded. Please upload your CV first.")
    cv_path = files[0]
    cv_text = await asyncio.to_thread(job_hunter.extract_cv_text, cv_path)
    cv_analysis = await asyncio.to_thread(job_hunter.analyze_cv, cv_text, cv_path)
    matched = await asyncio.to_thread(job_hunter.match_jobs, cv_analysis, body.jobs)
    return {"jobs": matched, "cv_analysis": cv_analysis}


class DraftRequest(BaseModel):
    jobs: list[dict]


@app.post("/api/jobs/draft")
async def draft_emails(body: DraftRequest):
    cv_dir = config.PROJECT_ROOT / "data" / "cv"
    files = sorted(cv_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True) if cv_dir.exists() else []
    files = [f for f in files if f.suffix.lower() in (".pdf", ".docx", ".txt")]
    if not files:
        raise HTTPException(400, "No CV uploaded.")
    cv_path = files[0]
    cv_text = await asyncio.to_thread(job_hunter.extract_cv_text, cv_path)
    cv_analysis = await asyncio.to_thread(job_hunter.analyze_cv, cv_text, cv_path)
    # Batched: single Ollama call for all drafts
    draft_results = await asyncio.to_thread(
        job_hunter.draft_emails_batch, body.jobs, cv_analysis, "abyjohny25@gmail.com")
    drafts = [{**job, "draft": draft_results[i]} for i, job in enumerate(body.jobs)]
    return {"drafts": drafts}


class SendRequest(BaseModel):
    drafts: list[dict]


@app.post("/api/jobs/send")
async def send_applications(body: SendRequest):
    cv_dir = config.PROJECT_ROOT / "data" / "cv"
    files = sorted(cv_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True) if cv_dir.exists() else []
    files = [f for f in files if f.suffix.lower() in (".pdf", ".docx", ".txt")]
    if not files:
        raise HTTPException(400, "No CV uploaded.")
    cv_path = files[0]
    results = []
    for item in body.drafts:
        draft = item.get("draft", {})
        to = draft.get("to") or item.get("contact_email", "")
        if not to:
            results.append({**item, "send_status": "skipped", "reason": "No recipient email"})
            continue
        try:
            result = await asyncio.to_thread(
                job_hunter.send_application, to, draft.get("subject", ""), draft.get("body", ""), cv_path)
            job_hunter.save_application(item, draft)
            results.append({**item, "send_status": "sent", "send_result": result})
        except Exception as exc:
            results.append({**item, "send_status": "failed", "reason": str(exc)})
    return {"results": results}


@app.get("/api/jobs/history")
def job_history():
    return {"applications": job_hunter.load_applications()}


@app.get("/api/jobs/config")
def get_job_config():
    return {"config": job_hunter.load_config()}


class JobConfigRequest(BaseModel):
    config: dict


@app.post("/api/jobs/config")
def save_job_config(body: JobConfigRequest):
    job_hunter.save_config(body.config)
    return {"status": "saved"}


@app.get("/api/jobs/analytics")
def job_analytics():
    return {"analytics": job_hunter.get_analytics()}


@app.post("/api/jobs/scrape-emails")
async def scrape_emails(body: dict):
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, "URL required")
    emails = await asyncio.to_thread(job_hunter.scrape_contact_emails, url)
    return {"emails": emails}


# ----------------------------------------------------------------
# Job Hunter: Presets CRUD
# ----------------------------------------------------------------

@app.get("/api/jobs/presets")
def list_presets():
    return {"presets": job_hunter.load_presets()}


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
def create_preset(body: PresetRequest):
    preset = job_hunter.add_preset(body.model_dump())
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
def update_preset(preset_id: str, body: PresetUpdateRequest):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    result = job_hunter.update_preset(preset_id, updates)
    if result is None:
        raise HTTPException(404, "Preset not found")
    return {"preset": result}


@app.delete("/api/jobs/presets/{preset_id}")
def delete_preset(preset_id: str):
    if not job_hunter.delete_preset(preset_id):
        raise HTTPException(404, "Preset not found")
    return {"status": "deleted"}


# ----------------------------------------------------------------
# Job Hunter: Daily Runner
# ----------------------------------------------------------------

@app.get("/api/jobs/runner")
def get_runner_config():
    cfg = job_hunter.load_runner_config()
    return {"runner": cfg}


class RunnerConfigRequest(BaseModel):
    enabled: bool | None = None
    run_time: str | None = None
    min_match_score: int | None = None
    to_email: str | None = None


@app.post("/api/jobs/runner")
def update_runner_config(body: RunnerConfigRequest):
    cfg = job_hunter.load_runner_config()
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    cfg.update(updates)
    job_hunter.save_runner_config(cfg)
    return {"runner": cfg}


@app.post("/api/jobs/runner/trigger")
async def trigger_daily_scan():
    try:
        results = await asyncio.to_thread(job_hunter.run_daily_scan)
        # Also send summary email if configured
        runner_cfg = job_hunter.load_runner_config()
        email_result = None
        if results.get("total_jobs", 0) > 0 and runner_cfg.get("to_email"):
            try:
                email_result = await asyncio.to_thread(
                    job_hunter.send_daily_summary, results, runner_cfg["to_email"])
            except Exception as exc:
                email_result = {"status": "failed", "error": str(exc)}
        return {"results": results, "email": email_result}
    except Exception as exc:
        raise HTTPException(500, f"Daily scan failed: {exc}") from exc


@app.get("/api/jobs/runner/results")
def get_daily_results():
    return {"results": job_hunter.load_daily_results()}


# ----------------------------------------------------------------
# Job Hunter: Gemini API Key
# ----------------------------------------------------------------

@app.get("/api/jobs/gemini")
def get_gemini_status():
    return {
        "configured": gemini_client.is_configured(),
        "model": gemini_client.GEMINI_MODEL,
    }


class GeminiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1, max_length=200)


@app.post("/api/jobs/gemini")
def save_gemini_key(body: GeminiKeyRequest):
    gemini_client.save_api_key(body.api_key)
    return {"status": "saved", "configured": True}


# ----------------------------------------------------------------
# Chat (streaming)
# ----------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = None
    mode: str = "chat"          # "chat" = documents only | "cowork" = documents + web | "agent" = your folders | "email" = Gmail


def source_for_ui(number: int, result) -> dict:
    meta = result.metadata
    is_web = meta.get("type") == "web"
    section = meta.get("section") or ""
    return {
        "n": number,
        "type": "web" if is_web else "document",
        "title": meta.get("source") or "Untitled",
        "url": meta.get("url") if is_web else None,
        "page": meta.get("page") or None,
        "section": section.split(" > ")[-1][:90] if section else None,
        "score": round(result.score, 3) if result.score is not None else None,
        "score_name": "web" if is_web else result.score_name,
        "preview": result.text[:700],
    }


def run_chat(session, question: str, mode: str, emit, cancel: threading.Event) -> None:
    """The whole question -> answer pipeline. Runs in a worker thread; talks via emit()."""
    started = time.perf_counter()
    cowork = mode == "cowork"
    with session.lock:                       # one answer at a time per chat
        mem = session.memory
        try:
            # ---- 0. small talk: greetings / thanks / "who are you?" skip RAG entirely ----
            if intent.is_small_talk(question):
                emit("sources", {"sources": [], "search_query": None,
                                 "origin": "small_talk", "notice": None})
                emit("status", {"text": "Writing..."})
                gen_stats: dict = {}
                parts: list[str] = []
                stream = generator.generate_small_talk(
                    question, mem.as_messages(), stats=gen_stats,
                )
                try:
                    for piece in stream:
                        if cancel.is_set():
                            break
                        parts.append(piece)
                        emit("token", {"text": piece})
                finally:
                    stream.close()
                answer = "".join(parts)
                total_ms = (time.perf_counter() - started) * 1000
                tps = None
                if gen_stats.get("eval_count") and gen_stats.get("eval_duration"):
                    tps = round(gen_stats["eval_count"] / (gen_stats["eval_duration"] / 1e9), 1)
                done = {"retrieval_ms": 0, "total_ms": round(total_ms),
                        "tokens": gen_stats.get("eval_count"), "tokens_per_second": tps,
                        "prompt_tokens": gen_stats.get("prompt_eval_count"), "stopped": False}
                if answer.strip():
                    mem.add(question, question, answer, [])
                    session.add_exchange(question, {
                        "content": answer, "mode": mode, "origin": "small_talk",
                        "search_query": None, "sources": [], "stats": done, "notice": None,
                    })
                    session.save()
                emit("done", {**done, "session": session.summary()})
                return

            # ---- 0b. Agent mode: tools over the user's allowed folders (rag/agent.py) ----
            if mode == "agent":
                emit("sources", {"sources": [], "search_query": None, "origin": "agent",
                                 "notice": None if agent_tools.list_folders() else
                                 "No folders allowed yet. Click 'Folders' above the input to add one."})
                result = agent.run(question, mem.as_messages(), emit, cancel)
                answer = result["answer"]
                gen_stats = result["stats"]
                total_ms = (time.perf_counter() - started) * 1000
                tps = None
                if gen_stats.get("eval_count") and gen_stats.get("eval_duration"):
                    tps = round(gen_stats["eval_count"] / (gen_stats["eval_duration"] / 1e9), 1)
                done = {"retrieval_ms": None, "total_ms": round(total_ms), "tokens": gen_stats.get("eval_count"),
                        "tokens_per_second": tps, "prompt_tokens": gen_stats.get("prompt_eval_count"),
                        "stopped": cancel.is_set(), "steps": len(result["steps"])}
                if tps:
                    state["last_generation"] = {**done, "at": time.time(), "model": config.CHAT_MODEL}
                if answer.strip():
                    mem.add(question, question, answer, [])
                    session.add_exchange(question, {
                        "content": answer, "mode": mode, "origin": "agent", "search_query": None,
                        "sources": [], "steps": result["steps"], "stats": done, "notice": None,
                    })
                    session.save()
                emit("done", {**done, "session": session.summary()})
                return

            # ---- 0c. Email mode: Gmail tools (rag/email_agent.py) ----
            if mode == "email":
                emit("sources", {"sources": [], "search_query": None, "origin": "email",
                                 "notice": None})
                result = email_agent.run(question, mem.as_messages(), emit, cancel)
                answer = result["answer"]
                gen_stats = result["stats"]
                total_ms = (time.perf_counter() - started) * 1000
                tps = None
                if gen_stats.get("eval_count") and gen_stats.get("eval_duration"):
                    tps = round(gen_stats["eval_count"] / (gen_stats["eval_duration"] / 1e9), 1)
                done = {"retrieval_ms": None, "total_ms": round(total_ms), "tokens": gen_stats.get("eval_count"),
                        "tokens_per_second": tps, "prompt_tokens": gen_stats.get("prompt_eval_count"),
                        "stopped": cancel.is_set(), "steps": len(result["steps"])}
                if tps:
                    state["last_generation"] = {**done, "at": time.time(), "model": config.CHAT_MODEL}
                if answer.strip():
                    mem.add(question, question, answer, [])
                    session.add_exchange(question, {
                        "content": answer, "mode": mode, "origin": "email", "search_query": None,
                        "sources": [], "steps": result["steps"], "stats": done, "notice": None,
                    })
                    session.save()
                emit("done", {**done, "session": session.summary()})
                return

            # ---- 1. sources: clarification / follow-up / new search (Phase 3 logic) ----
            emit("status", {"text": "Understanding your follow-up..." if mem.needs_rewrite(question)
                            else "Searching your documents..."})
            plan = pipeline.find_sources(question, mem)
            results = list(plan.results)
            notice = None
            if plan.origin == "clarify":
                notice = "Re-using the sources of the previous answer"
            elif plan.origin == "fallback":
                notice = "Nothing new found, using the previous answer's sources"

            # ---- 2. Cowork: also look on the live web ----
            if cowork and plan.origin != "clarify" and not cancel.is_set():
                emit("status", {"text": "Searching the web..."})
                try:
                    web_results = web_search.search_and_read(question, plan.search_query)
                    results = results[:4] + web_results    # documents first: they are the most trusted
                except web_search.WebSearchError as exc:
                    notice = f"Web search unavailable ({exc}). Answering from your documents."
            retrieval_ms = (time.perf_counter() - started) * 1000

            emit("sources", {"sources": [source_for_ui(i, r) for i, r in enumerate(results, start=1)],
                             "search_query": plan.search_query if plan.search_query != question else None,
                             "origin": plan.origin, "notice": notice})

            # ---- 3. generate (streamed) ----
            if cancel.is_set():
                return
            emit("status", {"text": "Writing..."})
            gen_stats: dict = {}
            parts: list[str] = []
            stopped = False
            stream = generator.generate_answer(
                question, results, mem.as_messages(), plan.hint,
                system_prompt=generator.COWORK_SYSTEM_PROMPT if cowork else generator.SYSTEM_PROMPT,
                allow_empty=cowork, stats=gen_stats,
            )
            try:
                for piece in stream:
                    if cancel.is_set():
                        stopped = True
                        break
                    parts.append(piece)
                    emit("token", {"text": piece})
            finally:
                stream.close()     # closes the HTTP stream -> Ollama stops writing

            answer = "".join(parts)
            total_ms = (time.perf_counter() - started) * 1000
            tps = None
            if gen_stats.get("eval_count") and gen_stats.get("eval_duration"):
                tps = round(gen_stats["eval_count"] / (gen_stats["eval_duration"] / 1e9), 1)
            done = {"retrieval_ms": round(retrieval_ms), "total_ms": round(total_ms),
                    "tokens": gen_stats.get("eval_count"), "tokens_per_second": tps,
                    "prompt_tokens": gen_stats.get("prompt_eval_count"), "stopped": stopped}
            if tps:
                state["last_generation"] = {**done, "at": time.time(), "model": config.CHAT_MODEL}

            # ---- 4. remember it (memory for follow-ups + saved chat for the sidebar) ----
            if answer.strip():
                mem.add(question, plan.search_query, answer, results)
                session.add_exchange(question, {
                    "content": answer, "mode": mode, "origin": plan.origin,
                    "search_query": plan.search_query,
                    "sources": [result_to_dict(r) for r in results],
                    "stats": done, "notice": notice,
                })
                session.save()
            emit("done", {**done, "session": session.summary()})

        except ollama_client.OllamaError as exc:
            emit("error", {"message": str(exc)})
        except Exception as exc:
            logger.exception("Chat failed")
            emit("error", {"message": f"Something went wrong: {exc}"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    question = body.message.strip().lstrip("\ufeff")
    if not question:
        raise HTTPException(400, "Empty message.")
    if body.mode not in ("chat", "cowork", "agent", "email"):
        raise HTTPException(400, "mode must be 'chat', 'cowork', 'agent' or 'email'.")
    if not ollama_client.is_running():
        raise HTTPException(503, "Ollama is not running. Start the Ollama app and try again.")

    session = sessions.get(body.session_id) if body.session_id else None
    if session is None:
        session = sessions.create()

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def emit(event: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def worker() -> None:
        try:
            run_chat(session, question, body.mode, emit, cancel)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)   # "no more events"

    async def event_stream():
        threading.Thread(target=worker, daemon=True, name="chat").start()
        yield _sse("session", {"id": session.id, "title": session.title})
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(*item)
                if await request.is_disconnected():
                    break
        finally:
            cancel.set()      # browser pressed Stop / closed the tab -> stop generating

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import os
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
