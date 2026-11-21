"""
users.py - Multi-user data management.
=======================================

Each user gets an isolated directory under data/users/<user_id>/
containing their CV, presets, runner config, applications, and Gemini key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("users")

DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_DIR = DATA_DIR / "users"


def get_user_dir(user_id: str) -> Path:
    """Return (and create) the data directory for a user."""
    d = USERS_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_user_cv_dir(user_id: str) -> Path:
    d = get_user_dir(user_id) / "cv"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_user_cv_path(user_id: str) -> Path | None:
    """Return the most recent CV file for a user, or None."""
    cv_dir = get_user_cv_dir(user_id)
    files = sorted(
        [f for f in cv_dir.iterdir() if f.suffix.lower() in (".pdf", ".docx", ".txt")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    ) if cv_dir.exists() else []
    return files[0] if files else None


def _user_json(user_id: str, filename: str) -> Path:
    return get_user_dir(user_id) / filename


# --- Profile ---

def load_user_profile(user_id: str) -> dict:
    p = _user_json(user_id, "profile.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_user_profile(user_id: str, profile: dict) -> None:
    p = _user_json(user_id, "profile.json")
    p.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Gemini API key ---

def load_user_gemini_key(user_id: str) -> str | None:
    p = _user_json(user_id, "gemini_api_key.txt")
    if not p.exists():
        return None
    key = p.read_text(encoding="utf-8").strip()
    return key if key else None


def save_user_gemini_key(user_id: str, key: str) -> None:
    p = _user_json(user_id, "gemini_api_key.txt")
    p.write_text(key.strip(), encoding="utf-8")


# --- Presets ---

def load_user_presets(user_id: str) -> list[dict]:
    p = _user_json(user_id, "presets.json")
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_user_presets(user_id: str, presets: list[dict]) -> None:
    p = _user_json(user_id, "presets.json")
    p.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Runner config ---

def load_user_runner(user_id: str) -> dict:
    defaults = {
        "enabled": False,
        "run_time": "09:00",
        "min_match_score": 50,
        "to_email": "",
        "last_run": None,
        "last_run_date": None,
    }
    p = _user_json(user_id, "runner.json")
    if not p.exists():
        return defaults
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        defaults.update(data)
        return defaults
    except (json.JSONDecodeError, OSError):
        return defaults


def save_user_runner(user_id: str, cfg: dict) -> None:
    p = _user_json(user_id, "runner.json")
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Applications ---

def load_user_applications(user_id: str) -> list[dict]:
    p = _user_json(user_id, "applications.json")
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_user_application(user_id: str, job: dict, email_draft: dict) -> None:
    from datetime import datetime, timezone
    apps = load_user_applications(user_id)
    apps.append({
        "job": job,
        "email": email_draft,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    })
    p = _user_json(user_id, "applications.json")
    p.write_text(json.dumps(apps, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Daily results ---

def load_user_daily_results(user_id: str) -> dict:
    p = _user_json(user_id, "daily_results.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_user_daily_results(user_id: str, results: dict) -> None:
    p = _user_json(user_id, "daily_results.json")
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Search config ---

def load_user_config(user_id: str) -> dict:
    p = _user_json(user_id, "config.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_user_config(user_id: str, cfg: dict) -> None:
    p = _user_json(user_id, "config.json")
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# --- List all users with runner enabled ---

def list_users_with_runner() -> list[str]:
    """Return user_ids that have runner enabled."""
    result = []
    if not USERS_DIR.exists():
        return result
    for user_dir in USERS_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        runner_file = user_dir / "runner.json"
        if runner_file.exists():
            try:
                cfg = json.loads(runner_file.read_text(encoding="utf-8"))
                if cfg.get("enabled"):
                    result.append(user_dir.name)
            except (json.JSONDecodeError, OSError):
                continue
    return result
