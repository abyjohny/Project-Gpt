"""
gemini_client.py - Gemini API client for Job Hunter.
=====================================================

Uses Google's Gemini API (free tier: 15 RPM, 1500 req/day) for
CV analysis, job matching, and email drafting.

The API key is stored in data/gemini_api_key.txt or set via
the GEMINI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("gemini_client")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_KEY_FILE = PROJECT_ROOT / "data" / "gemini_api_key.txt"

GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
GEMINI_MODEL = GEMINI_MODELS[0]

_client = None


def _get_api_key() -> str | None:
    """Read API key from file or environment."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def is_configured() -> bool:
    """Check if Gemini API key is available."""
    return bool(_get_api_key())


def save_api_key(key: str) -> None:
    """Save API key to file."""
    API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_KEY_FILE.write_text(key.strip(), encoding="utf-8")
    global _client
    _client = None


def ask(system: str, user: str) -> str:
    """Send a chat request to Gemini and return the response text."""
    key = _get_api_key()
    if not key:
        raise RuntimeError(
            "Gemini API key not configured. "
            "Set GEMINI_API_KEY env var or save it via the Job Hunter settings."
        )

    # 1. Try modern google.genai package
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.2,
                        max_output_tokens=4096,
                    ),
                )
                if response.text:
                    return response.text
            except Exception as e:
                logger.warning("Gemini model %s failed: %s, trying next...", model_name, e)
                continue
    except ImportError:
        pass

    # 2. Fallback to google.generativeai package
    try:
        import google.generativeai as genai_legacy

        genai_legacy.configure(api_key=key)
        model = genai_legacy.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system,
            generation_config={"temperature": 0.2, "max_output_tokens": 4096},
        )
        response = model.generate_content(user)
        return response.text or ""
    except ImportError:
        pass

    # 3. Fallback to direct HTTP request using requests
    import requests

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "")
    return ""
