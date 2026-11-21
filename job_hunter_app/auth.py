"""
auth.py - Google OAuth 2.0 web flow for multi-user authentication.
===================================================================

Handles:
  - Building Google OAuth authorization URL
  - Exchanging auth code for tokens at callback
  - Storing per-user refresh tokens
  - Building per-user Gmail service from stored tokens
  - Session cookie creation/validation using itsdangerous
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("auth")

# --- Config from env ---
def get_client_id() -> str:
    return os.environ.get("GOOGLE_CLIENT_ID", "")

def get_client_secret() -> str:
    return os.environ.get("GOOGLE_CLIENT_SECRET", "")

SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-to-a-random-string")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8080")

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_DIR = DATA_DIR / "users"

_serializer = URLSafeTimedSerializer(SECRET_KEY)
SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days


# ------------------------------------------------------------------
# Session cookies
# ------------------------------------------------------------------

def create_session_cookie(user_data: dict) -> str:
    """Create a signed session cookie containing user info."""
    return _serializer.dumps(user_data)


def validate_session_cookie(cookie: str) -> dict | None:
    """Validate and decode a session cookie. Returns user data or None."""
    try:
        return _serializer.loads(cookie, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


# ------------------------------------------------------------------
# OAuth flow helpers
# ------------------------------------------------------------------

def is_oauth_configured() -> bool:
    return bool(get_client_id() and get_client_secret())


def get_auth_url() -> str:
    """Build the Google OAuth authorization URL."""
    import urllib.parse
    redirect_uri = f"{BASE_URL}/auth/callback"
    params = {
        "client_id": get_client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"


def exchange_code(code: str) -> dict:
    """Exchange the authorization code for tokens. Returns token data."""
    import requests
    redirect_uri = f"{BASE_URL}/auth/callback"
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": get_client_id(),
            "client_secret": get_client_secret(),
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_user_info(access_token: str) -> dict:
    """Fetch user profile info from Google."""
    import requests
    resp = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def save_user_tokens(user_id: str, tokens: dict) -> None:
    """Save OAuth tokens for a user."""
    user_dir = USERS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    token_file = user_dir / "oauth_tokens.json"
    token_file.write_text(json.dumps(tokens, indent=2), encoding="utf-8")


def load_user_tokens(user_id: str) -> dict | None:
    """Load OAuth tokens for a user."""
    token_file = USERS_DIR / user_id / "oauth_tokens.json"
    if not token_file.exists():
        return None
    try:
        return json.loads(token_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def refresh_access_token(user_id: str) -> str | None:
    """Refresh the access token using stored refresh token."""
    import requests
    tokens = load_user_tokens(user_id)
    if not tokens or not tokens.get("refresh_token"):
        return None
    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": get_client_id(),
                "client_secret": get_client_secret(),
                "refresh_token": tokens["refresh_token"],
                "grant_type": "refresh_token",
            },
            timeout=15,
        )
        resp.raise_for_status()
        new_tokens = resp.json()
        # Keep the refresh token (it's not always returned)
        new_tokens.setdefault("refresh_token", tokens["refresh_token"])
        save_user_tokens(user_id, new_tokens)
        return new_tokens.get("access_token")
    except Exception as exc:
        logger.warning("Token refresh failed for user %s: %s", user_id, exc)
        return None


# ------------------------------------------------------------------
# Gmail service (per-user)
# ------------------------------------------------------------------

def get_gmail_service(user_id: str):
    """Build an authenticated Gmail API service for a user."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    tokens = load_user_tokens(user_id)
    if not tokens:
        raise RuntimeError("No OAuth tokens found. Please sign in again.")

    creds = Credentials(
        token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=get_client_id(),
        client_secret=get_client_secret(),
        scopes=SCOPES,
    )

    if creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request
        creds.refresh(Request())
        # Save refreshed tokens
        save_user_tokens(user_id, {
            "access_token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
        })

    return build("gmail", "v1", credentials=creds)
