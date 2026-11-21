"""
email_client.py - Gmail API client for Project GPT.
=====================================================

Handles OAuth 2.0 authentication and all Gmail operations:
  - List / search / read emails
  - Send emails and reply to threads
  - Mark as read/unread
  - Get inbox summary

The first time you use any email function, a browser window opens asking you
to log in to your Google account and grant permission. After that, a token
is saved and reused automatically.

Credentials:  data/credentials.json   (download from Google Cloud Console)
Token cache:  data/gmail_token.json   (created automatically after first login)
"""

from __future__ import annotations

import base64
import json
import logging
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from rag import config

logger = logging.getLogger("email_client")

CREDENTIALS_FILE = config.PROJECT_ROOT / "data" / "credentials.json"
TOKEN_FILE = config.PROJECT_ROOT / "data" / "gmail_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


class EmailError(Exception):
    """An email operation failed — the agent should see and react to this."""


def _get_service():
    """Build and return an authenticated Gmail API service object."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise EmailError(
            "Gmail dependencies not installed. Run:\n"
            "  pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        ) from exc

    if not CREDENTIALS_FILE.exists():
        raise EmailError(
            f"Gmail credentials not found at {CREDENTIALS_FILE}.\n"
            "Download credentials.json from Google Cloud Console and place it in data/."
        )

    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        logger.info("Gmail token saved to %s", TOKEN_FILE)

    return build("gmail", "v1", credentials=creds)


# Cache the service so we don't re-auth every call
_service_cache = None


def _service():
    global _service_cache
    if _service_cache is None:
        _service_cache = _get_service()
    return _service_cache


def reset_service():
    """Force re-authentication on next call."""
    global _service_cache
    _service_cache = None


# ------------------------------------------------------------------
# Read operations
# ------------------------------------------------------------------

def _parse_headers(headers: list[dict]) -> dict[str, str]:
    """Extract common headers into a flat dict."""
    result = {}
    wanted = {"From", "To", "Subject", "Date", "Cc", "Bcc", "Reply-To", "Message-ID", "In-Reply-To"}
    for h in headers:
        if h["name"] in wanted:
            result[h["name"]] = h["value"]
    return result


def _get_body(payload: dict) -> str:
    """Extract the plain text body from a Gmail message payload."""
    # Simple message
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    # Multipart: recurse into parts
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
        # Nested multipart
        body = _get_body(part)
        if body:
            return body

    # Fallback: try HTML
    if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data"):
        html = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        # Very basic HTML to text — strip tags
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
            html = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
            text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", "", text)
            return text.strip()

    return "(no readable body)"


def list_emails(query: str = "", max_results: int = 15, label: str = "INBOX") -> list[dict]:
    """List emails matching a query. Returns a list of email summaries."""
    try:
        svc = _service()
        kwargs: dict[str, Any] = {
            "userId": "me",
            "maxResults": min(max_results, 50),
        }
        if label:
            kwargs["labelIds"] = [label]
        if query:
            kwargs["q"] = query

        result = svc.users().messages().list(**kwargs).execute()
        messages = result.get("messages", [])
        if not messages:
            return []

        emails = []
        for msg_ref in messages:
            msg = svc.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"],
            ).execute()
            headers = _parse_headers(msg.get("payload", {}).get("headers", []))
            labels = msg.get("labelIds", [])
            emails.append({
                "id": msg["id"],
                "thread_id": msg.get("threadId"),
                "subject": headers.get("Subject", "(no subject)"),
                "from": headers.get("From", "unknown"),
                "to": headers.get("To", ""),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
                "unread": "UNREAD" in labels,
                "labels": [l for l in labels if l not in ("INBOX", "CATEGORY_PERSONAL", "CATEGORY_UPDATES",
                                                           "CATEGORY_SOCIAL", "CATEGORY_PROMOTIONS", "CATEGORY_FORUMS")],
            })
        return emails

    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to list emails: {exc}") from exc


def read_email(msg_id: str) -> dict:
    """Read the full content of a specific email."""
    try:
        svc = _service()
        msg = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
        headers = _parse_headers(msg.get("payload", {}).get("headers", []))
        body = _get_body(msg.get("payload", {}))
        labels = msg.get("labelIds", [])

        # List attachments
        attachments = []
        for part in msg.get("payload", {}).get("parts", []):
            filename = part.get("filename")
            if filename:
                attachments.append({
                    "filename": filename,
                    "mime_type": part.get("mimeType", ""),
                    "size": part.get("body", {}).get("size", 0),
                })

        return {
            "id": msg["id"],
            "thread_id": msg.get("threadId"),
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", "unknown"),
            "to": headers.get("To", ""),
            "cc": headers.get("Cc", ""),
            "date": headers.get("Date", ""),
            "body": body,
            "snippet": msg.get("snippet", ""),
            "unread": "UNREAD" in labels,
            "labels": labels,
            "attachments": attachments,
            "message_id": headers.get("Message-ID", ""),
        }

    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to read email {msg_id}: {exc}") from exc


def search_emails(query: str, max_results: int = 20) -> list[dict]:
    """Search emails using Gmail search syntax (from:, subject:, after:, etc.)."""
    return list_emails(query=query, max_results=max_results, label="")


def get_inbox_summary() -> dict:
    """Get a quick summary of the inbox: total unread, recent senders, etc."""
    try:
        svc = _service()
        # Get unread count
        profile = svc.users().getProfile(userId="me").execute()
        unread_result = svc.users().messages().list(
            userId="me", labelIds=["INBOX", "UNREAD"], maxResults=50,
        ).execute()
        unread_msgs = unread_result.get("messages", [])
        unread_count = unread_result.get("resultSizeEstimate", len(unread_msgs))

        # Summarize recent unread senders + subjects
        recent = []
        for msg_ref in unread_msgs[:10]:
            msg = svc.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = _parse_headers(msg.get("payload", {}).get("headers", []))
            recent.append({
                "id": msg["id"],
                "from": headers.get("From", "unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
            })

        return {
            "email": profile.get("emailAddress", ""),
            "total_unread": unread_count,
            "recent_unread": recent,
        }

    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to get inbox summary: {exc}") from exc


# ------------------------------------------------------------------
# Write operations
# ------------------------------------------------------------------

def send_email(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> dict:
    """Send a new email."""
    try:
        svc = _service()
        message = MIMEMultipart()
        message["to"] = to
        message["subject"] = subject
        if cc:
            message["cc"] = cc
        if bcc:
            message["bcc"] = bcc
        message.attach(MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()

        return {
            "status": "sent",
            "id": sent.get("id"),
            "thread_id": sent.get("threadId"),
            "to": to,
            "subject": subject,
        }

    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to send email: {exc}") from exc


def send_email_with_attachment(to: str, subject: str, body: str, attachment_path: str, attachment_name: str = "", cc: str = "", bcc: str = "") -> dict:
    """Send an email with a file attachment."""
    from email.mime.base import MIMEBase
    from email import encoders
    try:
        svc = _service()
        message = MIMEMultipart()
        message["to"] = to
        message["subject"] = subject
        if cc:
            message["cc"] = cc
        if bcc:
            message["bcc"] = bcc
        message.attach(MIMEText(body, "plain"))

        filepath = Path(attachment_path)
        if not filepath.exists():
            raise EmailError(f"Attachment not found: {attachment_path}")
        fname = attachment_name or filepath.name

        part = MIMEBase("application", "octet-stream")
        part.set_payload(filepath.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        message.attach(part)

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"status": "sent", "id": sent.get("id"), "thread_id": sent.get("threadId"), "to": to, "subject": subject, "attachment": fname}
    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to send email with attachment: {exc}") from exc


def reply_email(msg_id: str, body: str) -> dict:
    """Reply to a specific email (preserves the thread)."""
    try:
        svc = _service()
        # Read the original to get thread info
        original = svc.users().messages().get(userId="me", id=msg_id, format="metadata",
                                               metadataHeaders=["From", "To", "Subject", "Message-ID"]).execute()
        headers = _parse_headers(original.get("payload", {}).get("headers", []))
        thread_id = original.get("threadId")

        reply_to = headers.get("Reply-To") or headers.get("From", "")
        subject = headers.get("Subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        message = MIMEMultipart()
        message["to"] = reply_to
        message["subject"] = subject
        message["In-Reply-To"] = headers.get("Message-ID", "")
        message["References"] = headers.get("Message-ID", "")
        message.attach(MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        sent = svc.users().messages().send(
            userId="me", body={"raw": raw, "threadId": thread_id},
        ).execute()

        return {
            "status": "replied",
            "id": sent.get("id"),
            "thread_id": sent.get("threadId"),
            "to": reply_to,
            "subject": subject,
        }

    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to reply to email {msg_id}: {exc}") from exc


def mark_read(msg_ids: list[str]) -> dict:
    """Mark one or more emails as read."""
    try:
        svc = _service()
        for mid in msg_ids:
            svc.users().messages().modify(
                userId="me", id=mid, body={"removeLabelIds": ["UNREAD"]},
            ).execute()
        return {"status": "marked_read", "count": len(msg_ids)}

    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to mark emails as read: {exc}") from exc


def mark_unread(msg_ids: list[str]) -> dict:
    """Mark one or more emails as unread."""
    try:
        svc = _service()
        for mid in msg_ids:
            svc.users().messages().modify(
                userId="me", id=mid, body={"addLabelIds": ["UNREAD"]},
            ).execute()
        return {"status": "marked_unread", "count": len(msg_ids)}

    except EmailError:
        raise
    except Exception as exc:
        raise EmailError(f"Failed to mark emails as unread: {exc}") from exc
