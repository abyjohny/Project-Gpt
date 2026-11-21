"""
email_sender.py - Per-user Gmail send using OAuth tokens.
=========================================================

Sends emails and emails with attachments through each user's
authenticated Gmail account.
"""

from __future__ import annotations

import base64
import logging
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from auth import get_gmail_service

logger = logging.getLogger("email_sender")


def send_email(user_id: str, to: str, subject: str, body: str,
               cc: str = "", bcc: str = "") -> dict:
    """Send a plain text email using the user's Gmail account."""
    try:
        svc = get_gmail_service(user_id)
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
    except Exception as exc:
        raise RuntimeError(f"Failed to send email: {exc}") from exc


def send_email_with_attachment(user_id: str, to: str, subject: str, body: str,
                                attachment_path: str, attachment_name: str = "",
                                cc: str = "", bcc: str = "") -> dict:
    """Send an email with a file attachment using the user's Gmail account."""
    try:
        svc = get_gmail_service(user_id)
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
            raise FileNotFoundError(f"Attachment not found: {attachment_path}")
        fname = attachment_name or filepath.name

        part = MIMEBase("application", "octet-stream")
        part.set_payload(filepath.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        message.attach(part)

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {
            "status": "sent",
            "id": sent.get("id"),
            "thread_id": sent.get("threadId"),
            "to": to,
            "subject": subject,
            "attachment": fname,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to send email with attachment: {exc}") from exc
