"""
email_agent.py - Email Agent mode: manage Gmail through tool calls.
===================================================================

Like agent.py (the file/code agent), but instead of reading folders it reads
and writes emails. The model decides which email tools to use based on your
question, e.g.:

    you:   "What unread emails do I have?"
    model: -> get_inbox_summary()
    us:    <summary: 5 unread, from Alice, Bob, ...>
    model: "You have 5 unread emails. The most recent are from..."

    you:   "Send a reply to Bob saying I'll be there at 5"
    model: -> search_emails(query="from:bob")
    us:    <list of Bob's emails>
    model: -> reply_email(msg_id="...", body="Hi Bob, I'll be there at 5. Cheers!")
    us:    <sent confirmation>
    model: "Done! I've replied to Bob's email."

Guard rails:
    - MAX_STEPS: cannot loop forever
    - send_email and reply_email require the model to compose the content
    - All operations go through the authenticated Gmail API
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable

from rag import config, email_client, ollama_client

MAX_STEPS = 10
KEEP_FULL_RESULTS = 3
OLD_RESULT_CHARS = 700

EMAIL_SYSTEM_PROMPT = """You are AJP (Agent Jose Pereira) in Email mode: a helpful email assistant
connected to the user's Gmail account ({email}).

How to work:
- Use tools to check emails before answering. Never guess email contents.
- For "check my inbox" or "what's new": call get_inbox_summary first.
- For specific emails: use search_emails or list_emails to find them, then read_email for details.
- When asked to send/reply: compose a clear, professional message. Show the user what you'll send
  BEFORE calling send_email or reply_email, unless they already gave you the exact text.
- When the user says "mark as read", use mark_read with the message IDs.
- Be concise in your answers. Summarize emails clearly: sender, subject, key points.

IMPORTANT SAFETY RULES:
- NEVER send an email without the user explicitly asking you to.
- If the user's request is ambiguous about WHO to send to or WHAT to say, ASK first.
- When replying, always confirm the recipient and content before calling reply_email.

Final answer:
- Clear and well structured (short headings / bullets).
- Mention sender names, subjects, and dates when summarizing.
- For sent emails, confirm: to whom, subject, and a brief summary of what was sent."""

Emit = Callable[[str, dict], None]


def _summarize_args(args: dict) -> str:
    parts = [f"{k}={v!r}" if not isinstance(v, str) or len(v) < 80 else f"{k}='{v[:77]}...'"
             for k, v in (args or {}).items()]
    return ", ".join(parts)


def _shrink_old_results(messages: list[dict]) -> None:
    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_indexes[:-KEEP_FULL_RESULTS]:
        content = messages[i]["content"]
        if len(content) > OLD_RESULT_CHARS and not content.endswith("[shortened]"):
            messages[i]["content"] = content[:OLD_RESULT_CHARS] + "\n... [shortened]"


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

MAX_RESULT_CHARS = 6000

def _clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _chat_once(messages: list[dict], tools: list[dict] | None, stats: dict) -> dict:
    payload = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_ctx": config.AGENT_CONTEXT_WINDOW},
    }
    if tools:
        payload["tools"] = tools
    data = ollama_client._post("/api/chat", payload).json()
    for key in ("eval_count", "eval_duration", "prompt_eval_count"):
        stats[key] = stats.get(key, 0) + data.get(key, 0)
    message = data.get("message", {})
    message["content"] = _THINK.sub("", message.get("content") or "").strip()
    return message


# ------------------------------------------------------------------
# Email tool functions + schemas
# ------------------------------------------------------------------

def _tool_get_inbox_summary() -> str:
    summary = email_client.get_inbox_summary()
    lines = [f"Email: {summary['email']}", f"Total unread: {summary['total_unread']}", ""]
    if summary["recent_unread"]:
        lines.append("Recent unread emails:")
        for e in summary["recent_unread"]:
            lines.append(f"  ID: {e['id']}")
            lines.append(f"  From: {e['from']}")
            lines.append(f"  Subject: {e['subject']}")
            lines.append(f"  Date: {e['date']}")
            lines.append(f"  Preview: {e['snippet'][:200]}")
            lines.append("")
    else:
        lines.append("No unread emails!")
    return "\n".join(lines)


def _tool_list_emails(query: str = "", max_results: int = 15, label: str = "INBOX") -> str:
    emails = email_client.list_emails(query=query, max_results=max_results, label=label)
    if not emails:
        return "No emails found."
    lines = [f"Found {len(emails)} emails:\n"]
    for e in emails:
        flag = "📩 " if e["unread"] else "  "
        lines.append(f"{flag}ID: {e['id']}")
        lines.append(f"  From: {e['from']}")
        lines.append(f"  Subject: {e['subject']}")
        lines.append(f"  Date: {e['date']}")
        lines.append(f"  Preview: {e['snippet'][:150]}")
        lines.append("")
    return "\n".join(lines)


def _tool_read_email(msg_id: str) -> str:
    email = email_client.read_email(msg_id)
    lines = [
        f"Subject: {email['subject']}",
        f"From: {email['from']}",
        f"To: {email['to']}",
    ]
    if email["cc"]:
        lines.append(f"CC: {email['cc']}")
    lines.append(f"Date: {email['date']}")
    if email["attachments"]:
        att = ", ".join(a["filename"] for a in email["attachments"])
        lines.append(f"Attachments: {att}")
    lines.append(f"\n--- Body ---\n{email['body']}")
    return _clip("\n".join(lines))


def _tool_search_emails(query: str, max_results: int = 20) -> str:
    emails = email_client.search_emails(query=query, max_results=max_results)
    if not emails:
        return f"No emails found matching: {query}"
    lines = [f"Search results for '{query}' ({len(emails)} found):\n"]
    for e in emails:
        flag = "📩 " if e["unread"] else "  "
        lines.append(f"{flag}ID: {e['id']}")
        lines.append(f"  From: {e['from']}")
        lines.append(f"  Subject: {e['subject']}")
        lines.append(f"  Date: {e['date']}")
        lines.append(f"  Preview: {e['snippet'][:150]}")
        lines.append("")
    return "\n".join(lines)


def _tool_send_email(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    result = email_client.send_email(to=to, subject=subject, body=body, cc=cc, bcc=bcc)
    return f"Email sent successfully!\n  To: {result['to']}\n  Subject: {result['subject']}\n  Message ID: {result['id']}"


def _tool_reply_email(msg_id: str, body: str) -> str:
    result = email_client.reply_email(msg_id=msg_id, body=body)
    return f"Reply sent successfully!\n  To: {result['to']}\n  Subject: {result['subject']}\n  Thread ID: {result['thread_id']}"


def _tool_mark_read(msg_ids: str) -> str:
    ids = [mid.strip() for mid in msg_ids.split(",") if mid.strip()]
    if not ids:
        return "Error: no message IDs provided."
    result = email_client.mark_read(ids)
    return f"Marked {result['count']} email(s) as read."


def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []}}}


EMAIL_TOOLS = {
    "get_inbox_summary": (_tool_get_inbox_summary, _schema(
        "get_inbox_summary", "Get a summary of the inbox: unread count, recent senders, subjects. "
                             "Call this FIRST when asked about inbox status.", {})),
    "list_emails": (_tool_list_emails, _schema(
        "list_emails", "List recent emails in the inbox with subject, sender, date. "
                       "Use query for Gmail search syntax (from:, subject:, is:unread, etc.).",
        {"query": {"type": "string", "description": "Gmail search query (optional)"},
         "max_results": {"type": "integer", "description": "Max emails to return (default 15)"},
         "label": {"type": "string", "description": "Gmail label (default INBOX)"}},
    )),
    "read_email": (_tool_read_email, _schema(
        "read_email", "Read the full content of a specific email by its ID.",
        {"msg_id": {"type": "string", "description": "The email message ID"}},
        ["msg_id"],
    )),
    "search_emails": (_tool_search_emails, _schema(
        "search_emails", "Search all emails (not just inbox) using Gmail search syntax. "
                         "Examples: 'from:alice subject:meeting', 'after:2024/01/01 has:attachment'.",
        {"query": {"type": "string", "description": "Gmail search query"},
         "max_results": {"type": "integer", "description": "Max results (default 20)"}},
        ["query"],
    )),
    "send_email": (_tool_send_email, _schema(
        "send_email", "Send a NEW email. Compose a clear, professional message. "
                      "IMPORTANT: Only call this when the user explicitly asks to send an email.",
        {"to": {"type": "string", "description": "Recipient email address"},
         "subject": {"type": "string", "description": "Email subject line"},
         "body": {"type": "string", "description": "Email body text"},
         "cc": {"type": "string", "description": "CC recipients (optional)"},
         "bcc": {"type": "string", "description": "BCC recipients (optional)"}},
        ["to", "subject", "body"],
    )),
    "reply_email": (_tool_reply_email, _schema(
        "reply_email", "Reply to an existing email (stays in the same thread). "
                       "IMPORTANT: Only call this when the user explicitly asks to reply.",
        {"msg_id": {"type": "string", "description": "The original email's message ID to reply to"},
         "body": {"type": "string", "description": "Reply body text"}},
        ["msg_id", "body"],
    )),
    "mark_read": (_tool_mark_read, _schema(
        "mark_read", "Mark one or more emails as read.",
        {"msg_ids": {"type": "string", "description": "Comma-separated message IDs to mark as read"}},
        ["msg_ids"],
    )),
}


def _run_tool(name: str, arguments: dict | str | None) -> tuple[bool, str]:
    """Run one email tool. Returns (ok, text)."""
    if name not in EMAIL_TOOLS:
        return False, f"Unknown tool '{name}'. Available: {', '.join(EMAIL_TOOLS)}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            arguments = {}
    func = EMAIL_TOOLS[name][0]
    allowed = func.__code__.co_varnames[:func.__code__.co_argcount]
    kwargs = {k: v for k, v in (arguments or {}).items() if k in allowed and v not in (None, "")}
    try:
        return True, _clip(func(**kwargs))
    except email_client.EmailError as exc:
        return False, f"Error: {exc}"
    except TypeError as exc:
        return False, f"Error: bad arguments for {name}: {exc}"
    except Exception as exc:
        return False, f"Error: {name} failed: {exc}"


# ------------------------------------------------------------------
# The agent loop
# ------------------------------------------------------------------

def run(question: str, history: list[dict], emit: Emit, cancel: threading.Event) -> dict:
    """Run the email agent. Streams progress through emit().

    Events:  step {n, tool, args, label}   step_result {n, ok, preview}
             token {text}                   (the final answer)
    Returns: {"answer", "steps": [...], "stats": {...}}
    """
    # Try to get the email address for the system prompt
    try:
        email_addr = email_client._service().users().getProfile(userId="me").execute().get("emailAddress", "your Gmail")
    except Exception:
        email_addr = "your Gmail"

    system = EMAIL_SYSTEM_PROMPT.format(email=email_addr)
    messages: list[dict] = [{"role": "system", "content": system}, *history,
                            {"role": "user", "content": question}]
    tools = [schema for _, schema in EMAIL_TOOLS.values()]
    steps: list[dict] = []
    stats: dict = {}
    last_call = None
    answer = ""

    for step_number in range(1, MAX_STEPS + 1):
        if cancel.is_set():
            break
        emit("status", {"text": "Thinking..." if step_number == 1 else f"Thinking (step {step_number})..."})
        _shrink_old_results(messages)
        reply = _chat_once(messages, tools, stats)
        calls = reply.get("tool_calls") or []

        if not calls:
            answer = reply.get("content", "")
            break

        messages.append({"role": "assistant", "content": reply.get("content", ""), "tool_calls": calls})
        for call in calls:
            if cancel.is_set():
                break
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            n = len(steps) + 1
            emit("step", {"n": n, "tool": name, "args": args, "label": _summarize_args(args)})

            signature = (name, json.dumps(args, sort_keys=True))
            started = time.perf_counter()
            if signature == last_call:
                ok, result = False, ("You just made this exact call. "
                                     "Use the result above, try something different, or give your final answer.")
            else:
                ok, result = _run_tool(name, args)
            last_call = signature
            elapsed = round((time.perf_counter() - started) * 1000)

            step = {"n": n, "tool": name, "args": args, "ok": ok, "ms": elapsed,
                    "preview": result[:1500]}
            steps.append(step)
            emit("step_result", {"n": n, "ok": ok, "ms": elapsed, "preview": step["preview"],
                                 "chars": len(result)})
            messages.append({"role": "tool", "content": result, "tool_name": name})
    else:
        if not cancel.is_set():
            emit("status", {"text": "Writing the answer..."})
            messages.append({"role": "user", "content": "You have used all your tool steps. "
                             "Give your best final answer now from what you found."})
            _shrink_old_results(messages)
            answer = _chat_once(messages, None, stats).get("content", "")

    if answer and not cancel.is_set():
        emit("status", {"text": "Writing..."})
        for i in range(0, len(answer), 24):
            if cancel.is_set():
                break
            emit("token", {"text": answer[i:i + 24]})
    return {"answer": answer, "steps": steps, "stats": stats}
