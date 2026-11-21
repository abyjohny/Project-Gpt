"""
sessions.py - PHASE 4: Saved conversations for the web UI.
==========================================================

chat.py keeps ONE conversation in memory and forgets it when you quit.
The web UI needs more: a sidebar of past chats you can reopen, even after a restart.

Each conversation is one small JSON file in data/sessions/:

    data/sessions/3f9c...e1.json
        { "id", "title", "created_at", "updated_at",
          "messages": [ {role: "user", content}, {role: "assistant", content, sources, ...}, ... ] }

When a chat is reopened we rebuild its ConversationMemory from the last few
messages - INCLUDING the sources of each answer - so follow-ups ("and the second one?")
and clarifications ("didn't get it") keep working exactly like in chat.py.

Learn more: docs/learning/16_web_ui.md
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from rag import config, memory
from rag.vector_store import SearchResult

SESSIONS_DIR = config.PROJECT_ROOT / "data" / "sessions"
TITLE_MAX_CHARS = 60
_ID = re.compile(r"^[0-9a-f]{32}$")     # only our own ids -> no "../../" tricks in file names


def result_to_dict(result: SearchResult) -> dict:
    return asdict(result)


def dict_to_result(data: dict) -> SearchResult:
    fields = {k: data.get(k) for k in ("text", "metadata", "similarity", "id", "keyword_score",
                                       "fused_score", "rerank_score", "found_by")}
    fields["metadata"] = fields["metadata"] or {}
    fields["found_by"] = fields["found_by"] or []
    fields["fused_score"] = fields["fused_score"] or 0.0
    fields["id"] = fields["id"] or ""
    return SearchResult(**fields)


def make_title(first_question: str) -> str:
    text = re.sub(r"\s+", " ", first_question).strip()
    return text if len(text) <= TITLE_MAX_CHARS else text[:TITLE_MAX_CHARS].rsplit(" ", 1)[0] + "..."


class Session:
    """One conversation: its messages (for the UI) + its memory (for follow-ups)."""

    def __init__(self, session_id: str | None = None, title: str = "New chat",
                 created_at: float | None = None, updated_at: float | None = None,
                 messages: list[dict] | None = None) -> None:
        self.id = session_id or uuid.uuid4().hex
        self.title = title
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.messages: list[dict] = messages or []
        self.lock = threading.Lock()
        self.memory = self._rebuild_memory()

    # ---- memory ----
    def _rebuild_memory(self) -> memory.ConversationMemory:
        mem = memory.ConversationMemory()
        pending_question = None
        for message in self.messages:
            if message["role"] == "user":
                pending_question = message["content"]
            elif message["role"] == "assistant" and pending_question is not None and not message.get("error"):
                sources = [dict_to_result(s) for s in message.get("sources", [])]
                mem.add(pending_question, message.get("search_query") or pending_question,
                        message["content"], sources)
                pending_question = None
        return mem

    # ---- bookkeeping ----
    def add_exchange(self, question: str, answer: dict) -> None:
        if not self.messages:
            self.title = make_title(question)
        self.messages.append({"role": "user", "content": question, "created_at": time.time()})
        self.messages.append({"role": "assistant", "created_at": time.time(), **answer})
        self.updated_at = time.time()

    def summary(self) -> dict:
        return {"id": self.id, "title": self.title, "updated_at": self.updated_at,
                "message_count": len(self.messages)}

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "created_at": self.created_at,
                "updated_at": self.updated_at, "messages": self.messages}

    # ---- disk ----
    @property
    def path(self) -> Path:
        return SESSIONS_DIR / f"{self.id}.json"

    def save(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        temp.replace(self.path)   # write-then-rename: a crash never leaves half a file


class SessionStore:
    """All conversations. Loaded lazily from disk and cached in memory."""

    def __init__(self) -> None:
        self._cache: dict[str, Session] = {}
        self._lock = threading.Lock()

    @staticmethod
    def valid_id(session_id: str) -> bool:
        return bool(_ID.match(session_id or ""))

    def create(self) -> Session:
        session = Session()
        with self._lock:
            self._cache[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        if not self.valid_id(session_id):
            return None
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]
            path = SESSIONS_DIR / f"{session_id}.json"
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            session = Session(data["id"], data.get("title", "Chat"), data.get("created_at"),
                              data.get("updated_at"), data.get("messages", []))
            self._cache[session_id] = session
            return session

    def list(self) -> list[dict]:
        summaries: dict[str, dict] = {}
        if SESSIONS_DIR.exists():
            for path in SESSIONS_DIR.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    summaries[data["id"]] = {"id": data["id"], "title": data.get("title", "Chat"),
                                             "updated_at": data.get("updated_at", 0),
                                             "message_count": len(data.get("messages", []))}
                except (OSError, ValueError, KeyError):
                    continue   # a broken file must not break the sidebar
        with self._lock:
            for session in self._cache.values():
                if session.messages:            # unsaved empty chats are not listed
                    summaries[session.id] = session.summary()
        return sorted(summaries.values(), key=lambda s: s["updated_at"], reverse=True)

    def delete(self, session_id: str) -> bool:
        if not self.valid_id(session_id):
            return False
        with self._lock:
            self._cache.pop(session_id, None)
        path = SESSIONS_DIR / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
