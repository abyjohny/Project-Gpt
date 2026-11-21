"""
agent.py - PHASE 5: Agent mode - the "think -> use a tool -> look -> repeat" loop.
=================================================================================

Chat and Cowork answer in ONE shot: search, then write. An agent works in STEPS:

    you:   "What does the auth module in D:\\dev\\shop do?"
    model: -> repo_overview("D:\\dev\\shop")              (tool call)
    us:    <tree, stack, README>                         (tool result)
    model: -> grep("login|token", file_glob="*.py")      (tool call)
    us:    <matches>
    model: -> read_file("app/auth.py")
    us:    <the file>
    model: "The auth module issues JWT tokens in ..."    (no tool call = final answer)

Ollama does the hard part: we send the tool descriptions (JSON schemas) along with the
conversation, and the model replies either with text or with "tool_calls". We run the
tools (rag/agent_tools.py - sandboxed + read-only) and send the results back.

Guard rails for a small 4B model:
    - MAX_STEPS: it cannot loop forever.
    - Old tool results are shortened, so the context window does not overflow.
    - The same call twice in a row is answered from memory with a nudge to move on.

Learn more: docs/learning/17_agent_mode.md
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable

from rag import agent_tools, config, ollama_client

MAX_STEPS = 12
KEEP_FULL_RESULTS = 3          # the newest N tool results stay complete; older ones are shortened
OLD_RESULT_CHARS = 700

AGENT_SYSTEM_PROMPT = """You are AJP (Agent Jose Pereira) in Agent mode: a careful assistant that can look at
the user's files through tools. You run fully locally on the user's computer.

Allowed folders (you can ONLY access these):
{folders}

How to work:
- Use tools to find facts before you answer. Never guess file contents or names.
- Do not describe what a file does from its name alone ("probably", "likely"): read it first,
  at least the top of the main entry files. If you skipped a file, say so.
- For a question about a project/codebase, call repo_overview first, then grep / find_files to locate
  the relevant code, then read_file the important parts.
- Prefer a few targeted calls over reading everything. Read only the files you need.
- For storage questions use disk_usage first, then drill into the largest folders.
  Use find_temp_junk to identify caches and temp files, find_large_files to spot space hogs,
  and find_duplicates if the user suspects repeated content.
- You are READ-ONLY in this version: you cannot edit, delete, move or run anything. If the user asks
  for a change, explain exactly what to change (file, line, and a code block with the new code).
  For cleanup, list every item and its size, then tell the user the commands to run themselves.
- If a tool returns an error, fix the arguments or try another approach. Do not repeat the same call.

Final answer:
- Clear and well structured (short headings / bullets), in plain language.
- Mention the files you looked at as `path/to/file.py` (with line numbers when useful).
- Be honest about what you did not check."""

Emit = Callable[[str, dict], None]


def _summarize_args(args: dict) -> str:
    parts = [f"{k}={v!r}" if not isinstance(v, str) or len(v) < 80 else f"{k}='{v[:77]}...'"
             for k, v in (args or {}).items()]
    return ", ".join(parts)


def _shrink_old_results(messages: list[dict]) -> None:
    """Keep the newest tool results whole; shorten older ones to save context."""
    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_indexes[:-KEEP_FULL_RESULTS]:
        content = messages[i]["content"]
        if len(content) > OLD_RESULT_CHARS and not content.endswith("[shortened]"):
            messages[i]["content"] = content[:OLD_RESULT_CHARS] + "\n... [shortened]"


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _chat_once(messages: list[dict], tools: list[dict] | None, stats: dict) -> dict:
    """One non-streamed call to Ollama. Returns the assistant message (content + tool_calls)."""
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


def run(question: str, history: list[dict], emit: Emit, cancel: threading.Event) -> dict:
    """Run the agent until it gives a final answer. Streams progress through emit().

    Events:  step {n, tool, args, label}   step_result {n, ok, preview}
             token {text}                   (the final answer)
    Returns: {"answer", "steps": [...], "stats": {...}}
    """
    folders = agent_tools.list_folders()
    system = AGENT_SYSTEM_PROMPT.format(
        folders="\n".join(f"- {f}" for f in folders) if folders else "- (none yet: tell the user to add a folder)")
    messages: list[dict] = [{"role": "system", "content": system}, *history,
                            {"role": "user", "content": question}]
    tools = agent_tools.schemas()
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

        if not calls:                                   # no tool wanted -> this is the final answer
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
                ok, result = False, ("You just made this exact call and already have the result above. "
                                     "Use it, try something different, or give your final answer.")
            else:
                ok, result = agent_tools.run_tool(name, args)
            last_call = signature
            elapsed = round((time.perf_counter() - started) * 1000)

            step = {"n": n, "tool": name, "args": args, "ok": ok, "ms": elapsed,
                    "preview": result[:1500]}
            steps.append(step)
            emit("step_result", {"n": n, "ok": ok, "ms": elapsed, "preview": step["preview"],
                                 "chars": len(result)})
            messages.append({"role": "tool", "content": result, "tool_name": name})
    else:
        # Out of steps: ask for the best answer with what we have, tools switched off.
        if not cancel.is_set():
            emit("status", {"text": "Writing the answer..."})
            messages.append({"role": "user", "content": "You have used all your tool steps. "
                             "Give your best final answer now from what you found."})
            _shrink_old_results(messages)
            answer = _chat_once(messages, None, stats).get("content", "")

    if answer and not cancel.is_set():
        emit("status", {"text": "Writing..."})
        # The answer came in one piece (tool calling is not streamed); send it in small
        # pieces so the UI renders it the same way as the other modes.
        for i in range(0, len(answer), 24):
            if cancel.is_set():
                break
            emit("token", {"text": answer[i:i + 24]})
    return {"answer": answer, "steps": steps, "stats": stats}
