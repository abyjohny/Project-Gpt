"""
ollama_client.py - Talk to Ollama using plain HTTP requests.
===========================================================

Ollama runs a small web server on your computer (http://localhost:11434).
We "talk" to it the same way a browser talks to a website: send a request,
get a response in JSON. No magic library - just `requests`.

Two things we ask Ollama to do:

  1. EMBED  ->  POST /api/embed
     "Here is some text, give me its meaning-numbers (a vector)."

  2. CHAT   ->  POST /api/chat
     "Here is a conversation, please write the next reply."

Learn more: docs/learning/00_setup_and_ollama.md
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from rag import config


class OllamaError(RuntimeError):
    """Raised when Ollama is not running or returns an error."""


def _post(endpoint: str, payload: dict, stream: bool = False) -> requests.Response:
    url = f"{config.OLLAMA_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, stream=stream, timeout=600)
    except requests.ConnectionError as exc:
        raise OllamaError(
            "Cannot reach Ollama at " + config.OLLAMA_URL
            + ". Is it running? Start the Ollama app or run:  ollama serve"
        ) from exc

    if response.status_code != 200:
        raise OllamaError(f"Ollama error {response.status_code}: {response.text}")
    return response


def is_running() -> bool:
    """Quick health-check: is the Ollama server up?"""
    try:
        return requests.get(config.OLLAMA_URL, timeout=3).status_code == 200
    except requests.ConnectionError:
        return False


def list_models() -> list[str]:
    """Names of the models you have downloaded (like `ollama list`)."""
    response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
    return [m["name"] for m in response.json().get("models", [])]


def embed(texts: list[str], model: str = config.EMBED_MODEL) -> list[list[float]]:
    """Turn a list of texts into a list of vectors (one vector per text).

    Example:
        embed(["hello world"])  ->  [[0.012, -0.551, 0.309, ... 768 numbers]]
    """
    response = _post("/api/embed", {"model": model, "input": texts})
    return response.json()["embeddings"]


def chat(
    messages: list[dict],
    model: str = config.CHAT_MODEL,
    temperature: float = config.TEMPERATURE,
    stream: bool = True,
    extra_options: dict | None = None,
    stats: dict | None = None,
) -> Iterator[str]:
    """Send a conversation to the chat model and yield the reply piece by piece.

    `messages` looks like:
        [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "What is RAG?"},
        ]
    A message may also carry pictures: {"role": "user", "content": "...", "images": [base64]}

    With stream=True the words arrive one by one (like ChatGPT typing),
    instead of waiting for the full answer.

    Pass a dict as `stats` to receive Ollama's timing numbers when the reply is
    finished (eval_count = tokens written, eval_duration = nanoseconds spent).
    The web UI uses this to show "tokens per second".
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": config.ENABLE_THINKING,
        "options": {
            "temperature": temperature,
            "num_ctx": config.CONTEXT_WINDOW,
            **(extra_options or {}),
        },
    }
    response = _post("/api/chat", payload, stream=stream)

    if not stream:
        data = response.json()
        _copy_stats(data, stats)
        yield data["message"]["content"]
        return

    # Streaming: Ollama sends one small JSON object per line.
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            raise OllamaError(data["error"])
        piece = data.get("message", {}).get("content", "")
        if piece:
            yield piece
        if data.get("done"):
            _copy_stats(data, stats)
            break


_STAT_KEYS = ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "total_duration")


def _copy_stats(data: dict, stats: dict | None) -> None:
    if stats is None:
        return
    for key in _STAT_KEYS:
        if key in data:
            stats[key] = data[key]
