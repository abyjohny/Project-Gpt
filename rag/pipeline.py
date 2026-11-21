"""
pipeline.py - PHASE 3: "Where do the sources for THIS message come from?"
========================================================================

chat.py, evaluate.py and (soon) the web UI all need the same decision, so it
lives in ONE place:

    message ──► is it a CLARIFICATION of the last answer?   ("didn't get it, clarify?")
                    yes ─► re-use the last answer's sources        (no search, instant)
                    no  ─► memory.prepare() ─► search ─► re-rank
                              │
                              └─ found NOTHING, but it's a follow-up?
                                     ─► fall back to the last answer's sources

Why the fallback? "I could not find anything relevant" in the MIDDLE of a
conversation is a dead end. If the message leans on the conversation, the
chunks we just talked about are the best guess - and Qwen is still told to
say "I could not find this in your documents" if they don't answer it.

Learn more: docs/learning/15_conversation_memory.md
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag import config, memory, retriever
from rag.vector_store import SearchResult

CLARIFY_HINT = ("The user did not fully understand your previous answer. Explain it again more "
                "clearly and simply, step by step, using plain words and the same sources. "
                "If it helps, give a short example taken from the sources.")

FALLBACK_HINT = ("The search found nothing new for this message, so these are the sources of your "
                 "previous answer. Use them if they answer the question.")


@dataclass
class SourcePlan:
    """Everything the answer step needs, plus a note of HOW we got the sources."""

    search_query: str                       # what we searched for (or would have)
    hint: str | None                        # extra note for Qwen about this message
    results: list[SearchResult] = field(default_factory=list)
    origin: str = "search"                  # "search" | "clarify" | "fallback"
    trace: dict = field(default_factory=dict)


def find_sources(
    question: str,
    mem: memory.ConversationMemory,
    top_k: int = config.TOP_K,
    mode: str = config.RETRIEVAL_MODE,
    use_reranker: bool = config.USE_RERANKER,
) -> SourcePlan:
    """Decide where the sources come from, and fetch them."""
    previous = mem.previous_sources()

    # 1. "didn't understand, clarify?" -> same sources as last time, no search at all
    if previous and mem.rewrite_mode != "off" and memory.is_clarification(question):
        return SourcePlan(mem.turns[-1].search_query, CLARIFY_HINT, previous, origin="clarify")

    # 2. normal path: give follow-ups their context, then search
    trace: dict = {}
    started = time.perf_counter()
    search_query, hint = mem.prepare(question)
    trace["prepare_ms"] = (time.perf_counter() - started) * 1000
    results = retriever.retrieve(search_query, top_k=top_k, mode=mode, use_reranker=use_reranker, trace=trace)
    if results:
        return SourcePlan(search_query, hint, results, origin="search", trace=trace)

    # 3. nothing found, but the message leans on the conversation -> last sources
    if previous and mem.needs_rewrite(question):
        hint = f"{hint} {FALLBACK_HINT}" if hint else FALLBACK_HINT
        return SourcePlan(search_query, hint, previous, origin="fallback", trace=trace)

    return SourcePlan(search_query, hint, [], origin="search", trace=trace)
