"""
generator.py - STEPS 7 & 8: Build the prompt and let Qwen write a grounded answer.
=================================================================================

This is the "A" (Augmented) and "G" (Generation) in RAG.

[7] AUGMENT: we build a message for the AI that contains
      - RULES      ("answer ONLY from the sources, cite them, say if you don't know")
      - SOURCES    (the chunks the retriever found, numbered [1], [2], ...)
      - QUESTION   (what you asked)

[8] GENERATE: Qwen reads all of that and writes the answer.

The AI never "searches" anything itself. WE put the right pages in front
of it, and it reads them - exactly like an open-book exam.

Learn more: docs/learning/07_generation.md
"""

from __future__ import annotations

from typing import Iterator

from rag import ollama_client
from rag.vector_store import SearchResult

# The "system prompt" = permanent instructions that shape the AI's behaviour.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided sources.

Rules:
1. Use only the information inside the SOURCES section. Do not use outside knowledge.
2. After every fact, cite the source number in square brackets, like [1] or [2][3].
3. If the sources do not contain the answer, say exactly:
   "I could not find this in your documents." Then stop. Never guess or invent.
4. Be clear and well structured. Use bullet points or steps when it helps.
5. If sources disagree with each other, point out the disagreement."""

NO_CONTEXT_REPLY = (
    "I could not find anything relevant in your documents for that question. "
    "Try rephrasing it, or check that the right files were ingested (`python ingest.py --list`)."
)

# Phase 4 - "Cowork" mode in the web UI: the AI may also read LIVE WEB results.
# Still grounded and cited, but it is allowed to fill gaps with general knowledge
# as long as it SAYS so - a coworker, not a strict open-book exam.
COWORK_SYSTEM_PROMPT = """You are a capable research coworker. You answer using the provided SOURCES:
some come from the user's own documents, others are live web search results.

Rules:
1. Prefer the SOURCES. After every fact taken from them, cite the number in square brackets, like [1] or [2][3].
2. The user's documents are the most trusted. If a web result disagrees with them, say so.
3. Web results can be outdated or incomplete - mention it when it matters (dates, prices, versions).
4. If the sources do not cover something, you may add your own general knowledge, but mark it
   clearly, e.g. "(general knowledge, not from the sources)". Never invent citations.
5. Be clear and well structured. Use bullet points, steps or short tables when it helps."""


SMALL_TALK_SYSTEM_PROMPT = """You are Agent Jose Pereira (AJP for short), a friendly and capable private document assistant.
The user is just being social (greeting, thanking you, saying goodbye, or asking who you are).

Rules:
1. Respond warmly and naturally in 1-3 sentences, like a helpful colleague would.
2. If it's a greeting, greet back and briefly mention you're ready to help with their documents.
3. If they thank you, acknowledge it warmly.
4. If they ask who you are, say you're Agent Jose Pereira (AJP), their private document assistant that can answer
   questions about any documents they've uploaded — everything runs locally on their machine.
5. Do NOT cite sources, do NOT use bullet points, do NOT write an essay. Keep it short and human."""


def generate_small_talk(
    message: str,
    history: list[dict] | None = None,
    stats: dict | None = None,
) -> Iterator[str]:
    """Stream a short, friendly reply to a greeting / thanks / goodbye."""
    messages = [{"role": "system", "content": SMALL_TALK_SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})
    yield from ollama_client.chat(messages, stats=stats)


def format_source_label(meta: dict) -> str:
    """'NHCX_Spec.pdf, page 12'  or just  'notes.md'."""
    page = meta.get("page", 0)
    return f"{meta.get('source')}, page {page}" if page else str(meta.get("source"))


def format_source_detail(meta: dict) -> str:
    """Label plus the section 'address' (Phase 3), e.g. 'Handbook.docx  >  4.1 Provider Codes'."""
    label = format_source_label(meta)
    section = meta.get("section")
    if section:
        last = section.split(" > ")[-1]          # the most specific heading is enough
        label += f"  >  {last[:70]}"
    return label


def build_context(results: list[SearchResult]) -> str:
    """Turn retrieved chunks into a numbered SOURCES block."""
    if not results:
        return "(no sources found)"
    blocks = []
    for number, result in enumerate(results, start=1):
        meta = result.metadata
        if meta.get("type") == "web":
            label = f"web: {meta.get('source')} - {meta.get('url')}"
        else:
            label = f"document: {format_source_label(meta)}"
        blocks.append(f"[{number}] ({label})\n{result.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(
    question: str,
    results: list[SearchResult],
    history: list[dict] | None = None,
    hint: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Assemble the full conversation we send to Qwen.

    `hint` (Phase 3) explains a follow-up question, made by memory.prepare(), e.g.
        "In this conversation, that means: What does payer response code 2 mean?"
    We show the original question AND the hint, so the AI answers in the flow of
    the chat but knows exactly what is meant.
    """
    question_block = f"QUESTION: {question}"
    if hint:
        question_block += f"\n({hint})"
    user_message = (
        f"SOURCES:\n\n{build_context(results)}\n\n"
        f"====================\n\n"
        f"{question_block}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])   # earlier questions/answers (for follow-ups)
    messages.append({"role": "user", "content": user_message})
    return messages


def generate_answer(
    question: str,
    results: list[SearchResult],
    history: list[dict] | None = None,
    hint: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    allow_empty: bool = False,
    stats: dict | None = None,
) -> Iterator[str]:
    """Stream the answer word by word. If nothing was retrieved, don't even ask the AI.

    `allow_empty=True` (Cowork mode) asks the AI anyway - it may then answer from
    general knowledge, clearly marked as such by COWORK_SYSTEM_PROMPT.
    """
    if not results and not allow_empty:
        yield NO_CONTEXT_REPLY
        return
    yield from ollama_client.chat(build_messages(question, results, history, hint, system_prompt), stats=stats)
