"""
embedder.py - STEP 3: Turn text into "meaning numbers" (embeddings / vectors).
=============================================================================

An EMBEDDING is a long list of numbers that describes what a text MEANS.
nomic-embed-text gives 768 numbers for every piece of text.

The magic property:
    texts with SIMILAR MEANING get SIMILAR NUMBERS,
    even if they use completely different words.

    "cancel the pre-authorization"  -> [0.81, 0.12, -0.33, ...]  \\  close
    "preauth cancellation request"  -> [0.79, 0.15, -0.30, ...]  /  together
    "ABHA address creation"         -> [-0.20, 0.66, 0.41, ...]     far away

IMPORTANT RULE: documents and questions MUST be embedded with the SAME model.
Different models speak different "number languages" - mixing them is like
comparing a map in kilometres with a map in miles.

Learn more: docs/learning/04_embeddings.md
"""

from __future__ import annotations

from typing import Callable

from rag import config, ollama_client


def embed_documents(
    texts: list[str],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Embed many document chunks, a batch at a time (faster than one by one)."""
    vectors: list[list[float]] = []
    batch_size = config.EMBED_BATCH_SIZE

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        # Tell the model "these are documents to be stored".
        batch = [config.EMBED_DOCUMENT_PREFIX + t for t in batch]
        vectors.extend(ollama_client.embed(batch))
        if on_progress:
            on_progress(min(start + batch_size, len(texts)), len(texts))

    return vectors


def embed_query(question: str) -> list[float]:
    """Embed ONE user question. Tell the model "this is a search question"."""
    return ollama_client.embed([config.EMBED_QUERY_PREFIX + question])[0]
