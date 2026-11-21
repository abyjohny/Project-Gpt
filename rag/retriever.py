"""
retriever.py - STEPS 5 & 6: Question -> the most relevant chunks.
================================================================

PHASE 1 (simple):
    question -> embed -> vector DB -> top 5

PHASE 2 (smart) - the full pipeline:

    "How do I cancel preauth E-1042?"
            |
            +-----------------------------+
            |                             |
     [A] MEANING search             [B] KEYWORD search
     (embeddings, vector DB)        (BM25, exact words)
     good at: paraphrases           good at: codes, names, rare terms
            |                             |
       ~20 candidates                ~20 candidates
            |                             |
            +------------+----------------+
                         |
              [C] MERGE the two lists  (Reciprocal Rank Fusion)
                         |
              [D] RE-RANK the shortlist (cross-encoder "judge")
                         |
              [E] throw away weak chunks, keep the best TOP_K
                         |
                   chunks for the AI

Every step can be switched off in config.py (RETRIEVAL_MODE, USE_RERANKER)
so you can SEE what each one adds. Try it in chat with /mode and /rerank.

If retrieval finds the wrong pages, even the smartest AI will give a bad answer.
So when answers are bad, ALWAYS check retrieval first!

Learn more: docs/learning/06_retrieval.md, 09_keyword_search.md,
            10_hybrid_search.md, 11_reranking.md
"""

from __future__ import annotations

import time

from rag import config, embedder, keyword_search, reranker, vector_store
from rag.vector_store import SearchResult

VALID_MODES = ("vector", "keyword", "hybrid")


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[SearchResult]], k: int = config.RRF_K
) -> list[SearchResult]:
    """Merge several ranked lists into one, using only the POSITIONS (ranks).

    Why ranks and not scores? The two searches speak different "languages":
    similarity is 0..1, BM25 can be 0..30. You can't add apples and oranges.
    But "1st place", "2nd place" means the same thing in both lists.

    Each chunk earns  1 / (k + rank)  points from EVERY list it appears in:
        1st place -> 1/61 = 0.0164      10th place -> 1/70 = 0.0143
    A chunk found by BOTH searches collects points twice -> rises to the top.
    """
    merged: dict[str, SearchResult] = {}
    for method, results in ranked_lists.items():
        for rank, result in enumerate(results, start=1):
            entry = merged.setdefault(result.id, result)
            if entry is not result:   # same chunk found again by another method -> combine info
                entry.similarity = entry.similarity if entry.similarity is not None else result.similarity
                entry.keyword_score = entry.keyword_score if entry.keyword_score is not None else result.keyword_score
                entry.found_by = sorted(set(entry.found_by) | set(result.found_by))
            entry.fused_score += 1.0 / (k + rank)

    for entry in merged.values():
        entry.fused_score = round(entry.fused_score, 5)
    return sorted(merged.values(), key=lambda r: r.fused_score, reverse=True)


def retrieve(
    question: str,
    top_k: int = config.TOP_K,
    mode: str = config.RETRIEVAL_MODE,
    use_reranker: bool = config.USE_RERANKER,
    min_similarity: float = config.MIN_SIMILARITY,
    trace: dict | None = None,
) -> list[SearchResult]:
    """Find the best chunks for a question.

    Pass a dict as `trace` to get a look behind the scenes: how many
    candidates each step produced and how long each step took (in ms).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    trace = trace if trace is not None else {}
    use_reranker = use_reranker and reranker.is_available()
    trace["reranker_used"] = use_reranker

    # When something AFTER the search will filter/re-order, fetch a bigger shortlist.
    pool = config.CANDIDATES_PER_SEARCH if (use_reranker or mode == "hybrid") else top_k
    ranked_lists: dict[str, list[SearchResult]] = {}

    # [A] meaning search
    if mode in ("vector", "hybrid"):
        started = time.perf_counter()
        question_vector = embedder.embed_query(question)
        found = vector_store.search(question_vector, top_k=pool)
        ranked_lists["meaning"] = [r for r in found if r.similarity >= min_similarity]
        trace["meaning_ms"] = (time.perf_counter() - started) * 1000
        trace["meaning_found"] = len(ranked_lists["meaning"])

    # [B] keyword search
    if mode in ("keyword", "hybrid"):
        started = time.perf_counter()
        ranked_lists["keyword"] = keyword_search.search(question, top_k=pool)
        trace["keyword_ms"] = (time.perf_counter() - started) * 1000
        trace["keyword_found"] = len(ranked_lists["keyword"])

    # [C] merge (only needed when we have two lists)
    if len(ranked_lists) > 1:
        candidates = reciprocal_rank_fusion(ranked_lists)
    else:
        candidates = next(iter(ranked_lists.values()))
    trace["candidates"] = len(candidates)

    # [D] re-rank + [E] filter
    if use_reranker and candidates:
        started = time.perf_counter()
        candidates = reranker.rerank(question, candidates)
        trace["rerank_ms"] = (time.perf_counter() - started) * 1000
        trace["best_rerank"] = candidates[0].rerank_score
        kept = [c for c in candidates if c.rerank_score >= config.MIN_RERANK_SCORE]
        # The best chunk is relevant -> the question IS about the documents. Keep its
        # runners-up too: they often hold the rest of the answer (see config.RERANK_MIN_KEEP).
        if kept and len(kept) < config.RERANK_MIN_KEEP:
            kept = candidates[:config.RERANK_MIN_KEEP]
        candidates = kept

    return candidates[:top_k]
