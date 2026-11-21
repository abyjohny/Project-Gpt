"""
reranker.py - PHASE 2: A second, smarter "judge" that re-orders the shortlist.
==============================================================================

THE PROBLEM WITH FAST SEARCH
    Meaning search turns the question and each chunk into numbers SEPARATELY,
    then compares the numbers. It is fast (milliseconds for 100,000 chunks)
    but a bit "blurry": the question and the chunk never actually meet.

THE RE-RANKER (a "cross-encoder")
    Reads the question and ONE chunk TOGETHER, word by word, and answers:
        "How well does this chunk answer this question?"  ->  0.0 ... 1.0

    It notices things embeddings miss:
        Q: "When can a preauth NOT be cancelled?"
        chunk A: "A preauth can be cancelled at any time..."      embeddings: very similar!
        chunk B: "Cancellation is blocked once the claim is..."   re-ranker: THIS is the answer

    It is much slower (it must read every pair), so we only use it on a SHORTLIST:

        100,000 chunks --[fast search]--> 20-40 candidates --[re-ranker]--> best 5

    Like hiring: a quick CV screen first, then proper interviews for the shortlist.

The model (BAAI/bge-reranker-v2-m3) is downloaded once from Hugging Face
(~2.2 GB) and then runs 100% locally on your GPU (or CPU).

Learn more: docs/learning/11_reranking.md
"""

from __future__ import annotations

import os
import threading

from rag import config
from rag.vector_store import SearchResult

# Privacy + quieter logs for the Hugging Face libraries.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_model = None
_tokenizer = None
_device = "cpu"
_load_lock = threading.Lock()   # the web server may warm up and answer at the same time


def is_available() -> bool:
    """True if PyTorch + transformers are installed (needed to run the re-ranker)."""
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except Exception:   # not installed (ImportError) or broken install (e.g. DLL errors)
        return False


def device_name() -> str:
    return _device


def _load() -> None:
    """Load the model into memory the first time we need it (takes a few seconds)."""
    if _model is not None:
        return
    with _load_lock:
        _load_locked()


def _load_locked() -> None:
    global _model, _tokenizer, _device
    if _model is not None:   # another thread finished loading while we waited
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()   # hide the "Loading weights: 37%..." bar

    wanted = config.RERANKER_DEVICE
    _device = "cuda" if (wanted in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"

    # Offline-first: after the first download the model lives on your disk
    # (~/.cache/huggingface). local_files_only=True uses that copy without
    # contacting the internet at all. Only if it's missing do we download it.
    try:
        _tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(
            config.RERANKER_MODEL, local_files_only=True)
    except OSError:   # not downloaded yet -> fetch it once
        _tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(config.RERANKER_MODEL)
    if _device == "cuda":
        _model = _model.half()   # 16-bit numbers: half the GPU memory, practically same quality
    _model.to(_device).eval()     # eval() = "we are using the model, not training it"


def warm_up() -> str:
    """Download (first time) + load the model. Returns the device it runs on."""
    _load()
    return _device


def score_pairs(question: str, texts: list[str]) -> list[float]:
    """The core idea in one function: judge every (question, text) pair -> 0..1."""
    import torch

    _load()
    scores: list[float] = []
    with torch.no_grad():   # no training -> no need to remember gradients (saves memory)
        for start in range(0, len(texts), config.RERANKER_BATCH_SIZE):
            batch = texts[start:start + config.RERANKER_BATCH_SIZE]
            # The model reads BOTH texts at once: "[question] [SEP] [chunk]"
            inputs = _tokenizer(
                [[question, text] for text in batch],
                padding=True, truncation=True,
                max_length=config.RERANKER_MAX_LENGTH, return_tensors="pt",
            ).to(_device)
            raw_scores = _model(**inputs).logits.view(-1).float()   # any number, e.g. -7.2 or 4.1
            scores.extend(torch.sigmoid(raw_scores).tolist())        # squash into 0..1
    return scores


def rerank(question: str, candidates: list[SearchResult]) -> list[SearchResult]:
    """Give every candidate a rerank_score and sort best-first."""
    if not candidates:
        return []
    for result, score in zip(candidates, score_pairs(question, [c.text for c in candidates])):
        result.rerank_score = round(score, 4)
    return sorted(candidates, key=lambda r: r.rerank_score, reverse=True)
