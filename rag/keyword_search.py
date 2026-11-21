"""
keyword_search.py - PHASE 2: Find chunks by EXACT WORDS (the BM25 algorithm).
==============================================================================

WHY DO WE NEED THIS? Meaning search (embeddings) is great at:
    "withdraw a preauth"  ~  "cancel a pre-authorization"     (same meaning, different words)

...but it is surprisingly BAD at exact things like:
    error codes    "E-1042"         policy numbers  "POL-88213"
    names          "Aby Johny"      versions        "v2.3.1"
    rare terms     "ABHA", "NHCX"   field names     "claim_status"

To an embedding model, "E-1042" and "E-1043" look almost identical.
To a keyword search, they are completely different words. We want both skills!

HOW BM25 WORKS (think of the index at the back of a textbook):
    1. TOKENIZE  - cut every chunk into lowercase words   "Send Cancel-Request" -> [send, cancel-request, cancel, request]
    2. INDEX     - for every word, remember which chunks contain it and how often
                   ("inverted index", like a book index:  preauth -> pages 3, 7, 12)
    3. SCORE     - for a question, add up points for every question-word found in a chunk:
                     + MORE points if the word is RARE in the whole library   (IDF)
                       "the" is everywhere -> worthless; "ABHA" is rare -> very valuable
                     + MORE points if the word appears OFTEN in that chunk    (TF)
                       ...but with diminishing returns (10 mentions is not 10x better)
                     + a small bonus for SHORT chunks                          (length normalisation)
                       a word in a 2-line chunk matters more than in a 2-page chunk

BM25 = "Best Matching, attempt 25". It is from the 1990s and still powers
search engines like Elasticsearch. No AI, no GPU - just counting, very fast.

Learn more: docs/learning/09_keyword_search.md
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from rag import vector_store
from rag.vector_store import SearchResult

# Words so common they tell us nothing about the topic. We ignore them.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "should", "so", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}

# A "word" = letters/digits, optionally joined by - _ . /   (so "E-1042", "v2.3", "claim_id" stay whole)
_WORD = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Turn text into a list of searchable words.

    "Send the Cancel-Request for E-1042"  ->  ['send', 'cancel-request', 'cancel', 'request', 'e-1042', 'e', '1042']

    Joined words are kept whole AND split into parts, so a search for
    "cancel request" still finds "Cancel-Request" (and "E 1042" finds "E-1042").
    """
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue
        tokens.append(word)
        parts = re.split(r"[-_./]", word)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


class BM25Index:
    """A tiny search engine. Build it once from all chunks, then search many times."""

    def __init__(self, ids: list[str], texts: list[str], metadatas: list[dict],
                 k1: float = 1.5, b: float = 0.75):
        # k1 = how quickly repeated words stop adding points (diminishing returns)
        # b  = how much we favour short chunks (0 = not at all, 1 = fully)
        self.k1, self.b = k1, b
        self.ids, self.texts, self.metadatas = ids, texts, metadatas

        # The "book index": word -> {chunk number: how many times the word appears there}
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.lengths: list[int] = []
        for chunk_number, text in enumerate(texts):
            tokens = tokenize(text)
            self.lengths.append(len(tokens))
            for word, count in Counter(tokens).items():
                self.postings[word][chunk_number] = count

        self.total_chunks = len(texts)
        self.average_length = (sum(self.lengths) / self.total_chunks) if self.total_chunks else 0.0

        # IDF = how RARE (= how valuable) each word is across the whole library.
        #   word in 1 of 1000 chunks   -> IDF ~ 6.5  (very informative)
        #   word in 900 of 1000 chunks -> IDF ~ 0.1  (almost useless)
        self.idf = {
            word: math.log(1 + (self.total_chunks - len(chunks) + 0.5) / (len(chunks) + 0.5))
            for word, chunks in self.postings.items()
        }

    def search(self, question: str, top_k: int = 20) -> list[SearchResult]:
        """Score every chunk that contains at least one question word; return the best."""
        scores: dict[int, float] = defaultdict(float)

        for word in set(tokenize(question)):
            if word not in self.postings:
                continue                       # nobody uses this word -> no points
            idf = self.idf[word]
            for chunk_number, tf in self.postings[word].items():
                length_ratio = self.lengths[chunk_number] / (self.average_length or 1)
                # The BM25 formula: rare word (idf) x frequent-but-saturating (tf part) x length bonus
                scores[chunk_number] += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * length_ratio)
                )

        best = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.texts[n], metadata=self.metadatas[n], id=self.ids[n],
                keyword_score=round(score, 3), found_by=["keyword"],
            )
            for n, score in best
        ]


# ----------------------------------------------------------------
# A shared index for the whole program, built from the vector DB.
# ----------------------------------------------------------------
_index: BM25Index | None = None
_index_size = -1


def get_index(force_rebuild: bool = False) -> BM25Index:
    """Build the index from every chunk in ChromaDB (only again if the chunk count changed)."""
    global _index, _index_size
    current_size = vector_store.count()
    if force_rebuild or _index is None or current_size != _index_size:
        ids, texts, metadatas = vector_store.get_all()
        _index = BM25Index(ids, texts, metadatas)
        _index_size = current_size
    return _index


def search(question: str, top_k: int = 20) -> list[SearchResult]:
    return get_index().search(question, top_k=top_k)
