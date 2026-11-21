"""
vector_store.py - STEP 4: Store vectors and search them by meaning.
==================================================================

A VECTOR DATABASE is a special database built for one job:
    "Here is a vector - find me the stored vectors closest to it."

Normal database:  WHERE title = 'preauth'        (exact match)
Vector database:  "closest in MEANING to 'how do I cancel a preauth?'"

For every chunk we store 4 things:
    id         -> the chunk's fingerprint (so we can update/delete it)
    embedding  -> its 768 meaning-numbers
    document   -> the original text (to show the AI and you)
    metadata   -> source file, page number, chunk number

We use ChromaDB: it runs INSIDE our Python program (no server to install)
and saves everything to the data/vector_db folder.

"Closeness" is measured with COSINE similarity:
    1.0  = pointing the same way  = same meaning
    0.0  = unrelated
ChromaDB reports "distance" = 1 - similarity, so we convert it back.

Learn more: docs/learning/05_vector_database.md
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings

from rag import config
from rag.chunker import Chunk


@dataclass
class SearchResult:
    """One chunk found by a search, plus every score it collected along the way.

    Phase 1 only had `similarity`. Phase 2 adds more "judges":
        similarity     -> meaning search score   (0..1, from the vector DB)
        keyword_score  -> exact-word score       (BM25, 0..unlimited)
        fused_score    -> merged rank score      (hybrid search)
        rerank_score   -> the re-ranker's verdict (0..1, the most trustworthy)
    """

    text: str
    metadata: dict
    similarity: float | None = None   # 0..1, higher = more relevant
    id: str = ""
    keyword_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    found_by: list[str] = field(default_factory=list)   # e.g. ["meaning", "keyword"]

    @property
    def score(self) -> float:
        """The single 'best available' score, used for display."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.similarity is not None:
            return self.similarity
        return self.keyword_score or 0.0

    @property
    def score_name(self) -> str:
        if self.rerank_score is not None:
            return "rerank"
        if self.similarity is not None:
            return "similarity"
        return "keyword"


_client = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False),   # keep everything private
        )
    return _client


def get_collection():
    """A "collection" is like a table. We create it the first time."""
    return _get_client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # measure closeness with cosine similarity
        embedding_function=None,             # WE make the embeddings (via Ollama), not Chroma
    )


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Save chunks + their vectors. `upsert` = insert new, or update if the ID exists."""
    collection = get_collection()
    batch = 500   # Chroma prefers moderately sized batches
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.upsert(
            ids=[c.id for c in part],
            embeddings=embeddings[start:start + batch],
            documents=[c.text for c in part],
            metadatas=[c.metadata for c in part],
        )


def delete_file(path: str) -> None:
    """Remove every chunk that came from one file (used before re-ingesting it)."""
    get_collection().delete(where={"path": path})


def search(query_embedding: list[float], top_k: int = config.TOP_K) -> list[SearchResult]:
    """Find the top_k chunks whose meaning is closest to the question."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns lists-of-lists (one list per question). We asked one question -> [0].
    return [
        SearchResult(text=doc, metadata=meta, similarity=round(1 - dist, 4), id=chunk_id,
                     found_by=["meaning"])
        for chunk_id, doc, meta, dist in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        )
    ]


def get_all() -> tuple[list[str], list[str], list[dict]]:
    """Every stored chunk: (ids, texts, metadatas). Used to build the keyword index."""
    raw = get_collection().get(include=["documents", "metadatas"])
    return raw["ids"], raw["documents"], raw["metadatas"]


def count() -> int:
    return get_collection().count()


def list_sources() -> dict[str, int]:
    """Which files are in the knowledge base, and how many chunks each has."""
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    return dict(Counter(m["source"] for m in metadatas))


def reset() -> None:
    """Delete the whole collection (start fresh)."""
    client = _get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass   # it did not exist yet - nothing to delete
