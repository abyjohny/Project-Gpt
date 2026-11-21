"""
Playground 03 - Build a tiny vector database in 30 lines.
========================================================

ChromaDB can feel like a black box. It isn't. At its heart, a vector DB does this:

    1. keep a list of (text, vector) pairs
    2. when a question comes in, turn it into a vector
    3. compare it with every stored vector
    4. return the closest ones

Run:   python playground/03_tiny_vector_db.py

(Real vector DBs add clever shortcuts so they don't have to compare with
EVERY vector when you have millions - but the idea is exactly this.)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config, ollama_client


class TinyVectorDB:
    def __init__(self):
        self.texts: list[str] = []
        self.vectors: list[np.ndarray] = []

    def add(self, texts: list[str]) -> None:
        embeddings = ollama_client.embed([config.EMBED_DOCUMENT_PREFIX + t for t in texts])
        for text, vec in zip(texts, embeddings):
            self.texts.append(text)
            v = np.array(vec)
            self.vectors.append(v / np.linalg.norm(v))   # normalise: make every arrow length 1

    def search(self, question: str, top_k: int = 3) -> list[tuple[float, str]]:
        q = np.array(ollama_client.embed([config.EMBED_QUERY_PREFIX + question])[0])
        q = q / np.linalg.norm(q)
        # With length-1 arrows, cosine similarity is just the dot product.
        scores = np.array(self.vectors) @ q
        best = np.argsort(scores)[::-1][:top_k]          # indexes of the highest scores
        return [(float(scores[i]), self.texts[i]) for i in best]


db = TinyVectorDB()
db.add([
    "ABHA is a 14-digit health ID number issued under the Ayushman Bharat Digital Mission.",
    "A preauth can be cancelled by sending a cancellation request with the original preauth ID.",
    "PostgreSQL is an open-source relational database.",
    "Claims are settled after the insurer verifies the final bill and discharge summary.",
    "Redis is an in-memory key-value store often used for caching.",
    "Consent artefacts let patients control who can see their health records.",
])

for question in [
    "how to withdraw a pre authorization?",
    "what is the health id number?",
    "which tool is good for caching?",
]:
    print(f"\nQ: {question}")
    for score, text in db.search(question):
        print(f"   {score:.3f}  {text}")

print("\nTRY THIS: ask a question that matches NOTHING (e.g. 'best pizza in town').")
print("          Look at how LOW the best score is. That's why we use MIN_SIMILARITY.")
