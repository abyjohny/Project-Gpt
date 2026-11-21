"""
Playground 04 - Keyword search vs meaning search, side by side.
==============================================================

Needs Ollama running (for the meaning search). Run it:

    python playground/04_keyword_vs_meaning.py

Watch which method wins for each question:
  - exact codes / names     -> keyword search shines
  - same meaning, new words -> meaning search shines
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "import rag" works

from rag import embedder
from rag.keyword_search import BM25Index, tokenize

# A tiny "library" of 8 chunks. Notice how similar the error codes look.
LIBRARY = [
    "E-1046 Signature verification failed: the digital signature on the payload could not be verified.",
    "E-1047 Consent artefact expired: ask the patient for a fresh consent through the ABHA app.",
    "E-1048 Unsupported document format: only PDF, JPEG and PNG attachments are accepted.",
    "E-1042 Duplicate submission: a request with the same correlation ID was already received.",
    "To cancel a pre-authorization, the hospital sends a CancelRequest with the original preauth ID.",
    "A claim must be submitted within 30 days of the patient's discharge from hospital.",
    "ABHA is a 14-digit health ID number issued under the Ayushman Bharat Digital Mission.",
    "Redis is an in-memory key-value store that is often used for caching.",
]

QUESTIONS = [
    "what does E-1047 mean?",              # exact code      -> keyword should win
    "how do I withdraw a preauth?",        # paraphrase      -> meaning should win ("withdraw" != "cancel")
    "deadline for sending a claim",        # paraphrase      -> meaning
    "ABHA",                                # rare term       -> both OK
]

print("How the keyword search 'sees' a sentence:")
print("  ", tokenize("E-1047 Consent artefact expired: ask the patient..."), "\n")

# ---- build both search engines over the same library
ids = [f"chunk{i}" for i in range(len(LIBRARY))]
bm25 = BM25Index(ids, LIBRARY, [{} for _ in LIBRARY])
doc_vectors = np.array(embedder.embed_documents(LIBRARY))
doc_vectors /= np.linalg.norm(doc_vectors, axis=1, keepdims=True)


def meaning_top3(question):
    q = np.array(embedder.embed_query(question))
    q /= np.linalg.norm(q)
    sims = doc_vectors @ q
    best = np.argsort(-sims)[:3]
    return [(float(sims[i]), LIBRARY[i]) for i in best]


for question in QUESTIONS:
    print("=" * 100)
    print(f"Q: {question}")
    print("  MEANING search (similarity 0..1):")
    for score, text in meaning_top3(question):
        print(f"     {score:.3f}  {text[:75]}")
    print("  KEYWORD search (BM25 points):")
    hits = bm25.search(question, top_k=3)
    if not hits:
        print("     (no chunk contains any of the question's words!)")
    for r in hits:
        print(f"     {r.keyword_score:6.2f}  {r.text[:75]}")

print("\nTRY THIS: add your own sentences to LIBRARY and your own QUESTIONS.")
print("          Can you find a question where only ONE of the two methods gets it right?")
