"""
Playground 02 - Feel what an embedding is.
=========================================

Needs Ollama running with `nomic-embed-text` pulled. Run:

    python playground/02_embeddings_demo.py

We embed a few sentences and compare EVERY pair.
You will see that sentences with the same MEANING score high,
even when they share almost no words.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config, ollama_client

SENTENCES = [
    "How do I cancel a pre-authorization?",              # 0
    "Steps to revoke a preauth request",                 # 1  same meaning as 0, different words
    "The claim was settled and payment released",        # 2  related topic (insurance), different meaning
    "My cat loves sleeping in the sun",                  # 3  totally unrelated
    "Procedure for withdrawing an approval request",     # 4  similar meaning to 0/1
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """The formula behind "semantic search", written out by hand.

    dot product  = multiply matching numbers and add them all up
    length       = how long each arrow is
    similarity   = dot / (length_a * length_b)   -> between -1 and 1
    """
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)


vectors = ollama_client.embed([config.EMBED_DOCUMENT_PREFIX + s for s in SENTENCES])

print(f"Each sentence became a list of {len(vectors[0])} numbers.")
print(f"First 8 numbers of sentence 0: {[round(x, 3) for x in vectors[0][:8]]} ...\n")

print("Similarity of every sentence to sentence 0:")
print(f'   "{SENTENCES[0]}"\n')
scores = sorted(
    ((cosine_similarity(vectors[0], vectors[i]), i) for i in range(1, len(SENTENCES))),
    reverse=True,
)
for score, i in scores:
    bar = "#" * int(score * 40)
    print(f"  {score:.3f}  {bar:<40}  {SENTENCES[i]}")

print("\nNotice: sentence 1 shares almost NO words with sentence 0, but scores highest.")
print("That is the power of embeddings: they compare MEANING, not spelling.")
print("\nTRY THIS: add your own sentences to SENTENCES and guess the scores before running.")
