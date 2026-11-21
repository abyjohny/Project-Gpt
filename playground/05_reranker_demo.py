"""
Playground 05 - Meet the re-ranker (the careful judge).
======================================================

Needs Ollama running + PyTorch/transformers installed. Run it:

    python playground/05_reranker_demo.py

The first run downloads the re-ranker model (~2.2 GB) - be patient once.

We compare two rankings of the SAME candidates:
  1. meaning search  (question and chunk turned into numbers SEPARATELY)
  2. re-ranker       (question and chunk read TOGETHER)
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "import rag" works

from rag import embedder, reranker

if not reranker.is_available():
    sys.exit("PyTorch/transformers are not installed. See docs/learning/11_reranking.md")

EXAMPLES = {
    "When can a preauth NOT be cancelled?": [
        "A preauth can be cancelled at any time by sending a cancellation request.",
        "Once the final claim has been submitted, the preauth is locked and cancellation is blocked.",
        "A preauth request contains patient details, policy number and estimated cost.",
        "Cancelled preauths cannot be reopened; create a fresh request instead.",
    ],
    "Who issues the ABHA number?": [
        "ABHA numbers are used to link a patient's health records across hospitals.",
        "The ABHA number is issued by the National Health Authority under the Ayushman Bharat Digital Mission.",
        "Patients can download their ABHA card from the ABHA mobile app.",
        "Health ID numbers are 14 digits long.",
    ],
}

print("Loading the re-ranker (first time: downloading ~2.2 GB)...")
started = time.perf_counter()
device = reranker.warm_up()
print(f"Ready on {device.upper()} in {time.perf_counter() - started:.1f}s\n")

for question, candidates in EXAMPLES.items():
    print("=" * 100)
    print(f"Q: {question}\n")

    # 1. meaning search
    q = np.array(embedder.embed_query(question))
    d = np.array(embedder.embed_documents(candidates))
    sims = (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q))

    # 2. re-ranker
    started = time.perf_counter()
    judge = reranker.score_pairs(question, candidates)
    ms = (time.perf_counter() - started) * 1000

    print("  MEANING SEARCH order:                       RE-RANKER order:")
    by_meaning = np.argsort(-sims)
    by_judge = np.argsort(-np.array(judge))
    for rank in range(len(candidates)):
        a, b = by_meaning[rank], by_judge[rank]
        print(f"  {rank + 1}. {sims[a]:.3f}  {candidates[a][:30]:<30}...   "
              f"{rank + 1}. {judge[b]:.3f}  {candidates[b][:45]}...")
    print(f"\n  (re-ranker judged {len(candidates)} pairs in {ms:.0f} ms)\n")

print("Notice: the re-ranker's scores are far more DECISIVE (close to 0 or close to 1),")
print("and it understands words like NOT and WHO that meaning search mostly ignores.")
print("\nTRY THIS: add your own question + 4 candidate sentences to EXAMPLES.")
