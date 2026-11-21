"""
Playground 01 - See chunking with your own eyes.
===============================================

No AI needed for this one. Run it:

    python playground/01_chunking_demo.py

Then change CHUNK_SIZE and OVERLAP below and run it again.
Watch how the pieces change. That is ALL chunking is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "import rag" works

from rag.chunker import split_text

# ---- try changing these two numbers! ----
CHUNK_SIZE = 300
OVERLAP = 120     # overlap repeats WHOLE sentences, so it must be >= one sentence long
                  # try OVERLAP = 40 and watch the overlap disappear!
# -----------------------------------------

SAMPLE = """Pre-authorization (preauth) is a request sent by a hospital to an insurance company before treatment. It asks the insurer to confirm that the treatment will be covered.

A preauth request contains the patient details, the policy number, the diagnosis, and the estimated cost. The insurer reviews it and replies with an approval, a partial approval, or a rejection.

Sometimes the hospital needs to cancel a preauth. For example, the patient may decide not to go ahead with the treatment. In that case, the hospital sends a cancellation request referencing the original preauth ID. Once cancelled, the preauth cannot be reused and a fresh request must be created.

After treatment, the hospital submits the final claim. The claim refers back to the approved preauth so the insurer can match them."""

chunks = split_text(SAMPLE, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

print(f"Original text: {len(SAMPLE)} characters")
print(f"CHUNK_SIZE={CHUNK_SIZE}  OVERLAP={OVERLAP}  ->  {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"========== CHUNK {i}  ({len(chunk)} chars) ==========")
    print(chunk)
    print()

print("TIP: Look at the END of one chunk and the START of the next.")
print("     The repeated words you see there are the OVERLAP.")
