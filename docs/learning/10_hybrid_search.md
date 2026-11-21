# 🔟 Hybrid search: two detectives are better than one

> **Phase 2 · Lesson 2** · Code: [`rag/retriever.py`](../../rag/retriever.py) → `reciprocal_rank_fusion()`

## The idea

You now have two detectives:

| 🕵️ Meaning detective (embeddings) | 🔎 Keyword detective (BM25) |
|---|---|
| Understands paraphrases: "withdraw" ≈ "cancel" | Spots exact codes, names, numbers: "E-1047" |
| Gets confused by codes and rare terms | Clueless about synonyms |

**Hybrid search** sends both detectives out on the same question, and each comes back with a ranked list of suspects (chunks).
Then we merge the two lists into one.

```
  question
     │
     ├──► meaning search ──► [ C7, C2, C9, C4, ... ]   (20 candidates)
     │
     └──► keyword search ──► [ C2, C5, C7, C1, ... ]   (20 candidates)
                                     │
                              MERGE (RRF)
                                     │
                         [ C2, C7, C5, C9, ... ]
```

## The tricky part: how do you merge two lists?

The two detectives use **different scoring systems**:
- meaning similarity: 0.0 to 1.0 (e.g. 0.71)
- BM25 keyword score: 0 to... anything (e.g. 14.8)

Adding 0.71 + 14.8 is like adding **kilometres and kilograms**. It makes no sense.

### The trick: ignore the scores, use the *positions*

"1st place" means the same thing in both lists. So we only use the **rank**.
This is called **Reciprocal Rank Fusion (RRF)**, a fancy name for a simple rule:

> Every chunk gets **1 ÷ (60 + its position)** points from **each** list it appears in.

| Position | Points |
|---|---|
| 1st | 1/61 = 0.0164 |
| 2nd | 1/62 = 0.0161 |
| 10th | 1/70 = 0.0143 |
| not in list | 0 |

**Example:**

| Chunk | Meaning rank | Keyword rank | Total points |
|---|---|---|---|
| C2 | 2nd → 0.0161 | 1st → 0.0164 | **0.0325** 🥇 |
| C7 | 1st → 0.0164 | 3rd → 0.0159 | **0.0323** 🥈 |
| C5 | — | 2nd → 0.0161 | 0.0161 |
| C9 | 3rd → 0.0159 | — | 0.0159 |

A chunk that **both** detectives liked rises to the top. That's exactly what we want.

> 🤔 **Why 60?** It comes from the research paper that introduced RRF. It stops the 1st place from
> completely dominating, so 1st and 2nd are close and agreement between lists matters more.
> It's `RRF_K` in `config.py`, and you rarely need to change it.

## What you'll see in the chat

With `--debug`, the **Found by** column tells you which detective found each chunk:

```
#  Found by          Meaning  Keyword  ...
1  keyword+meaning   0.712    9.41         ← both agreed: strong candidate
2  keyword           -        12.03        ← only exact words matched (e.g. a code)
3  meaning           0.655    -            ← only the meaning matched (a paraphrase)
```

## Settings (in `rag/config.py`)

| Setting | Default | What it does |
|---|---|---|
| `RETRIEVAL_MODE` | `"hybrid"` | `"vector"`, `"keyword"` or `"hybrid"` |
| `CANDIDATES_PER_SEARCH` | `20` | How many chunks each detective brings back |
| `RRF_K` | `60` | The "60" in the formula |

## 🧪 Try it yourself

```powershell
python chat.py --debug
```
1. Ask a question and look at the **Found by** column. How many chunks were found by *both*?
2. Ask the same question 3 times, switching `/mode vector`, `/mode keyword`, `/mode hybrid` in between.
   Does hybrid keep the best chunks from both?
3. Find a question where vector and keyword **disagree completely**. Those are the cases where hybrid helps most.

> 🎯 **Key lesson:** merging by *position* (not score) lets us combine search methods that score in different ways.
> Next, we add a judge that reads the merged list properly: [Re-ranking](11_reranking.md).
