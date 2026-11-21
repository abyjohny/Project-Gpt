# 1️⃣1️⃣ Re-ranking: the careful judge

> **Phase 2 · Lesson 3** · Code: [`rag/reranker.py`](../../rag/reranker.py) · Demo: `playground/05_reranker_demo.py`

## Fast search is a bit "blurry"

Remember how meaning search works:
1. The question becomes a list of numbers.
2. Every chunk became a list of numbers (earlier, during ingest).
3. We compare the numbers.

The question and the chunk **never actually meet**. Each one is summarised **separately** into numbers, and we compare the summaries.
That's why it's so fast (milliseconds for 100,000 chunks), but some detail gets lost.

**Example where it goes wrong:**

> Q: "When can a preauth **NOT** be cancelled?"

| Chunk | Meaning search thinks |
|---|---|
| A: "A preauth can be cancelled at any time before…" | 😍 0.82: same words, same topic! |
| B: "Once the final claim is submitted, the cancellation is blocked." | 🙂 0.71 |

Chunk **B** is the real answer. The word "NOT" barely changes the meaning-numbers.

## The re-ranker reads them *together*

A **re-ranker** (technically a *cross-encoder*) is a different kind of model.
You give it the question **and** one chunk **at the same time**, and it reads them side by side, word by word, like a teacher marking an answer:

```
 [ When can a preauth NOT be cancelled? ] + [ Once the final claim is submitted, ... ]
                                │
                          re-ranker model
                                │
                          score: 0.94  ✅
```

Because it sees both texts together, it understands *how* they relate: negations ("not"), who did what, and whether the chunk actually **answers** the question or just mentions the same words.

## Why not use the re-ranker for everything?

It's **slow**: it has to read *every* pair from scratch.

| | Meaning search | Re-ranker |
|---|---|---|
| Reads question + chunk | separately (pre-computed) | together, every time |
| 100,000 chunks | ~20 ms | ~hours 😅 |
| 30 chunks | ~20 ms | ~0.2–1 second ✅ |
| Accuracy | good | excellent |

So we use **both**, like a hiring process:

```
 all chunks ──[fast search: CV screening]──► ~30 shortlisted ──[re-ranker: interviews]──► best 5
```

This "retrieve, then re-rank" pattern is used by nearly every serious RAG system.

## The model we use

**`BAAI/bge-reranker-v2-m3`**, a free, open model:
- Downloaded **once** from Hugging Face (~2.2 GB), then runs **fully offline**
- Runs on your **GPU** in 16-bit mode (~1.1 GB of VRAM), next to Qwen
- Understands 100+ languages

It needs **PyTorch** (the engine that runs most AI models) and **transformers**
(a library that knows how to load models from Hugging Face).

### Scores and the safety filter

The re-ranker gives a score from **0 to 1**:

| Score | Meaning |
|---|---|
| 0.8 – 1.0 | Directly answers the question |
| 0.1 – 0.8 | Related, partly useful |
| below 0.02 | Not relevant, thrown away (`MIN_RERANK_SCORE`) |

This makes "I could not find this in your documents" more reliable: junk chunks no longer reach Qwen.

### ...but don't keep only ONE chunk (`RERANK_MIN_KEEP`)

A real failure: *"what is search api used for?"*. The document never says "search API". It says
"Payer **Discovery** API" and "Policy **Discovery** API". The judge kept just **1 of 33** chunks (score 0.30).
Qwen saw one view of the topic and answered only half of it.

So there is a safety net. If **at least one** chunk passes the filter (the question IS about your
documents) but fewer than `RERANK_MIN_KEEP = 3` do, the next-best chunks are kept anyway, up to 3.

- If **nothing** passes, nothing is kept. "I could not find this" still works.
- The order never changes, so the #1 chunk stays #1. Measured: Hit@1 and MRR on the exam were exactly the same.

## Read the code

Open [`rag/reranker.py`](../../rag/reranker.py). The important part is `score_pairs()`:

```python
inputs = _tokenizer([[question, text] for text in batch], ...)   # glue question + chunk together
raw_scores = _model(**inputs).logits                             # model reads both, outputs a number
scores = torch.sigmoid(raw_scores)                               # squash into 0..1
```

That's the whole idea in 3 lines.

## Settings (in `rag/config.py`)

| Setting | Default | What it does |
|---|---|---|
| `USE_RERANKER` | `True` | Switch the judge on or off |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Lighter option: `BAAI/bge-reranker-base` (English and Chinese only) |
| `RERANKER_DEVICE` | `"auto"` | `"cpu"` if your GPU runs out of memory |
| `MIN_RERANK_SCORE` | `0.02` | Raise it to be stricter, lower it if good chunks get thrown away |
| `RERANK_MIN_KEEP` | `3` | If at least one chunk passes, keep at least this many (the runners-up), so Qwen sees more than one view |
| `CANDIDATES_PER_SEARCH` | `20` | Bigger shortlist = better chance, but slower judging |

## 🧪 Try it yourself

```powershell
python playground\05_reranker_demo.py
```
Compare the meaning-search order with the re-ranker order. Look at the "NOT" example.

Then in the chat:
```
python chat.py --debug
```
1. Ask a question. Compare the **Meaning** column with the **Rerank** column. Did the order change?
2. Type `/rerank` to switch it **off**, and ask the same question. Different top chunk?
3. Ask something totally unrelated ("best pizza in town"). With the re-ranker on, are all chunks filtered out?

> 🎯 **Key lesson:** fast search finds *candidates*; a careful judge picks the *winners*.
> Speed and accuracy come from using **two stages**.
