# 1️⃣3️⃣ Measuring quality: giving your AI an exam

> **Phase 2 · Lesson 5** · Code: [`evaluate.py`](../../evaluate.py)

## "It feels better" is not good enough

You change a setting, ask 2 questions, and the answers look better. 🎉
But did 5 *other* questions quietly get worse? You'd never know.

Professionals never tune by feel. They build a **test set** (an exam) and measure every change.
That's what `evaluate.py` does.

## What exactly do we measure?

We test **retrieval**, the "find the right page" step, because:
1. If retrieval is wrong, the answer is wrong, no matter how smart Qwen is.
2. It's fast to test: no waiting for Qwen to write answers.
3. It's objective: the right page either came back or it didn't.

### The exam

A list of questions where we **know** which page holds the answer:

```json
{ "question": "What is the waiting period for maternity cover?", "source": "policy.pdf", "page": 14 }
```

### The scores

For every question we check: **did the right page come back, and at what position?**

| Score | Plain meaning | Example |
|---|---|---|
| **Hit@1** | % of questions where the right page came **1st** | 70% = right page on top 7 times out of 10 |
| **Hit@5** | % where the right page was **anywhere in the top 5** | 90% = Qwen *saw* the answer 9 times out of 10 |
| **MRR** | Average of **1 ÷ position** (1st → 1.0, 2nd → 0.5, 3rd → 0.33, missed → 0) | 0.80 = usually 1st, sometimes 2nd |

Higher is better for all three. **Hit@5** is the most important for RAG: if the right page is in the top 5, Qwen can use it.

## Where do the exam questions come from?

### Option 1: Let Qwen write them (fast start)
```powershell
python evaluate.py generate --n 25
```
It picks 25 random chunks from your knowledge base and asks Qwen:
*"Write a question that this passage answers."*
We know which chunk (and page) each question came from, so we know the right answer automatically.

> ⚠️ **Watch out:** AI-written questions often **copy words** from the passage,
> which makes keyword search look better than it really is.
> We ask Qwen to use its own words, but it doesn't always manage. Open `data\eval\questions.json`
> and fix or delete weird questions.

### Option 2: Write your own (the gold standard)
Add your real questions to `data\eval\questions.json`:
```json
{ "question": "Can I cancel after the claim is submitted?", "source": "claims_guide.pdf", "page": 8 }
```
Your own questions are **kept** when you generate new AI questions.
Even 10 good hand-written questions are worth more than 100 generated ones.

## Running the exam

```powershell
python evaluate.py run
```

Output looks like:

```
             Retrieval exam - 25 questions, top-5
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Setup           ┃ Hit@1 ┃ Hit@5 ┃   MRR ┃ Avg time ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ vector          │   60% │   84% │ 0.693 │    25 ms │
│ keyword         │   56% │   80% │ 0.652 │     2 ms │
│ hybrid          │   68% │   92% │ 0.772 │    28 ms │
│ hybrid + rerank │   80% │   96% │ 0.861 │   410 ms │
└─────────────────┴───────┴───────┴───────┴──────────┘
```
*(example numbers; yours will differ)*

Add `--misses` to see **which questions failed**. Those are the most educational part:
read each one and ask *why* the search missed it. Wrong words? Answer split across two chunks? Scanned page?

## The improvement loop 🔁

```
  1. run the exam        → write down the scores
  2. change ONE setting  → e.g. CHUNK_SIZE 1000 → 600
  3. re-ingest if needed → python ingest.py --reset   (for chunking/embedding changes)
  4. run the exam again  → better or worse?
  5. keep it or undo it
```

Change **one thing at a time**, otherwise you won't know what helped.

## 🧪 Try it yourself

```powershell
python evaluate.py generate --n 25
python evaluate.py run --misses
```
1. Which setup wins on *your* documents? Is the re-ranker worth its extra time?
2. Write 5 questions of your own into `data\eval\questions.json`, and make one of them tricky
   (different words from the document). Run again.
3. Experiment: set `CHUNK_SIZE = 600` in `config.py`, run `python ingest.py --reset`, then `python evaluate.py run`.
   Did the scores go up or down?

> 🎯 **Key lesson:** what you don't measure, you can't improve. A small exam turns guessing into engineering.
