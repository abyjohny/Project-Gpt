# 1️⃣ What is RAG? — the open-book exam

## The problem

Imagine a very smart friend who has read millions of books, but **none of yours**.

- They don't know your company's NHCX integration document.
- They don't know your ClaimLens code.
- They stopped reading at some date, so they don't know anything newer.
- And when they don't know something, they sometimes **confidently make it up**.

That friend is a language model like Qwen.

## The obvious fix (that doesn't work)

*"Just paste the whole document into the chat!"*

A 200-page PDF is around 100,000+ words. The model's "desk" (its **context window**)
is much smaller, and even when it fits:
- it gets **slow**,
- it gets **confused** (important details get lost in the middle),
- and you'd have to paste it **every single time**.

## The smart fix: RAG

**RAG = Retrieval-Augmented Generation.** Let's break the name down:

| Word | Meaning | Analogy |
|------|---------|---------|
| **Retrieval** | Find the few paragraphs that matter | Open the textbook at the right page |
| **Augmented** | Add those paragraphs to the question | Put the page next to the exam paper |
| **Generation** | The AI writes the answer from them | Write the answer using the page |

It's an **open-book exam**. The student (Qwen) is smart but doesn't know your book.
So before each question, a librarian finds the **right 5 pages** and puts them on the desk.
The student reads only those 5 pages and answers.

## The two halves of RAG

### Half A — Preparing the library (done once per document)

```
  Your files          📄 NHCX_spec.pdf   📝 notes.docx   💻 code.py
       │
  [1] LOAD            take the words out of each file
       │
  [2] CHUNK           cut the words into small pieces (a few paragraphs each)
       │
  [3] EMBED           turn each piece into "meaning numbers"
       │
  [4] STORE           save the pieces + numbers in a vector database
```

This is like a librarian reading every page of every book and writing an index card
for each page saying *what it's about*.

👉 Code: [`ingest.py`](../../ingest.py)

### Half B — Answering a question (every time you ask)

```
  "How do I cancel a preauth?"
       │
  [5] EMBED QUESTION  turn the question into meaning numbers too
       │
  [6] SEARCH          find the pieces whose numbers are closest
       │
  [7] BUILD PROMPT    "Using ONLY these pieces: ... answer: How do I cancel a preauth?"
       │
  [8] GENERATE        Qwen writes the answer and says which piece it used [1][2]
```

👉 Code: [`chat.py`](../../chat.py)

## Why this is so powerful

| Without RAG | With RAG |
|-------------|----------|
| "I don't know your document" or makes things up | Answers from **your** document |
| Can't tell you where the info came from | Cites **file and page** |
| Must retrain the model to teach it new things (very expensive) | Just drop a new PDF in the folder and run `ingest.py` |
| Your documents may go to a cloud company | Everything stays **on your laptop** |

## The one sentence to remember

> **RAG doesn't make the AI smarter. It puts the right information in front of it.**

That's also why, when answers are bad, the cause is usually **wrong pages retrieved**
(steps 2–6), not a "dumb" model. We'll learn to debug that in [note 8](08_run_and_experiment.md).

## Our project map

```
Project GPT/
├── data/documents/      ← 📥 put your PDFs, DOCX, TXT here
├── data/vector_db/      ← 🗄️ the "library index" (created automatically)
├── rag/
│   ├── config.py        ← ⚙️ all settings in one place
│   ├── loader.py        ← step 1
│   ├── chunker.py       ← step 2
│   ├── embedder.py      ← step 3 (+ step 5)
│   ├── vector_store.py  ← step 4 (+ step 6)
│   ├── retriever.py     ← steps 5–6 together
│   ├── generator.py     ← steps 7–8
│   └── ollama_client.py ← talks to Ollama
├── ingest.py            ← ▶️ run Half A
├── chat.py              ← ▶️ run Half B
└── playground/          ← 🧪 small experiments
```

## 🧪 Try it yourself

Open [`rag/config.py`](../../rag/config.py) and read every setting.
You don't need to understand them all yet. Just notice that **every step has its own settings**.
By the end of note 8 you'll know what each one does.
