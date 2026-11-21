# 6️⃣ Retrieval — finding the right pages

## What happens when you ask a question

```
 You: "How do I cancel a preauth?"
          │
  [5] EMBED the question       → its location on the map of meanings
          │
  [6] SEARCH the vector DB     → the 5 closest chunks (TOP_K = 5)
          │
      FILTER weak matches      → drop anything below MIN_SIMILARITY (0.35)
          │
      ✅ the pages handed to the AI
```

This is the **R** in RAG. It is *the most important step*.

> 🎯 **If retrieval brings the wrong pages, even the smartest AI gives a wrong answer.**
> It's like giving a brilliant student the wrong chapter in an open-book exam.

## The two retrieval knobs

### `TOP_K`: how many chunks to fetch

| TOP_K | Effect |
|-------|--------|
| 1–2 | Very focused, but might **miss** part of the answer |
| **4–6** | ✅ Usually the sweet spot |
| 10+ | Catches more, but adds **noise** and fills the AI's desk; small models get confused |

You can change it live in the chat: `/k 8`

### `MIN_SIMILARITY`: the quality bar

Every chunk comes with a similarity score (0 to 1).
If the **best** match for a question is only 0.20, the documents probably **don't contain** the answer.

So we throw away weak matches. And if **nothing** passes the bar,
we **don't even ask the AI**. We just say *"I couldn't find this in your documents."*

That's an important anti-hallucination trick:
**no relevant pages → no chance to make things up.**

> 💡 Similarity scores are **relative**, not absolute. Different embedding models give different ranges.
> With `nomic-embed-text`, strong matches are often **0.6–0.8** and unrelated text **below 0.35**.
> Watch the scores with `/debug` and adjust `MIN_SIMILARITY` to your documents.

## Seeing retrieval with your own eyes: `/debug`

In the chat, type `/debug`. Now before every answer you'll see a table:

```
 #  Similarity  Source                   Text preview
 1     0.742    NHCX_spec.pdf, page 12   To cancel a pre-authorization the provider...
 2     0.688    NHCX_spec.pdf, page 13   The cancel request must reference the...
 3     0.512    NHCX_spec.pdf, page 40   Claim settlement occurs after...
```

This is **exactly** what the AI was allowed to read. Nothing more.

## When retrieval fails (and what you'll learn later)

| Problem | Example | Fix (Phase 2) |
|---------|---------|---------------|
| Exact codes/IDs | "Find *CLM-2024-00123*" | Meaning search is weak at exact codes → add **keyword search** too ("hybrid search") |
| Vague follow-ups | "explain *that* simpler" | Search has no idea what "that" is → **rewrite the question** using the chat history |
| Right page ranked 7th | Answer is in a chunk just outside TOP_K | Fetch 20, then **re-rank** with a more careful model and keep the best 5 |
| Answer spread over many pages | "Compare v1 and v2 of the API" | Smarter strategies: per-document search, summaries |

## Read the code

[`rag/retriever.py`](../../rag/retriever.py) is tiny on purpose: just **3 lines** of real logic.
Embed → search → filter. Everything else is in the modules you've already seen.

## 🧪 Try it yourself (after ingesting your files)

```powershell
.\.venv\Scripts\python.exe chat.py --debug
```

1. Ask a question you **know** is answered in your PDF. Is the right page ranked #1?
2. Ask the same thing using **completely different words**. Still found? (That's embeddings at work.)
3. Ask something that's **not** in your documents. What are the scores? Does it refuse to answer?
4. Search for an **exact code or ID** from your document. Does meaning search find it?
   (Often not, and now you know why Phase 2 adds keyword search.)
