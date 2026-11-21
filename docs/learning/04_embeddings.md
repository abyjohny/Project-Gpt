# 4️⃣ Embeddings — turning meaning into numbers

This is the most important idea in RAG. Take your time with it.

## The problem with normal search

Normal search (Ctrl+F) looks for **the same letters**.

You search: `cancel preauth`
The document says: *"To **revoke** a **pre-authorization**, send…"*

Ctrl+F finds **nothing**. Same meaning, different words. 😞

We need a search that understands **meaning**, not spelling.

## The big idea: a map of meanings

Imagine a giant map where **every sentence has a location**,
and sentences with **similar meaning are placed close together**:

```
                     🗺️  THE MAP OF MEANINGS

     "cancel preauth" •  • "revoke pre-authorization"
                        • "withdraw approval request"

                                              • "claim settled"
                                              • "payment released"

  • "my cat sleeps in the sun"
                                    • "PostgreSQL database"
                                    • "Redis cache"
```

A location on a normal map is 2 numbers: **(latitude, longitude)**.

An **embedding** is the location of a sentence on the map of meanings.
But meaning is far richer than 2 directions, so this map has **768 directions**.
So every chunk becomes a list of **768 numbers**:

```
"revoke a pre-authorization"  →  [0.021, -0.443, 0.118, 0.902, ... 768 numbers]
```

You can't picture 768 directions (nobody can!). Just think of the 2-D map; the idea is the same.

> 💡 **Vector** is just the maths word for "a list of numbers that points somewhere".
> **Embedding** = the vector that represents a piece of text. People use the two words interchangeably.

## Who decides the numbers?

The **embedding model** (`nomic-embed-text`). It was trained on millions of
sentence pairs to learn: *"these two mean the same → put them close; these don't → put them far apart"*.

It's a **separate, small model**. It doesn't chat or write answers. It only places texts on the map.
Think of it as the **librarian** who reads every page and files it in the right spot.

## Measuring "closeness": cosine similarity

To compare two locations we check **which direction they point**.
If two arrows point the same way → same meaning.

```
   same direction       somewhat apart        opposite / unrelated
        ↗ ↗                  ↗  →                    ↗   ↙
   similarity ≈ 1.0     similarity ≈ 0.5        similarity ≈ 0
```

This score is called **cosine similarity**: 1.0 = identical meaning, around 0 = unrelated.

The formula (you'll see it written out in `playground/02_embeddings_demo.py`):
```
similarity = (multiply matching numbers and add them up)
             ÷ (length of arrow A × length of arrow B)
```

## ⚠️ The golden rule

> **Documents and questions MUST use the SAME embedding model.**

Each model draws its **own map**. Comparing a location from one map with a location
on another map is meaningless, like comparing kilometres with miles.
If you ever change `EMBED_MODEL`, you must **re-ingest everything** (`python ingest.py --reset`).

## The little "prefixes"

You'll see this in [`rag/config.py`](../../rag/config.py):
```python
EMBED_DOCUMENT_PREFIX = "search_document: "
EMBED_QUERY_PREFIX    = "search_query: "
```
`nomic-embed-text` was trained to work best when told **what kind of text** it's looking at:
a stored document, or a search question. It's a small trick that improves results.

## Read the code

[`rag/embedder.py`](../../rag/embedder.py):
- `embed_documents()`: embeds chunks **in batches of 32** (much faster than one at a time)
- `embed_query()`: embeds your question

## 🧪 Try it yourself

```powershell
.\.venv\Scripts\python.exe playground\02_embeddings_demo.py
```

1. Notice that *"Steps to revoke a preauth request"* scores high against
   *"How do I cancel a pre-authorization?"*, with almost no shared words!
2. Notice the cat sentence scores lowest.
3. Add your own sentences: something from ClaimLens, something from NHCX.
   **Guess the score before running.** How good is your intuition?
