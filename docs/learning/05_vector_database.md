# 5️⃣ The vector database — the library's index

## What it is

After embedding, every chunk has a "location" (768 numbers).
Now we need somewhere to **keep** them and quickly answer one question:

> *"Here's a location. Which stored chunks are **closest** to it?"*

That's exactly what a **vector database** does.

| Normal database (like PostgreSQL) | Vector database (like ChromaDB) |
|-----------------------------------|---------------------------------|
| `WHERE claim_id = 'CLM-123'` | "closest in meaning to *how do I cancel a preauth?*" |
| Finds **exact** matches | Finds **similar** matches |
| Answers "yes/no, it matches" | Answers "here are the 5 closest, with scores" |

## What we store for every chunk

Think of each chunk as an **index card** in a library drawer:

```
┌──────────────────────────────────────────────────────┐
│ ID:        a3f9c21e7b...                (fingerprint) │
│ EMBEDDING: [0.021, -0.443, 0.118, ...]   (location)   │
│ TEXT:      "To cancel a preauth, the hospital..."     │
│ METADATA:  source=NHCX_spec.pdf, page=12, chunk=3     │
└──────────────────────────────────────────────────────┘
```

- **Embedding** is used to **find** the card.
- **Text** is what we **give the AI** to read.
- **Metadata** is what we **show you** as the source.

## Why ChromaDB?

- It runs **inside our Python program**. No server to install or start.
- It saves to a normal folder: `data/vector_db/`
- It's simple, which makes it perfect for learning. (Later we may move to **Qdrant** or **pgvector**
  for bigger projects; the idea stays exactly the same.)

We also switched off Chroma's anonymous usage reporting (`anonymized_telemetry=False`),
so **nothing** leaves your machine.

## How does it search so fast?

**Simple way:** compare the question with **every** card. Fine for 1,000 cards.
That's exactly what our tiny database in `playground/03_tiny_vector_db.py` does.

**Clever way (what Chroma does):** with millions of cards, checking each one is slow.
So Chroma builds a **shortcut network** called **HNSW**. Think of it like this:

> To find a house in a city, you don't knock on every door.
> You take the **highway** to the right area, then **main roads** to the right neighbourhood,
> then **side streets** to the right house.

HNSW builds "highways" and "side streets" between cards, so a search only visits a few hundred
cards instead of millions. It's *almost* perfectly accurate and **thousands of times faster**.

## Updating files without duplicates

When you change a document and run `ingest.py` again, we:
1. **Delete** all old cards from that file (`delete_file`)
2. **Add** the new cards (`upsert` = "insert, or update if the ID already exists")

So your library never has stale or duplicate pages.

## Read the code

[`rag/vector_store.py`](../../rag/vector_store.py):
- `get_collection()`: opens (or creates) our "drawer", set to measure closeness with **cosine**
- `add_chunks()`: saves cards
- `search()`: finds the closest cards and converts Chroma's *distance* into *similarity* (`1 − distance`)
- `list_sources()`: which files are in the library

## 🧪 Try it yourself

```powershell
.\.venv\Scripts\python.exe playground\03_tiny_vector_db.py
```

1. Read the `TinyVectorDB` class. It's only ~20 lines. **That's a vector database.**
2. Add a few of your own sentences to `db.add([...])` and ask questions about them.
3. Ask something unrelated (*"best pizza in town"*). How low is the best score?
