# 3️⃣ Chunking — cutting the cake into slices

## Why cut documents into pieces?

Imagine asking a librarian *"How do I cancel a preauth?"* and they hand you
**the whole 200-page manual**. Technically correct… totally useless.

You want them to hand you **the one paragraph** that answers it.

To do that, the library must be organised by **paragraph-sized pieces**, not whole books.
Those pieces are called **chunks**.

Reasons we chunk:
1. **Precision**: small pieces = we find the exact spot.
2. **Fits the AI's desk**: we can only give the AI a limited amount of text (its context window).
3. **Better meaning-numbers**: a chunk about *one topic* gets a clear meaning.
   A chunk mixing 10 topics gets a blurry "average" meaning that matches nothing well.

## How big should a chunk be?

It's a balance, like slicing a cake:

| Chunk size | Good | Bad |
|-----------|------|-----|
| **Too small** (1 sentence) | Very precise | Loses context. *"It must be sent within 24 hours."* What is "it"? 🤷 |
| **Too big** (5 pages) | Lots of context | Blurry meaning, wastes the AI's desk space, worse search |
| **Just right** (2–4 paragraphs) | ✅ One idea with enough context | — |

Our setting: **`CHUNK_SIZE = 1000` characters** (about 2–3 paragraphs, ~250 tokens).

## The smart way to cut: "recursive" splitting

We don't cut blindly every 1000 letters. That would slice words and sentences in half.
Instead we try to cut at **natural borders**, best first:

```
 1st try:  blank lines      "\n\n"   → between paragraphs   (best)
 2nd try:  line breaks      "\n"     → between lines
 3rd try:  sentence ends    ". "     → between sentences
 4th try:  spaces           " "      → between words
 5th try:  anywhere                  → last resort           (worst)
```

If a paragraph is small enough → keep it.
If it's still too big → cut it with the *next* border on the list. And so on.
Then small neighbouring pieces are **glued back together** until each chunk is close to 1000 characters.

## Overlap — the safety net

What if the answer sits right on the border between two chunks?

```
chunk 1: "...The hospital must send a cancellation request"
chunk 2: "referencing the original preauth ID within 24 hours..."
```

Neither chunk has the full sentence! 😱

So each chunk **repeats the last bit of the previous one**:

```
 chunk 1: [ ======== text ======== | overlap ]
 chunk 2:                          [ overlap | ======== text ======== | overlap ]
 chunk 3:                                                             [ overlap | ==== ]
```

Our setting: **`CHUNK_OVERLAP = 150` characters**, about one or two sentences.

### We only repeat WHOLE sentences 🧱

Repeating *half* a sentence would be messy ("...the original preauth ID" — what about it?).
So the code works like building with Lego bricks:

1. First it breaks the text into **sentence-sized bricks**.
2. Then it stacks bricks into a chunk until the chunk is nearly full (~1000 characters).
3. When it starts the next chunk, it **copies the last brick or two** from the previous chunk
   (as many as fit into the 150-character overlap budget).

```
 chunk 1: [ S1 ][ S2 ][ S3 ][ S4 ]
 chunk 2:             [ S3 ][ S4 ][ S5 ][ S6 ][ S7 ]
                        ↑ repeated sentences = overlap
```

> ⚠️ If a single sentence is *longer* than the overlap budget, it can't be repeated,
> so that border has no overlap. That's why the overlap should be at least
> as long as a typical sentence in your documents.

## Chunk IDs — fingerprints

Every chunk gets an **ID** made from its file, page, position and text
(using a "hash", a function that turns any text into a short, unique-looking code like `a3f9c21e…`).

Why? If you run `ingest.py` twice, the same chunk gets the **same ID**,
so the database **updates** it instead of storing a duplicate.

## Read the code

Open [`rag/chunker.py`](../../rag/chunker.py) and find:
- `clean_text()`: tidies messy PDF spacing
- `_split_recursive()`: the "best border first" cutting (used twice: once for chunk-size pieces, once for sentence bricks)
- `split_text()`: the 3 passes (pieces → sentence bricks → chunks)
- `_merge_with_overlap()`: glues bricks and adds the overlap
- `chunk_documents()`: copies the metadata (file, page) onto every chunk

## 🧪 Try it yourself

```powershell
.\.venv\Scripts\python.exe playground\01_chunking_demo.py
```

1. Look at where each chunk starts and ends. Find the repeated sentence(s): that's the overlap.
2. Change `CHUNK_SIZE = 300` to `150`. Run again. What happens?
3. Change it to `2000`. How many chunks now?
4. Set `OVERLAP = 40`. The overlap disappears. Why? (Hint: are any sentences shorter than 40 characters?)
5. Set `OVERLAP = 0`. Can you find a thought that's now split across two chunks?

> 🎯 **Key lesson:** chunking quality decides search quality.
> Bad chunks → wrong pages found → bad answers, no matter how smart the AI is.
