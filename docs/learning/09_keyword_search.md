# 9️⃣ Keyword search (BM25): the "Ctrl+F" that got smart

> **Phase 2 · Lesson 1** · Code: [`rag/keyword_search.py`](../../rag/keyword_search.py) · Demo: `playground/04_keyword_vs_meaning.py`

## The problem we're fixing

In Phase 1, every search was a **meaning search** (embeddings). It's brilliant at this:

| You ask | It finds | Why it works |
|---|---|---|
| "How do I **withdraw** a preauth?" | "…to **cancel** a pre-authorization…" | Different words, same meaning ✅ |

But try asking about an **exact thing**:

| You ask | What can go wrong |
|---|---|
| "What does error **E-1047** mean?" | It might return E-1046 or E-1042. To an embedding model these look *almost identical*: "some error code" |
| "Show me policy **POL-77812**" | Numbers have almost no "meaning", so the meaning-numbers come out blurry |
| "What is **ABHA**?" | Rare abbreviations the model never learned well |

Embeddings capture the *gist*. Sometimes you need the *exact word*.

## The old-school hero: keyword search

Think of the **index at the back of a textbook**:

```
appendicitis ........ 12, 48
ABHA ................ 3, 7, 22
E-1047 .............. 31
```

You don't read the whole book. You look up the word and jump to the page.
That's exactly what keyword search does. It's called an **inverted index**
(normal: page → words; inverted: word → pages).

## But which page is BEST? Enter BM25

If your question has 5 words and 40 chunks contain at least one of them, which one wins?
**BM25** (a scoring recipe from the 1990s that still powers Elasticsearch) gives points using 3 common-sense rules:

### Rule 1: Rare words are worth more (called *IDF*)
- "**the**" appears in every chunk → worthless, tells you nothing
- "**E-1047**" appears in one chunk → *gold*, that chunk is almost certainly the one

Like in a crowd: "a person with two eyes" doesn't help you find anyone, but "a person with a green hat" does.

### Rule 2: Repeated words count more, up to a point (called *TF*)
A chunk that says "preauth" 4 times is probably more about preauth than one that says it once.
But 20 times isn't 20× better, so the points **level off** (diminishing returns).

### Rule 3: Short chunks get a small bonus
Finding "appendicitis" in a 3-line chunk is a stronger signal than finding it once in a 2-page chunk.

```
score(chunk) = sum over question words of:   rarity  x  repetition (levelled off)  x  shortness bonus
```

That's it. **No AI, no GPU, just counting.** It takes milliseconds.

## How we cut text into "words" (tokenizing)

Before counting, every text is turned into a list of lowercase words:

```
"Send the Cancel-Request for E-1042"
   → ['send', 'cancel-request', 'cancel', 'request', 'e-1042', 'e', '1042']
```

Notice:
- **"the", "for"** are dropped. These are *stopwords*, too common to help.
- **"Cancel-Request"** is kept whole *and* split, so both "cancel request" and "cancel-request" match.
- **"E-1042"** stays together, so codes match exactly.

## Its weaknesses (why we don't use it alone)

| Weakness | Example |
|---|---|
| No idea about meaning | "withdraw" ≠ "cancel" → zero points 😬 |
| Word forms | "cancelled" ≠ "cancel" (we don't do *stemming*, to keep it simple) |
| Spelling mistakes | "preuath" matches nothing |

So: **keyword search is strong where meaning search is weak, and the other way round.**
The obvious idea is to use both. That's the next lesson: [Hybrid search](10_hybrid_search.md).

## Read the code

Open [`rag/keyword_search.py`](../../rag/keyword_search.py):
- `tokenize()`: text → words
- `BM25Index.__init__()`: builds the "book index" (`postings`) and the rarity scores (`idf`)
- `BM25Index.search()`: the 3 rules in about 10 lines
- `get_index()`: builds the index from everything in ChromaDB, and rebuilds it when the chunk count changes

> 💡 The keyword index lives in memory and is rebuilt from ChromaDB when the chat starts.
> If you re-ingest an *edited* file while chat is open, type `/reload`.

## 🧪 Try it yourself

```powershell
python playground\04_keyword_vs_meaning.py
```
Watch which method wins for the error-code question, and which wins for the paraphrased question.

Then in the chat, using your own PDF:
```
python chat.py --debug
/mode keyword
```
1. Ask about an **exact term** from your PDF (a name, number or code). Look at the `Keyword` column.
2. Ask the same thing using **different words**. Keyword search struggles. Now try `/mode vector`.
3. Switch back with `/mode hybrid`.

> 🎯 **Key lesson:** meaning search understands *what you mean*; keyword search notices *exactly what you typed*.
> Real search systems need both.
