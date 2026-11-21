# 14 · Structure-aware chunking — follow the document's own shape

> **Code:** [`rag/loader.py`](../../rag/loader.py) (Word/PDF → Markdown) · [`rag/chunker.py`](../../rag/chunker.py) (`split_structured`)
> **Settings:** `CHUNKING_STRATEGY`, `SECTION_CHUNK_SIZE`, `SECTION_MERGE_MAX`, `CHUNK_LOCATION`, `TABLE_COMPACT`, `TABLE_OVERLAP_ROWS`

## The problem with cutting by size

Phase 1 cut text every ~1000 characters at paragraph or sentence borders. Think of tearing pages out of a book at random:

- One chunk can end in the middle of section 4.1 and continue into 4.2, mixing two topics.
- A table row like `| 3 | QUERIED | ... |` ends up far away from its column names and its heading.
  On its own it means nothing.
- The old Word loader even moved **all tables to the end** of the document, away from their headings.

## The idea: first follow the structure, then cut by size

```
 Word / PDF ──► loader.py ──► Markdown with real headings and tables
                                 # Section 4: Workflow Codes
                                 ## 4.1 Provider-Side Codes
                                 | Code | Meaning |
                                 ...
               chunker.py ──► 1. split at headings      (one topic per chunk)
                              2. glue tiny siblings      (5.6.1 + 5.6.2 + 5.6.3)
                              3. cut big tables BETWEEN rows, repeat the column names
                              4. write the "address" on top of every chunk
```

A finished chunk looks like this:

```
Location: Section 4: Workflow Codes Reference > 4.1 Provider-Side Workflow Codes

| Code | Name | Meaning |
| 2 | QUERIED | Payer needs more information ... |
| 3 | REJECTED | ... |
```

- **Word files:** headings come from the paragraph style ("Heading 1", "Heading 2"...), and tables stay in place.
- **PDFs:** they have no real headings, so we *guess* them: a line whose font is 15% bigger than
  normal text (`PDF_HEADING_SIZE_RATIO`) counts as a heading. Tables come from PyMuPDF's `find_tables()`.
  A section that continues on the next page keeps its address.

## ⚠️ Lesson learned: the first version was WORSE

This is the most important part of this note. A "smarter" idea is not automatically better.
**Measure it.** We ran the exam (`evaluate.py`) on two sets of 40 questions:

- Exam A was written from the *old* chunks.
- Exam B was written from the *new* chunks, with the address removed so the questions can't cheat.

Using both stops the exam from favouring one layout. Results with hybrid + re-rank:

| Setup | Exam A Hit@1 | Exam B Hit@1 | Overall Hit@1 | MRR |
|---|---|---|---|---|
| Old loader + recursive 1000 (Phase 2) | 85% | 95% | 90.0% | 0.930 |
| Structure, first try (1000 chars, no overlap) | 78% | 88% | 82.5% | 0.882 |
| + only nearest heading as address | 78% | 88% | 82.5% | ~0.88 |
| + 2000 chars | 78% | 90% | 83.8% | 0.892 |
| + 1200 chars, 1 overlap row | 80% | 90% | 85.0% | 0.898 |
| **Structure 1500, 1 overlap row, full address, compact tables** | **90%** | **92%** | **91.3%** | **0.931** |

**Why was the first try worse?** We compared the questions one by one. It came down to two things:

1. **No overlap in tables.** Plain text repeats ~150 characters between chunks, but table pieces didn't.
   A row near a border lost its neighbours. Fixed with `TABLE_OVERLAP_ROWS = 1`.
2. **Chunks too small.** Address line + column names + rows fill 1000 characters fast, which left too
   little real content per chunk. 1500 works best here: 1200 is too small and 2000 too big (one chunk then covers several topics again).

Also: compact tables (dropping the `| --- |` line) saves tokens and doesn't hurt quality.

## Speed

Re-ranking takes about 0.7–1.3 s for 20–35 candidates on the RTX 3050, about the same for every layout.
Earlier measurements of 3–30 s were **other programs using the GPU at the same time**, not the chunks.
Always measure speed on a quiet machine, and check more than once.

## 🧪 Try it yourself

1. Set `CHUNKING_STRATEGY = "recursive"` in `rag/config.py`, then run `python ingest.py --reset` and `python evaluate.py run`.
   Switch back to `"structure"` and compare.
2. Try `CHUNK_LOCATION = "none"`. Does keyword search get better or worse? Why?
3. Look at a few chunks: in `chat.py`, ask a question and read the `Location:` line of each source.
