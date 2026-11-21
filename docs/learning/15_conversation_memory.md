# 15 · Conversation memory — making "and the second one?" work

> **Code:** [`rag/memory.py`](../../rag/memory.py) · used by [`chat.py`](../../chat.py) · tested by `python evaluate.py followups`

## The problem

```
You: What payer response codes are there?
AI:  1 APPROVED, 2 QUERIED, 3 REJECTED ...
You: and what about the second one?
```

You know that "the second one" means **QUERIED**. Our search step doesn't, because it has no memory.
It turns the words *"and what about the second one?"* into a meaning vector and looks for chunks
that are similar to... nothing in particular. The wrong chunks come back, so you get a wrong answer.

Giving the chat history to the AI is **not enough**. The AI only reads the chunks the search
found, and that search was run with the vague question.

## Two ways to fix it

**A. Glue on the previous question (free).** Search with *previous question + follow-up*:

```
"What diagnosis type codes exist? What does the last one mean?"
```

This is ugly to read, but the **re-ranker** reads the whole thing and works out what is meant.

**B. Query rewriting with Qwen (~1.5 s).** Ask Qwen one extra, very small question:

> *"Here is the chat so far. Rewrite the last question so it makes sense on its own."*

```
"and what about the second one?"  ──►  "What does payer response code 2 (QUERIED) mean?"
```

Both are wired up the same way:

1. **Search** with the context-rich query, so the right chunks are found.
2. **Answer** the *original* question, with the chat history included, so the reply sounds natural.

```
 question ──► [add context from memory] ──► search ──► re-rank ──► Qwen answers
                    ▲                                                 │
                    └──────────── memory (last N turns) ◄────────────┘
```

## ⚠️ Measured: the simple trick wins

`python evaluate.py followups` (8 conversations, hybrid + re-rank):

| How the follow-up was searched | Hit@1 | Hit@5 | MRR | Extra time |
|---|---|---|---|---|
| no memory (as typed) | 50% | 75% | 0.60 | 0 |
| Qwen rewrite alone | 62–75% | 88–100% | 0.75–0.88 | ~1.2–1.7 s |
| **previous question + follow-up** | **88%** | **100%** | **0.94** | **0** |
| previous question + Qwen rewrite | 88% | 100% | 0.94 | ~1.5 s |

Why is the clever Qwen rewrite worse? A 4-billion-parameter model sometimes picks the **wrong item**.
Asked about "the queried one", it once wrote *"...partially approved?"*. A wrong rewrite
sends the search in the wrong direction. Gluing on the previous question can't make that mistake,
because it never throws information away.

So the default is **`QUERY_REWRITE = "context"`**. The Qwen modes still exist, and they also glue on
the previous question so a bad rewrite does less harm. With a bigger chat model they may win.
**Measure again** if you change models. (8 conversations is a small exam, so add your own!)

## Three pieces in `memory.py`

| Piece | What it does |
|---|---|
| `ConversationMemory` | Keeps the last `MEMORY_TURNS` question/answer pairs. Old answers are shortened (`HISTORY_ANSWER_MAX_CHARS`) so they don't push the sources out of the context window. |
| `looks_like_follow_up()` | A free, instant guess (no AI): very short questions, questions that start with "and / what about / also...", or that contain pointing words ("it", "that", "second", "those"). |
| `prepare()` | Decides what to search for: the question as-is, previous question + follow-up (`context`), or previous question + Qwen rewrite (`auto` / `always`). Long chains are cut to the last 60 words. |
| `rewrite()` | Calls Qwen with `temperature=0` and at most 80 tokens. `_clean_rewrite()` throws away bad output (empty, or so long that Qwen started *answering* instead of rewriting). In that case we fall back to the original question. |

## Settings (`rag/config.py`)

| Setting | Meaning |
|---|---|
| `MEMORY_TURNS = 3` | How many earlier turns are remembered. |
| `QUERY_REWRITE = "context"` | `context` = previous question + follow-up (free, default). `auto` = previous question + Qwen rewrite, for follow-ups only. `always` = the same for every question. `off` = no memory in the search (Phase 1/2). |
| `HISTORY_ANSWER_MAX_CHARS = 1200` | Old answers are cut to this length. |

In `chat.py` you can change it live with `/rewrite context|auto|always|off`, look at the memory with `/memory`,
and forget everything with `/clear`. When a question is rewritten, the chat shows
`searching for: …` so you can see what was actually searched.

## Measuring it

`data/eval/followups.json` holds small two-turn conversations. `python evaluate.py followups`
really chats through the first question (Qwen writes a real answer), then searches for the follow-up
in three ways:

| Strategy | How the follow-up is searched |
|---|---|
| no memory | exactly as typed |
| previous question + follow-up | the `context` mode |
| Qwen rewrite | only the stand-alone question from `rewrite()` |
| previous question + rewrite | the `auto` mode |

It prints Hit@1 / Hit@5 / MRR for each, plus the average rewrite time.

## "Didn't understand, can you clarify?" — a message with NO topic

A real chat that went wrong:

```
You: what is search api used for?
AI:  ...discovering payers [1]
You: didnt understand can you clarify..
AI:  I could not find anything relevant in your documents.     ← dead end!
```

What happened:

1. `looks_like_follow_up()` said **no**. 5 words, no "and / what about", no "it / that".
2. So we searched for *"didnt understand can you clarify"* on its own. Those words match nothing.
3. The re-ranker (correctly!) threw every chunk away. With 0 chunks, Qwen is not even asked.

The message isn't a question about your documents. It's a question about **the last answer**.
Searching again is the wrong move, so now there are three paths. They live in
[`rag/pipeline.py`](../../rag/pipeline.py), shared by `chat.py`, `evaluate.py` and the web UI:

```
message ──► is_clarification()?  (clarify words, and NO topic words)
               yes ─► re-use the last answer's sources    origin = "clarify"  (no search, instant)
               no  ─► prepare() ─► search ─► re-rank      origin = "search"
                                     │
                                     └─ found nothing, but it's a follow-up?
                                            ─► last answer's sources       origin = "fallback"
```

| Piece | What it does |
|---|---|
| `is_clarification()` | True for *"didn't get that"*, *"can you clarify?"*, *"explain it more simply"*, *"give an example"*... but only if no real topic words remain. *"Can you clarify the claim deadline?"* is a **new** question, so it is searched normally. |
| `Turn.sources` | Memory now also remembers **which chunks** each answer used, so they can be re-used. |
| `CLARIFY_HINT` | Tells Qwen: *"the user didn't understand, explain again more simply, same sources, maybe an example"*. |
| fallback | A follow-up that finds nothing gets the previous sources rather than a dead end. Qwen is still told to say "I could not find this" if they don't answer it. |

The chat shows `clarification -> re-using the sources of the previous answer` when this happens.

**Measured** (`evaluate.py followups`, 11 conversations: the 8 above + 3 clarifications).
The `chat pipeline (default)` row is exactly what `chat.py` does:

| How the follow-up was searched | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| no memory (as typed) | 36% | 55% | 0.44 |
| previous question + follow-up | 91% | 100% | 0.955 |
| Qwen rewrite alone | 55–73% | 82–91% | 0.67–0.82 |
| **chat pipeline (default)** | **91%** | **100%** | **0.955** |

All 3 clarifications come out at **#1 with origin "clarify"**, in 0.0 s, because there is no search.
The same chat replayed: the second answer is now a step-by-step explanation with an example.

> **Lesson from building the exam:** the first version of one test expected a fact (the IST timestamp rule)
> that the *previous answer never used*. A clarification should re-explain **what was said**, so the
> evidence must come from the previous answer's sources. Check that your test measures the right thing!


## 🧪 Try it yourself

1. `python chat.py`, ask *"What diagnosis type codes exist?"*, then *"what does the last one mean?"*.
   Look at the `searching for:` line.
2. Type `/rewrite auto` and ask the same follow-up again. Is Qwen's rewrite right? Is it worth the wait?
3. Type `/rewrite off` and try once more. Compare the sources.
4. Add your own conversation to `data/eval/followups.json` and run `python evaluate.py followups`.
