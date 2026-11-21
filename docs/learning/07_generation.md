# 7️⃣ Generation — the AI writes the answer

## What we actually send to Qwen

This is the moment everything comes together. We build one message (the **prompt**) that contains:

```
┌─────────────────────────── SYSTEM (house rules) ───────────────────────────┐
│ You answer using ONLY the provided sources.                                 │
│ Cite sources like [1]. If the answer isn't there, say                       │
│ "I could not find this in your documents." Never guess.                     │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────── USER ────────────────────────────────────────────┐
│ SOURCES:                                                                    │
│                                                                             │
│ [1] (NHCX_spec.pdf, page 12)                                                │
│ To cancel a pre-authorization, the provider sends a Cancel request...       │
│ ---                                                                         │
│ [2] (NHCX_spec.pdf, page 13)                                                │
│ The cancel request must include the original preauth ID...                  │
│                                                                             │
│ ====================                                                        │
│ QUESTION: How do I cancel a preauth?                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

And Qwen replies:

> To cancel a preauth, the provider sends a Cancel request [1] that includes the original preauth ID [2].

That's the **A** (Augmented: we *added* the sources) and the **G** (Generation: Qwen *writes*) of RAG.

> 💡 You can see the **real** prompt any time: type `/prompt` in the chat after asking a question.
> Do it at least once. It removes all the "magic".

## Why each house rule exists

| Rule | Why |
|------|-----|
| *"Use ONLY the sources"* | Stops the AI mixing in things it half-remembers from training |
| *"Cite [1], [2]"* | You can **check** every claim against the original page |
| *"Say you couldn't find it"* | Gives the AI a polite way out instead of **inventing** an answer |
| *"Point out disagreements"* | Useful when two versions of a document conflict |

> ⚠️ Small models (like 4b) follow rules **most** of the time, not always.
> That's why we **also** have code-level protection: if retrieval finds nothing,
> we don't call the AI at all.

## Temperature — the creativity dial

```
TEMPERATURE = 0.2
```

When the AI picks its next word, it has many options with different chances.

- **Low (0–0.3)**: almost always picks the most likely word → factual, consistent ✅ for documents
- **High (0.8–1.0)**: sometimes picks less likely words → creative, varied ✅ for stories/brainstorming

For answering from documents, we want a strict, careful student, not a poet.

## Streaming — the typing effect

Instead of waiting 10 seconds for the full answer, we ask Ollama to **stream**:
it sends each word as soon as it's ready. That's why the answer "types itself" like ChatGPT.
Same total time, but it *feels* much faster.

## Memory for follow-up questions

The model **forgets everything** between requests. To allow follow-ups like
*"now make that a checklist"*, `chat.py` keeps the **last 3 question/answer pairs**
and sends them again with each new question. `/clear` wipes that memory.

## "Thinking" mode is off

Qwen3.5 can "think out loud" before answering (it writes hidden reasoning first).
That helps with hard maths and logic, but it's **slower** and not needed for "read this page and answer".
We switched it off with `ENABLE_THINKING = False`. Try turning it on later and compare!

## Read the code

[`rag/generator.py`](../../rag/generator.py):
- `SYSTEM_PROMPT`: the house rules (edit them and see what changes!)
- `build_context()`: numbers the chunks as `[1]`, `[2]`…
- `build_messages()`: system + history + sources + question
- `generate_answer()`: skips the AI entirely if nothing was retrieved

## 🧪 Try it yourself

1. Ask a question, then type `/prompt`. Read the **entire** prompt Qwen received.
2. Edit `SYSTEM_PROMPT` and add: *"Always answer in exactly 3 bullet points."* Restart `chat.py`. Did it obey?
3. Change `TEMPERATURE` to `1.0`, ask the same question 3 times. Notice the answers vary more.
4. Set `ENABLE_THINKING = True`. Is the answer better? How much slower is it?
