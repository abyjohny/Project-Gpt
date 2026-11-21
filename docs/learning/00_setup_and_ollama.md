# 0️⃣ Setup & Ollama — running an AI on your own laptop

## What is a "model"?

When people say "ChatGPT" or "Qwen", they mean a **model**: one very large file
full of numbers. Those numbers were learned by reading a huge amount of text.

A model has one skill: **given some text, guess the next word.**
It does this extremely well, so well that it can explain, summarise, write code and hold conversations.

```
"The capital of France is"  →  model  →  "Paris"
```

That's it. Everything else (chatting, answering from documents, using tools) is built
**around** this one skill.

## What is Ollama?

Running a model yourself used to be complicated. **Ollama** makes it easy:

- It **downloads** models for you (`ollama pull qwen3.5:4b`)
- It **runs** them on your graphics card (GPU)
- It opens a **small local web address** (`http://localhost:11434`) that any program can talk to

> 💡 **"localhost"** means *this computer*. When our Python code talks to
> `localhost:11434`, the message never leaves your laptop.

> ⚠️ **A real bug we hit: "localhost" made every request 2 seconds slower!**
> Computers have two kinds of addresses: the old style (IPv4, like `127.0.0.1`)
> and the new style (IPv6, like `::1`). On Windows, the name "localhost" tries the
> **new** address first. Ollama only listens on the **old** one, so Windows
> knocked on the wrong door, waited ~2 seconds, then tried the right door.
>
> Each question was taking ~2.1 s just to be turned into numbers. After we
> switched to `http://127.0.0.1:11434` in `rag/config.py`, it took ~0.065 s,
> about **30× faster**, with no other changes.
>
> **Lesson:** if something is slow by a suspiciously *round, constant* amount
> (exactly ~2 s every time), it's usually a network timeout, not "the AI being slow".

Think of Ollama as a **kitchen** and the models as **chefs**. Our Python code is the
waiter: it takes the order (your question) to the kitchen and brings back the dish (the answer).

## The two models we use

| Model | Job | Size | Analogy |
|-------|-----|------|---------|
| `qwen3.5:4b` | Reads and writes answers | ~3 GB | The **expert** who answers questions |
| `nomic-embed-text` | Turns text into meaning-numbers | ~0.3 GB | The **librarian** who files every page by topic |

Why two? Because they do completely different jobs. The librarian is small and fast
and organises your library. The expert is big and smart and writes the answers.
You'll understand the librarian fully in [note 4](04_embeddings.md).

## Why "4b" for your laptop?

Your laptop has an **RTX 3050 with 6 GB of video memory (VRAM)**.
The model has to fit in that memory to run fast.

| Model size | Memory needed (approx.) | On your laptop |
|-----------|--------------------------|----------------|
| 2b | ~2 GB | Very fast, but less smart |
| **4b** | **~3.5 GB** | ✅ **Best balance** |
| 9b | ~6.5 GB | Doesn't fully fit → part of it runs on the slower CPU |

> 💡 **"Quantized"**: the downloaded models are *compressed*. Each number is
> stored with fewer digits (like rounding 3.14159265 to 3.14). The model gets
> much smaller and only very slightly less accurate. That's how a 4-billion-number
> model fits in about 3 GB.

## How our code talks to Ollama

Open [`rag/ollama_client.py`](../../rag/ollama_client.py). It's just two web requests:

**1. Embed** (turn text into numbers)
```json
POST http://localhost:11434/api/embed
{ "model": "nomic-embed-text", "input": ["hello world"] }

→ { "embeddings": [[0.012, -0.551, 0.309, ...]] }
```

**2. Chat** (get an answer)
```json
POST http://localhost:11434/api/chat
{
  "model": "qwen3.5:4b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is RAG?"}
  ]
}
```

**Roles** in a chat:
- `system` → the house rules ("always cite sources", "be concise")
- `user` → what you typed
- `assistant` → what the AI replied before (so it remembers the conversation)

> 💡 The model has **no memory** between requests. To "remember" a conversation,
> we send the **whole conversation again** every time. ChatGPT works the same way.

## 🧪 Try it yourself

Open a terminal and run:

```powershell
ollama list                      # which models you have
ollama run qwen3.5:4b            # chat directly in the terminal (type /bye to exit)
```

Ask it: *"What did the NHCX document say about preauth cancellation?"*

It will either say it doesn't know, or (worse) **make something up**.
It has never seen your document. Keep that in mind; it's exactly the problem
[RAG](01_what_is_rag.md) solves.
