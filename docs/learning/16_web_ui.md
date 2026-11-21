# 16 - The Web UI: streaming, sources, sessions and Cowork mode

> **Code:** `server.py`, `web/index.html`, `web/app.js`, `web/style.css`,
> `rag/sessions.py`, `rag/web_search.py`

Until now you talked to your AI in a terminal (`python chat.py`). Phase 4 gives it a
real app in your browser, with no change to the RAG brain underneath. The web UI
calls the **same** `pipeline.find_sources()` and `generator.generate_answer()` that `chat.py` uses.

```
 Browser (web/app.js)                 server.py (FastAPI)                 rag/ (unchanged brain)
 ───────────────────                  ───────────────────                 ──────────────────────
 type question  ── POST /api/chat ──► worker thread ──► pipeline.find_sources() ─► search + re-rank
                                         │            ──► web_search (Cowork only)
 status line   ◄── event: status ───────┤
 source cards  ◄── event: sources ──────┤
 words appear  ◄── event: token  ───────┤ ◄── generator.generate_answer() ◄── Qwen (Ollama)
 speed + time  ◄── event: done   ───────┘ ──► sessions.save()  (data/sessions/*.json)
```

## 1. Streaming with Server-Sent Events (SSE)

A normal web request is "ask, wait, get everything at once". For an AI answer that means
staring at nothing for 10 seconds. Instead, the response **stays open** and the server pushes
small named messages as soon as they exist:

```
event: status
data: {"text": "Searching your documents..."}

event: token
data: {"text": "The deadline"}
```

Each message is `event:` + `data:` + an empty line. The browser reads the stream piece by piece
(`response.body.getReader()` in `app.js`) and splits on the empty lines.

**Why a worker thread?** Search, re-ranking and Qwen are ordinary *blocking* Python. If they ran
on the server's main loop, the whole server would freeze while one answer is being written:
no stats, no uploads, nothing. So `run_chat()` runs in a thread and drops events into a queue,
and the async side just forwards the queue to the browser.

**Stop button:** when you press Stop, the browser closes the connection. The server notices,
sets a `cancel` flag, the worker breaks out of the token loop and closes the stream to Ollama.
Qwen stops writing, so the GPU is free again straight away.

## 2. Citations you can click

The system prompt already makes Qwen write `[1]`, `[2][3]` after facts (note 07).
The UI turns those into chips:

1. The answer's markdown is converted to HTML (`marked`) and **sanitised** (`DOMPurify`).
   Never put AI output into a page without sanitising it: a document could contain HTML or script tags.
2. A tree walker finds `[n]` in the *text* (not inside code blocks) and replaces it with a button.
3. Clicking it opens the source card and shows the exact chunk text the AI was given.

That last step is the real point of RAG: **you can check the AI's homework.**

## 3. Sessions: memory that survives a restart

`chat.py` forgets everything when you quit. `rag/sessions.py` saves every chat as a JSON file
in `data/sessions/`, **including the sources of every answer**. When you reopen a chat, the
`ConversationMemory` (note 15) is rebuilt from those messages, so follow-ups ("and the second one?")
and clarifications ("didn't get it") still work after a restart.

## 4. Cowork mode: documents + the live web

| | Chat mode | Cowork mode |
|---|---|---|
| Sources | your documents only | your documents (top 4) + web pages |
| If nothing is found | "I could not find this in your documents." | may use general knowledge, **clearly marked** |
| Internet | never | only the search query is sent out |
| System prompt | `SYSTEM_PROMPT` | `COWORK_SYSTEM_PROMPT` |

`rag/web_search.py` works in four steps:

1. **Search** DuckDuckGo (no API key) for the question, or the follow-up-aware search query.
2. **Read** the top 3 pages in parallel (6-second timeout each, so a slow site can't freeze the chat).
   Search snippets alone are ~200 characters, too thin to answer from.
3. **Pick** the paragraphs that share the most words with the question (BM25's tokenizer, note 09).
4. **Wrap** them as `SearchResult` objects with `type: "web"`, so the generator numbers and cites
   them exactly like document chunks.

Documents come **first** in the numbering and the prompt says they are the most trusted:
your own spec beats a random blog post.

## 5. Live system stats

The System tab polls `/api/stats` every 2 seconds:

- **GPU memory / load / temperature** come from `nvidia-smi` (ships with the NVIDIA driver).
- **Loaded models** come from Ollama's `/api/ps`, which shows how much of Qwen is in VRAM.
- **Tokens per second** come from Ollama itself: the last streamed message carries `eval_count`
  (tokens written) and `eval_duration` (nanoseconds). `ollama_client.chat(stats=...)` captures them.

## 6. Safer uploads

Uploading a file runs the same steps as `python ingest.py <file>`, plus a few guards a web
endpoint needs:

- `Path(filename).name` strips folders, so a file named `../../server.py` can only land in `data/documents`.
- Only `SUPPORTED_EXTENSIONS` are accepted, up to 200 MB.
- If a file yields **0 chunks** (blank, or a scan OCR couldn't read), you get an error instead of a fake "success".
- After ingesting, the BM25 index is rebuilt. Re-ingesting an edited file can keep the
  same chunk count, and the index only auto-rebuilds when the count changes (note 09).

## 🧪 Try it yourself

1. Run `python server.py` and open <http://127.0.0.1:8000>.
2. Ask a document question, then click a `[1]` chip. Is the chunk really what the AI said?
3. Ask a follow-up ("explain that more simply"), then restart the server, reopen the chat from
   the sidebar, and ask another follow-up. Does it still know what "that" means?
4. Open **System & Stats** while an answer is streaming and watch GPU load and VRAM.
5. Switch to **Cowork** and ask something that is *not* in your documents. Which sources are
   web, and did the AI mark general-knowledge sentences?
6. Press **Stop** halfway through an answer and watch GPU load drop in the System tab.
