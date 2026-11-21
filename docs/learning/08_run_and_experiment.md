# 8️⃣ Run it & experiment — your RAG on your own files

## ▶️ Everyday usage

All commands are run from the project folder (`D:\dev\personal\Project GPT`).

**1. Make sure Ollama is running.** It starts automatically with Windows (look for the 🦙 llama icon
in the system tray). If not, open the **Ollama** app from the Start menu.

**2. Put your files in** `data\documents\` (PDF, DOCX, TXT, MD, code; sub-folders are fine).

**3. Build the knowledge base:**
```powershell
.\.venv\Scripts\python.exe ingest.py
```
You'll see every step happen for every file:
```
>> NHCX_spec.pdf
   step 1 load  : 186 page(s)/part(s)
   step 2 chunk : 512 chunk(s)
   step 3 embed : 512 vector(s) x 768 numbers each
   step 4 store : saved to ChromaDB
```

**4. Chat:**
```powershell
.\.venv\Scripts\python.exe chat.py           # normal
.\.venv\Scripts\python.exe chat.py --debug   # also shows retrieved chunks
```

> 💡 **Tip:** activate the virtual environment once per terminal and you can just type `python`:
> ```powershell
> .\.venv\Scripts\Activate.ps1
> python chat.py
> ```

## 🧰 Useful commands

| Command | What it does |
|---------|--------------|
| `python ingest.py` | Ingest / update all files in `data/documents` |
| `python ingest.py some\file.pdf` | Ingest just one file |
| `python ingest.py --list` | Show which files are in the knowledge base |
| `python ingest.py --reset` | Wipe everything and rebuild (needed after changing chunk size or embedding model) |
| In chat: `/debug` | Show/hide retrieved chunks |
| In chat: `/k 8` | Retrieve 8 chunks instead of 5 |
| In chat: `/mode hybrid` | Search mode: `vector`, `keyword` or `hybrid` (Phase 2) |
| In chat: `/rerank` | Turn the re-ranker on/off (Phase 2) |
| In chat: `/rewrite context` | How follow-ups are searched: `context`, `auto`, `always`, `off` (Phase 3) |
| In chat: `/memory` | Show what the assistant remembers of the conversation (Phase 3) |
| In chat: `/prompt` | Show the exact prompt sent to Qwen |
| In chat: `/clear` | Forget the conversation |
| In chat: `/exit` | Quit |
| `python evaluate.py run --misses` | Score retrieval on the exam, list failed questions (Phase 2) |
| `python evaluate.py followups` | Score follow-up questions / conversation memory (Phase 3) |

## 🔬 Experiments (this is where the real learning happens)

Do these one at a time. After each change in `rag/config.py`, **re-ingest with `--reset`**
if you touched chunking or embeddings.

### Experiment 1: Chunk size
1. Pick 5 questions you know the answers to. Write them down.
2. Try `CHUNK_SIZE = 500`, then `1000`, then `2000` (re-ingest each time).
3. For each, ask your 5 questions with `--debug`. Which size finds the right page most often?

### Experiment 2: TOP_K
Ask a question whose answer spans several pages. Try `/k 3`, `/k 5`, `/k 10`.
When does the answer become complete? When does it start getting confused?

### Experiment 3: The "not in my documents" test
Ask 3 questions that are definitely **not** in your files.
Does it refuse properly? What similarity scores do you see? Tune `MIN_SIMILARITY`.

### Experiment 4: Different words, same meaning
Ask the same question 3 ways (formal, casual, with synonyms). Is the same page found?

### Experiment 5: Bigger brain
Try `CHAT_MODEL = "qwen3.5:9b"` (run `ollama pull qwen3.5:9b` first).
Better answers? How much slower? (It won't fully fit in your 6 GB, so expect slower.)

> 📝 **Keep a learning log.** Write down what you changed and what happened.
> This "change one thing, observe" habit is exactly how AI engineers tune real RAG systems.

## 🩺 Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `Ollama is not running` | Ollama app closed | Open **Ollama** from Start menu, or run `ollama serve` |
| `model "..." not found` | Model not downloaded | `ollama pull qwen3.5:4b` and `ollama pull nomic-embed-text` |
| `No text found (scanned image PDF?)` | PDF is a picture | OCR is on by default (Phase 2, `OCR_ENABLED`) |
| Answers are wrong | Usually **retrieval**, not the AI | Use `--debug`. Are the right chunks there? If not → chunking / question wording |
| Answers say "couldn't find" but it IS there | Similarity bar too high, or chunk too blurry | Lower `MIN_SIMILARITY`, try smaller `CHUNK_SIZE` |
| Very slow answers | Model spilling out of GPU memory | Close other GPU apps; stick to 4b; lower `CONTEXT_WINDOW` |
| Re-ranking sometimes takes 3–30 s | Another program is using the GPU | Normal is ~0.7–1.3 s. Close games/browsers with GPU video, measure again |
| Changed settings but nothing changed | Old chunks still in DB | `python ingest.py --reset` |

## ✅ Phase 1 checklist

You've finished Phase 1 when you can explain, **in your own words**:

- [ ] Why we can't just paste a whole PDF into the AI
- [ ] What a chunk is, and why overlap exists
- [ ] What an embedding is (the map of meanings)
- [ ] What cosine similarity measures
- [ ] What a vector database stores for each chunk
- [ ] What TOP_K and MIN_SIMILARITY control
- [ ] What the final prompt sent to Qwen looks like
- [ ] Why bad answers are usually a retrieval problem

When all of these are ticked, you're ready for **Phase 2: better RAG** 🚀
