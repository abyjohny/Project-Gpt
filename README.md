# 🤖 Project GPT — Your Own Local ChatGPT

A private AI assistant that runs **entirely on your laptop**, built phase by phase to learn how it works.

**Stack:** Qwen3.5 (brain) · Ollama (runs the model) · RAG (knows your documents) · *local UI (Phase 3)*

```
 Your files ──► load ──► chunk ──► embed ──► ChromaDB        (python ingest.py)
                                                 │
 Your question ──► embed ──► search ─────────────┘
                                │
                     top 5 chunks + question ──► Qwen3.5 ──► answer with [sources]   (python chat.py)
```

## Quick start

```powershell
# 1. Put your PDFs / DOCX / TXT / code files into:
#    data\documents\

# 2. Build the knowledge base
.\.venv\Scripts\python.exe ingest.py

# 3. Chat with your documents
.\.venv\Scripts\python.exe chat.py --debug
```

Requires Ollama running with: `qwen3.5:4b` and `nomic-embed-text`.

## 📚 Learn

Start here → **[docs/learning/README.md](docs/learning/README.md)**

## Project layout

```
├── data/documents/        your files (not committed to git)
├── data/vector_db/        generated knowledge base (not committed)
├── docs/learning/         beginner-friendly notes, one per concept
├── playground/            small standalone experiments
├── rag/                   the RAG engine, one file per step
│   ├── config.py          all settings
│   ├── loader.py          1. read files
│   ├── chunker.py         2. cut into chunks
│   ├── embedder.py        3. text → vectors
│   ├── vector_store.py    4. store / search (ChromaDB)
│   ├── retriever.py       5-6. question → relevant chunks
│   ├── generator.py       7-8. prompt → grounded answer
│   └── ollama_client.py   raw HTTP calls to Ollama
├── ingest.py              build / update the knowledge base
└── chat.py                terminal chat
```

## Roadmap

- [x] **Phase 1**: RAG from scratch (raw Python)
- [ ] **Phase 2**: Better RAG: hybrid search, re-ranking, OCR, evaluation
- [ ] **Phase 3**: Local chat UI
- [ ] **Phase 4**: Tools & agents (Git, Python, files, terminal)
- [ ] **Phase 5**: Internet search
- [ ] **Phase 6**: Specialised assistants (tutor, API/Postman, ClaimLens code expert)
- [ ] **Phase 7**: Automations & voice
