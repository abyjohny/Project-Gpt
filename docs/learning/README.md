# 📚 Learning Path — Build Your Own Local ChatGPT

Welcome! These notes go with the code in this project. Read them **in order**.
Each one explains **one idea**, in plain language, and points to the exact code file that does it.

> **How to use these notes**
> 1. Read the note.
> 2. Open the code file it mentions and read the comments.
> 3. Do the "🧪 Try it yourself" part at the end.
> 4. Only then move to the next note.

## Phase 1 — RAG from scratch

| # | Note | What you'll understand | Code |
|---|------|------------------------|------|
| 0 | [Setup & Ollama](00_setup_and_ollama.md) | What Ollama is, what a "model" is, why 4b fits your laptop | `rag/ollama_client.py` |
| 1 | [What is RAG?](01_what_is_rag.md) | The big picture: the open-book exam idea | `rag/config.py` |
| 2 | [Loading documents](02_loading_documents.md) | Getting plain text out of PDFs and Word files | `rag/loader.py` |
| 3 | [Chunking](03_chunking.md) | Why and how we cut text into small pieces | `rag/chunker.py` |
| 4 | [Embeddings](04_embeddings.md) | How text becomes "meaning numbers" | `rag/embedder.py` |
| 5 | [Vector database](05_vector_database.md) | Where those numbers are stored and searched | `rag/vector_store.py` |
| 6 | [Retrieval](06_retrieval.md) | Finding the right pages for a question | `rag/retriever.py` |
| 7 | [Generation](07_generation.md) | How the AI writes an answer from those pages | `rag/generator.py` |
| 8 | [Run it & experiment](08_run_and_experiment.md) | Use it on your own files, tune it, fix problems | `ingest.py`, `chat.py` |

## Phase 2 — Better search

| # | Note | What you'll understand | Code |
|---|------|------------------------|------|
| 9 | [Keyword search](09_keyword_search.md) | BM25: the classic search engine | `rag/keyword_search.py` |
| 10 | [Hybrid search](10_hybrid_search.md) | Meaning + keywords, merged with RRF | `rag/retriever.py` |
| 11 | [Re-ranking](11_reranking.md) | A second, smarter judge for the shortlist | `rag/reranker.py` |
| 12 | [OCR](12_ocr_scanned_pdfs.md) | Reading scanned PDFs and images | `rag/loader.py` |
| 13 | [Measuring quality](13_measuring_quality.md) | Hit@K, MRR: give your RAG a score | `evaluate.py` |

## Phase 3 — Smarter chunks & conversations

| # | Note | What you'll understand | Code |
|---|------|------------------------|------|
| 14 | [Structure-aware chunking](14_structure_aware_chunking.md) | Headings, tables, and why the first try was worse | `rag/chunker.py` |
| 15 | [Conversation memory](15_conversation_memory.md) | Making "and the second one?" work | `rag/memory.py` |

## Phase 4 — Your own chat app (you are here)

| # | Note | What you'll understand | Code |
|---|------|------------------------|------|
| 16 | [The web UI](16_web_ui.md) | Streaming, clickable citations, saved chats, Cowork web search, live GPU stats | `server.py`, `web/`, `rag/sessions.py`, `rag/web_search.py` |

## Coming next

| Phase | What we'll build |
|-------|------------------|
| 5 | Tools & agents: let the AI use Git, Python, files, terminal |
| 6 | Deeper internet research (multi-step search, reading more pages) |
| 7 | Special assistants: tutor, API/Postman helper, ClaimLens code expert |
| 8 | Automations and voice |

## 📖 Mini dictionary (come back here whenever a word confuses you)

| Word | Plain meaning |
|------|---------------|
| **Model / LLM** | The AI "brain" file. It predicts the next word, very, very well. |
| **Ollama** | A free app that runs AI models on your own computer. |
| **Parameters (4b, 9b…)** | The "brain cells" of a model. 4b = 4 billion. More = smarter but heavier. |
| **Token** | A small piece of a word. "unbelievable" ≈ "un" + "believ" + "able". ~4 letters ≈ 1 token. |
| **Context window** | How much text the AI can "keep in view" at once (like the size of its desk). |
| **RAG** | Retrieval-Augmented Generation: *find the right pages first, then answer from them.* |
| **Chunk** | A small piece of a document (a few paragraphs). |
| **Embedding / vector** | A list of numbers that captures what a text *means*. |
| **Vector database** | A database that finds texts with *similar meaning* very fast. |
| **Similarity** | A score from 0 to 1 — how close two meanings are. |
| **Top-K** | "Give me the K best matches." K = 5 means the 5 best chunks. |
| **Metadata** | Labels attached to a chunk: which file, which page. |
| **Prompt** | The full message we send to the AI. |
| **System prompt** | The permanent "house rules" we give the AI before every chat. |
| **Hallucination** | When the AI confidently makes something up. RAG is our main weapon against it. |
| **Temperature** | Creativity dial. Low = strict and factual. High = creative. |
