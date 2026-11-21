"""
rag - our hand-built Retrieval-Augmented Generation package.

Each file is ONE step of the RAG pipeline:

    loader.py        Step 1  Read files -> plain text
    chunker.py       Step 2  Cut text into small overlapping pieces
    ollama_client.py         Talk to Ollama (embeddings + chat)
    embedder.py      Step 3  Turn text into meaning-numbers (vectors)
    vector_store.py  Step 4  Save / search vectors in ChromaDB
    retriever.py     Step 5-6  Question -> most relevant chunks
    generator.py     Step 7-8  Chunks + question -> grounded answer
"""
