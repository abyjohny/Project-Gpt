"""
ingest.py - Build (or update) your knowledge base.
=================================================

This runs the INGESTION pipeline (Steps 1-4) for every file in data/documents:

    file -> [1] load -> [2] chunk -> [3] embed -> [4] store in vector DB

Usage:
    python ingest.py                 # ingest everything in data/documents
    python ingest.py path/to/file.pdf  # ingest specific files or folders
    python ingest.py --reset         # wipe the knowledge base and rebuild it
    python ingest.py --list          # show what is already in the knowledge base
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from rag import chunker, config, embedder, loader, ollama_client, vector_store

console = Console()


def show_knowledge_base() -> None:
    sources = vector_store.list_sources()
    if not sources:
        console.print("[yellow]The knowledge base is empty. Add files to data/documents and run: python ingest.py[/]")
        return
    table = Table(title=f"Knowledge base - {sum(sources.values())} chunks")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for name, n in sorted(sources.items()):
        table.add_row(name, str(n))
    console.print(table)


def collect_files(targets: list[str]) -> list[Path]:
    if not targets:
        return loader.find_files(config.DOCUMENTS_DIR)
    files: list[Path] = []
    for target in targets:
        path = Path(target).resolve()
        files.extend(loader.find_files(path) if path.is_dir() else [path])
    return files


def ingest(files: list[Path]) -> int:
    """Load, chunk, embed and store `files`. Returns how many chunks were added/updated."""
    summary = Table(title="Ingestion summary")
    summary.add_column("File", style="cyan")
    summary.add_column("Pages/parts", justify="right")
    summary.add_column("Chunks", justify="right", style="green")
    summary.add_column("Seconds", justify="right")

    total_chunks = 0
    for path in files:
        started = time.perf_counter()
        console.print(f"\n[bold]>> {path.name}[/]")

        # STEP 1 - load (scanned pages / images are OCR'd by Qwen here - that part is slow)
        documents = loader.load_file(path)
        if not documents:
            console.print("   [yellow]No text found (blank file, or OCR is switched off). Skipped.[/]")
            continue
        ocr_pages = sum(1 for d in documents if d.metadata.get("ocr"))

        # STEP 2 - chunk
        chunks = chunker.chunk_documents(documents)
        ocr_note = f"  [magenta]({ocr_pages} read with OCR)[/]" if ocr_pages else ""
        console.print(f"   [dim]step 1 load  :[/] {len(documents)} page(s)/part(s){ocr_note}")
        console.print(f"   [dim]step 2 chunk :[/] {len(chunks)} chunk(s)")

        # STEP 3 - embed (with a progress bar, this is the slow part)
        with Progress(
            TextColumn("   [dim]step 3 embed :[/]"), BarColumn(), MofNCompleteColumn(),
            console=console, transient=True,
        ) as progress:
            task = progress.add_task("embed", total=len(chunks))
            vectors = embedder.embed_documents(
                [c.text for c in chunks],
                on_progress=lambda done, total: progress.update(task, completed=done),
            )
        console.print(f"   [dim]step 3 embed :[/] {len(vectors)} vector(s) x {len(vectors[0])} numbers each")

        # STEP 4 - store (remove the old version of this file first, so edits don't leave stale chunks)
        vector_store.delete_file(str(path))
        vector_store.add_chunks(chunks, vectors)
        console.print("   [dim]step 4 store :[/] saved to ChromaDB")

        seconds = time.perf_counter() - started
        summary.add_row(path.name, str(len(documents)), str(len(chunks)), f"{seconds:.1f}")
        total_chunks += len(chunks)

    console.print()
    console.print(summary)
    console.print(f"[bold green]Done.[/] Added/updated {total_chunks} chunks. "
                  f"Knowledge base now holds {vector_store.count()} chunks.")
    console.print("Next:  [bold]python chat.py[/]")
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents into the local knowledge base.")
    parser.add_argument("targets", nargs="*", help="files or folders (default: data/documents)")
    parser.add_argument("--reset", action="store_true", help="delete the knowledge base first")
    parser.add_argument("--list", action="store_true", help="show what is in the knowledge base")
    args = parser.parse_args()

    if args.list:
        show_knowledge_base()
        return

    if not ollama_client.is_running():
        console.print("[red]Ollama is not running.[/] Start the Ollama app (or run `ollama serve`) and try again.")
        sys.exit(1)

    if args.reset:
        vector_store.reset()
        console.print("[yellow]Knowledge base wiped.[/]")

    files = collect_files(args.targets)
    if not files:
        console.print(f"[yellow]No supported files found.[/] Put PDFs/DOCX/TXT in: {config.DOCUMENTS_DIR}")
        return

    console.print(f"Found [bold]{len(files)}[/] file(s). Embedding model: [cyan]{config.EMBED_MODEL}[/]")
    ingest(files)


if __name__ == "__main__":
    main()
