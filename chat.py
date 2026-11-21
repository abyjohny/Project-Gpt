"""
chat.py - Ask questions about your documents.
============================================

This runs the QUERY pipeline every time you ask something:

    question -> [rewrite follow-ups using memory] -> search (meaning + keywords)
             -> merge -> re-rank -> [7] build prompt -> [8] Qwen answers

Usage:
    python chat.py              # normal chat
    python chat.py --debug      # also show exactly which chunks were retrieved

Commands inside the chat:
    /debug          toggle showing the retrieved chunks and their scores
    /k 8            change how many chunks are given to the AI (TOP_K)
    /mode hybrid    search mode: vector | keyword | hybrid        (Phase 2)
    /rerank         toggle the re-ranker on/off                   (Phase 2)
    /rewrite context   follow-ups: context | auto | always | off  (Phase 3)
    /memory         show what the assistant remembers             (Phase 3)
    /reload         rebuild the keyword index (after re-ingesting changed files)
    /prompt         show the full prompt that was sent to Qwen last time
    /clear          forget the conversation history
    /exit           quit
"""

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag import (config, generator, intent, keyword_search, memory, ollama_client,
                 pipeline, reranker, retriever, vector_store)

console = Console()


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def show_retrieved(results, trace: dict) -> None:
    table = Table(title="Retrieved chunks (what the AI was allowed to read)", show_lines=True)
    table.add_column("#", style="bold")
    table.add_column("Found by", style="yellow")
    table.add_column("Meaning", justify="right", style="green")
    table.add_column("Keyword", justify="right", style="blue")
    table.add_column("Rerank", justify="right", style="magenta")
    table.add_column("Source", style="cyan")
    table.add_column("Text preview")
    for i, r in enumerate(results, start=1):
        preview = r.text[:180].replace("\n", " ") + ("..." if len(r.text) > 180 else "")
        table.add_row(
            str(i), "+".join(r.found_by), _fmt(r.similarity), _fmt(r.keyword_score, 2),
            _fmt(r.rerank_score), generator.format_source_detail(r.metadata), preview,
        )
    console.print(table)

    # A one-line "behind the scenes" report of the pipeline.
    steps = []
    if "meaning_ms" in trace:
        steps.append(f"meaning: {trace['meaning_found']} found in {trace['meaning_ms']:.0f} ms")
    if "keyword_ms" in trace:
        steps.append(f"keyword: {trace['keyword_found']} found in {trace['keyword_ms']:.0f} ms")
    steps.append(f"shortlist: {trace.get('candidates', 0)}")
    if "rerank_ms" in trace:
        steps.append(f"re-ranked in {trace['rerank_ms']:.0f} ms")
    steps.append(f"kept: {len(results)}")
    console.print("[dim]" + "  ->  ".join(steps) + "[/]")


def show_sources_footer(results) -> None:
    if not results:
        return
    lines = [f"[{i}] {generator.format_source_detail(r.metadata)}  ({r.score_name} {r.score:.2f})"
             for i, r in enumerate(results, start=1)]
    console.print(Panel("\n".join(lines), title="Sources", border_style="dim", expand=False))


def describe_settings(mode: str, use_reranker: bool, top_k: int, rewrite: str = config.QUERY_REWRITE) -> str:
    rr = "on" if use_reranker else "off"
    if use_reranker and not reranker.is_available():
        rr = "off (PyTorch not installed)"
    return (f"search: [cyan]{mode}[/]   re-ranker: [cyan]{rr}[/]   top-k: [cyan]{top_k}[/]   "
            f"follow-up rewrite: [cyan]{rewrite}[/]")


def show_memory(mem: memory.ConversationMemory) -> None:
    if not len(mem):
        console.print("[dim]Memory is empty.[/]")
        return
    table = Table(title=f"What I remember (last {mem.max_turns} turns)", show_lines=True)
    table.add_column("You asked")
    table.add_column("Searched for", style="yellow")
    table.add_column("Answer (start)", style="dim")
    for turn in mem.turns:
        searched = "(same)" if turn.search_query == turn.question else turn.search_query
        table.add_row(turn.question, searched, turn.answer[:120].replace("\n", " ") + "...")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your documents.")
    parser.add_argument("--debug", action="store_true", help="show retrieved chunks")
    args = parser.parse_args()

    if not ollama_client.is_running():
        console.print("[red]Ollama is not running.[/] Start the Ollama app (or run `ollama serve`).")
        sys.exit(1)

    chunk_count = vector_store.count()
    if chunk_count == 0:
        console.print("[yellow]Your knowledge base is empty.[/] Put files in data/documents and run: python ingest.py")
        sys.exit(1)

    debug, top_k = args.debug, config.TOP_K
    mode, use_reranker = config.RETRIEVAL_MODE, config.USE_RERANKER
    mem = memory.ConversationMemory()
    last_messages: list[dict] = []

    # Prepare the slow parts up-front, so the first question is not extra slow.
    with console.status("Preparing the keyword index and re-ranker..."):
        keyword_search.get_index()
        if use_reranker and reranker.is_available():
            reranker.warm_up()

    console.print(Panel.fit(
        f"[bold]Project GPT - Local RAG[/]\n"
        f"Chat model: [cyan]{config.CHAT_MODEL}[/]   Embeddings: [cyan]{config.EMBED_MODEL}[/]\n"
        f"Knowledge base: [green]{chunk_count}[/] chunks from {len(vector_store.list_sources())} file(s)\n"
        f"{describe_settings(mode, use_reranker, top_k, mem.rewrite_mode)}\n"
        f"[dim]Commands: /debug  /k N  /mode vector|keyword|hybrid  /rerank  /rewrite context|auto|always|off\n"
        f"          /memory  /reload  /prompt  /clear  /exit[/]",
        border_style="cyan",
    ))

    while True:
        try:
            question = console.input("\n[bold green]You >[/] ").strip().lstrip("\ufeff")   # \ufeff = invisible Windows marker
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue

        # ---- commands ----
        if question in ("/exit", "/quit"):
            break
        if question == "/debug":
            debug = not debug
            console.print(f"[dim]debug = {debug}[/]")
            continue
        if question.startswith("/k"):
            try:
                top_k = int(question.split()[1])
                console.print(f"[dim]TOP_K = {top_k}[/]")
            except (IndexError, ValueError):
                console.print("[red]Usage: /k 8[/]")
            continue
        if question.startswith("/mode"):
            parts = question.split()
            if len(parts) == 2 and parts[1] in retriever.VALID_MODES:
                mode = parts[1]
                console.print(describe_settings(mode, use_reranker, top_k, mem.rewrite_mode))
            else:
                console.print("[red]Usage: /mode vector | /mode keyword | /mode hybrid[/]")
            continue
        if question == "/rerank":
            if not reranker.is_available():
                console.print("[yellow]The re-ranker needs PyTorch + transformers (see docs/learning/11_reranking.md).[/]")
                continue
            use_reranker = not use_reranker
            if use_reranker:
                with console.status("Loading the re-ranker..."):
                    reranker.warm_up()
            console.print(describe_settings(mode, use_reranker, top_k, mem.rewrite_mode))
            continue
        if question.startswith("/rewrite"):
            parts = question.split()
            if len(parts) == 2 and parts[1] in memory.VALID_REWRITE_MODES:
                mem.rewrite_mode = parts[1]
                console.print(describe_settings(mode, use_reranker, top_k, mem.rewrite_mode))
            else:
                console.print("[red]Usage: /rewrite context | auto | always | off[/]")
            continue
        if question == "/memory":
            show_memory(mem)
            continue
        if question == "/reload":
            with console.status("Rebuilding the keyword index..."):
                keyword_search.get_index(force_rebuild=True)
            console.print(f"[dim]Keyword index rebuilt ({vector_store.count()} chunks).[/]")
            continue
        if question == "/clear":
            mem.clear()
            console.print("[dim]History cleared.[/]")
            continue
        if question == "/prompt":
            for m in last_messages:
                console.print(Panel(m["content"], title=m["role"], border_style="magenta"))
            continue

        # ---- small talk: greetings / thanks / "who are you?" skip RAG entirely ----
        if intent.is_small_talk(question):
            console.print("\n[bold cyan]AI >[/] ", end="")
            answer_parts = []
            try:
                for piece in generator.generate_small_talk(question, mem.as_messages()):
                    answer_parts.append(piece)
                    console.print(piece, end="", markup=False, highlight=False)
            except ollama_client.OllamaError as exc:
                console.print(f"\n[red]{exc}[/]")
                continue
            console.print()
            answer = "".join(answer_parts)
            if answer.strip():
                mem.add(question, question, answer, [])
            continue

        # ---- Phase 3: where do the sources come from? (clarify / follow-up / new question) ----
        started = time.perf_counter()
        status = "Understanding your follow-up..." if mem.needs_rewrite(question) else "Searching..."
        with console.status(status):
            plan = pipeline.find_sources(question, mem, top_k=top_k, mode=mode, use_reranker=use_reranker)
        retrieval_ms = (time.perf_counter() - started) * 1000
        results, hint, search_query, trace = plan.results, plan.hint, plan.search_query, plan.trace

        if plan.origin == "clarify":
            console.print("[dim]clarification -> re-using the sources of the previous answer (no new search)[/]")
        elif search_query != question:
            console.print(f"[dim]searching for: [yellow]{search_query}[/][/]")
        if plan.origin == "fallback":
            console.print("[dim]nothing new found -> falling back to the sources of the previous answer[/]")
        if debug and plan.origin != "clarify":
            show_retrieved(results, trace)
            console.print(f"[dim]retrieval took {retrieval_ms:.0f} ms in total[/]")

        # ---- steps 7 + 8: generation (streamed) ----
        history = mem.as_messages()
        last_messages = generator.build_messages(question, results, history, hint) if results else []
        console.print("\n[bold cyan]AI >[/] ", end="")
        answer_parts = []
        try:
            for piece in generator.generate_answer(question, results, history, hint):
                answer_parts.append(piece)
                console.print(piece, end="", markup=False, highlight=False)
        except ollama_client.OllamaError as exc:
            console.print(f"\n[red]{exc}[/]")
            continue
        console.print()
        show_sources_footer(results)

        # Remember this exchange (without the big SOURCES block) for follow-up questions,
        # plus WHICH chunks it used, so "can you clarify?" can re-use them.
        mem.add(question, search_query, "".join(answer_parts), results)

    console.print("\n[dim]Bye![/]")


if __name__ == "__main__":
    main()
