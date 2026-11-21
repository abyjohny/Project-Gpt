"""
evaluate.py - PHASE 2: Give your RAG system a SCORE, so you can measure improvements.
====================================================================================

"It feels better" is not a measurement. Change one setting and some answers get
better while others silently get worse. So we build a small EXAM:

    1. A list of questions where we KNOW which page holds the answer.
    2. Run every question through the retriever.
    3. Check: did the right page come back? At which position?

Then we can compare setups fairly:  vector  vs  keyword  vs  hybrid  vs  hybrid + re-ranker

Usage:
    python evaluate.py generate --n 25     # let Qwen write exam questions from your documents
    python evaluate.py run                 # take the exam with every setup, print a score table
    python evaluate.py run --misses        # also show which questions failed
    python evaluate.py followups           # Phase 3: test follow-up questions (conversation memory)

The exam lives in data/eval/questions.json - you can (and should!) add your own
hand-written questions there too:
    {"question": "What is the claim submission deadline?", "source": "policy.pdf", "page": 12}
    {"question": "Which code means consent expired?", "source": "codes.docx",
     "evidence": "E-1047 The patient's consent artefact has expired"}

HOW WE DECIDE "CORRECT" (Phase 3):
    Every generated question stores an EVIDENCE quote: the sentence that holds the answer.
    A retrieved chunk is correct if it CONTAINS that quote (we compare word triplets,
    so small differences like "|" between table cells don't matter).
    This does not depend on HOW the document was chunked, so you can change the
    chunking strategy and still compare scores fairly: before vs after.

Learn more: docs/learning/13_measuring_quality.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

from rich.console import Console
from rich.table import Table

from rag import config, generator, keyword_search, memory, ollama_client, pipeline, reranker, retriever, vector_store

console = Console()

# name shown in the table -> (search mode, use re-ranker?)
SETUPS = {
    "vector":          ("vector", False),
    "keyword":         ("keyword", False),
    "hybrid":          ("hybrid", False),
    "hybrid + rerank": ("hybrid", True),
}

QUESTION_PROMPT = """Here is a passage from a document:

\"\"\"
{passage}
\"\"\"

Write ONE question that a person might ask, which this passage answers.
Rules:
- The question must be answerable using only this passage.
- Ask it the way a real user would: natural and short.
- Where possible, use your own words instead of copying phrases from the passage.

Reply in EXACTLY this format (two lines, nothing else):
QUESTION: <your question>
EVIDENCE: <copy word-for-word the ONE sentence or table row from the passage that contains the answer>"""


# ----------------------------------------------------------------
# Comparing text without caring about formatting
# ----------------------------------------------------------------

def _words(text: str) -> list[str]:
    """'| E-1047 | Consent expired |'  ->  ['e', '1047', 'consent', 'expired']"""
    return re.findall(r"[a-z0-9]+", text.lower())


def evidence_coverage(evidence: str, text: str) -> float:
    """How much of the evidence quote appears in `text`? 0.0 (none) ... 1.0 (all).

    We compare word TRIPLETS ("consent artefact has", "artefact has expired", ...)
    instead of single words, so the words must also appear in the same ORDER.
    """
    ev = _words(evidence)
    tx = _words(text)
    if len(ev) < 3:                       # very short quote -> every word must be there
        return 1.0 if ev and set(ev) <= set(tx) else 0.0
    ev_tri = {tuple(ev[i:i + 3]) for i in range(len(ev) - 2)}
    tx_tri = {tuple(tx[i:i + 3]) for i in range(len(tx) - 2)}
    return len(ev_tri & tx_tri) / len(ev_tri)


EVIDENCE_MATCH = 0.6   # at least 60% of the quote must be inside the retrieved chunk


# ----------------------------------------------------------------
# Loading / saving the exam
# ----------------------------------------------------------------

def load_questions() -> list[dict]:
    if not config.EVAL_QUESTIONS_FILE.exists():
        return []
    return json.loads(config.EVAL_QUESTIONS_FILE.read_text(encoding="utf-8"))


def save_questions(items: list[dict]) -> None:
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_QUESTIONS_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------
# 1. GENERATE questions with Qwen
# ----------------------------------------------------------------

def _clean_question(raw: str) -> str:
    line = raw.strip()
    line = re.sub(r"^(question\s*:|q\s*:|\d+[.)])\s*", "", line, flags=re.IGNORECASE)
    return line.strip().strip('"\'*').strip()


def _parse_reply(raw: str) -> tuple[str, str]:
    """Pull the QUESTION and EVIDENCE lines out of Qwen's reply."""
    question = evidence = ""
    for line in raw.strip().splitlines():
        clean = line.strip().strip("*").strip()
        if clean.upper().startswith("QUESTION:"):
            question = _clean_question(clean[len("QUESTION:"):])
        elif clean.upper().startswith("EVIDENCE:"):
            evidence = clean[len("EVIDENCE:"):].strip().strip('"\'').strip()
    return question, evidence


def generate(n: int, seed: int) -> None:
    ids, texts, metadatas = vector_store.get_all()
    pool = [i for i, t in enumerate(texts) if len(t) >= 300]   # skip tiny chunks (titles, footers)
    if not pool:
        console.print("[yellow]No chunks to build questions from. Run ingest.py first.[/]")
        return

    random.Random(seed).shuffle(pool)
    console.print(f"Asking [cyan]{config.CHAT_MODEL}[/] to write {n} questions...")

    new_items = []
    for i in pool:
        if len(new_items) >= n:
            break
        prompt = QUESTION_PROMPT.format(passage=texts[i][:2500])
        raw = "".join(ollama_client.chat([{"role": "user", "content": prompt}], temperature=0.3, stream=False))
        question, evidence = _parse_reply(raw)

        # Quality gate: the evidence must REALLY be in the passage (models sometimes paraphrase).
        if not question or len(_words(evidence)) < 4 or evidence_coverage(evidence, texts[i]) < 0.9:
            console.print("  [dim]skipped one (evidence was not an exact quote)[/]")
            continue

        meta = metadatas[i]
        new_items.append({
            "question": question,
            "evidence": evidence,
            "source": meta.get("source"),
            "page": meta.get("page", 0),
            "generated": True,
        })
        console.print(f"  [dim]{len(new_items):>3}.[/] {question}  [dim]({meta.get('source')}, p{meta.get('page', 0)})[/]")

    # Keep your hand-written questions; replace previously generated ones.
    hand_written = [q for q in load_questions() if not q.get("generated")]
    save_questions(hand_written + new_items)
    console.print(f"\n[green]Saved[/] {len(new_items)} generated + {len(hand_written)} hand-written questions "
                  f"to {config.EVAL_QUESTIONS_FILE}")
    console.print("Tip: open the file and delete any weird questions. Next:  [bold]python evaluate.py run[/]")


# ----------------------------------------------------------------
# 2. RUN the exam
# ----------------------------------------------------------------

def is_correct(result, item: dict) -> bool:
    """Did the retriever open the book at the right place?"""
    if result.metadata.get("source") != item.get("source"):
        return False
    evidence = item.get("evidence")
    if evidence:                                # best: the answer sentence is inside the chunk
        quotes = evidence if isinstance(evidence, list) else [evidence]   # several places may hold the answer
        return any(evidence_coverage(q, result.text) >= EVIDENCE_MATCH for q in quotes)
    page = item.get("page") or 0
    if page:                                    # PDFs: the right page is good enough
        return result.metadata.get("page") == page
    if item.get("expected_chunk_id"):           # old exams (before Phase 3): the exact chunk
        return result.id == item["expected_chunk_id"]
    return True                                 # hand-written, no page given: right file is enough


def run(top_k: int, show_misses: bool) -> None:
    items = load_questions()
    if not items:
        console.print("[yellow]No exam questions yet.[/] Run:  python evaluate.py generate --n 25")
        return

    setups = dict(SETUPS)
    if not reranker.is_available():
        setups.pop("hybrid + rerank")
        console.print("[yellow]PyTorch not installed - skipping the re-ranker setup.[/]")
    else:
        with console.status("Loading the re-ranker..."):
            reranker.warm_up()
    keyword_search.get_index()

    table = Table(title=f"Retrieval exam - {len(items)} questions, top-{top_k}")
    table.add_column("Setup", style="cyan")
    table.add_column("Hit@1", justify="right")
    table.add_column(f"Hit@{top_k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Avg time", justify="right", style="dim")

    misses: dict[str, list[str]] = {}
    for name, (mode, use_rr) in setups.items():
        hit1 = hitk = reciprocal_sum = 0.0
        total_ms = 0.0
        misses[name] = []
        with console.status(f"Running setup: {name}"):
            for item in items:
                started = time.perf_counter()
                results = retriever.retrieve(item["question"], top_k=top_k, mode=mode, use_reranker=use_rr)
                total_ms += (time.perf_counter() - started) * 1000

                position = next((p for p, r in enumerate(results, start=1) if is_correct(r, item)), None)
                if position:
                    hitk += 1
                    hit1 += position == 1
                    reciprocal_sum += 1 / position     # 1st -> 1.0, 2nd -> 0.5, 3rd -> 0.33 ...
                else:
                    misses[name].append(item["question"])

        n = len(items)
        table.add_row(name, f"{hit1 / n:.0%}", f"{hitk / n:.0%}", f"{reciprocal_sum / n:.3f}", f"{total_ms / n:.0f} ms")

    console.print(table)
    console.print("[dim]Hit@1 = right page ranked FIRST.  Hit@K = right page anywhere in the top K.\n"
                  "MRR = average of 1/position (1.0 = always first). Higher is better for all three.[/]")

    if show_misses:
        for name, questions in misses.items():
            if questions:
                console.print(f"\n[bold]Missed by {name}[/] ({len(questions)}):")
                for q in questions:
                    console.print(f"  - {q}")


# ----------------------------------------------------------------
# 3. FOLLOW-UP exam (Phase 3 - conversation memory)
# ----------------------------------------------------------------
# data/eval/followups.json holds small conversations:
#   {"setup": ["What diagnosis type codes exist?"], "follow_up": "What does the last one mean?",
#    "source": "...docx", "evidence": "final | Final Diagnosis - definitive confirmed diagnosis"}
# We REALLY chat through the setup questions (Qwen writes real answers), then check
# which chunks the search finds for the follow-up - searched in three different ways.

FOLLOW_UP_STRATEGIES = {
    "no memory (as typed)":       "raw",      # Phase 1/2: search the follow-up words only
    "previous question + follow-up": "concat",  # cheap trick: glue the last question in front
    "Qwen rewrite":               "rewrite",  # ask Qwen for a stand-alone question
    "previous question + rewrite": "combo",   # both: a rewrite mistake is softened by the context
    "chat pipeline (default)":    "pipeline", # what chat.py does: clarify re-use / search / fallback
}


def _position(results, item: dict) -> int | None:
    return next((p for p, r in enumerate(results, start=1) if is_correct(r, item)), None)


def run_followups(top_k: int) -> None:
    path = config.EVAL_DIR / "followups.json"
    if not path.exists():
        console.print(f"[yellow]No follow-up exam found at {path}[/]")
        return
    items = json.loads(path.read_text(encoding="utf-8"))
    use_rr = reranker.is_available()
    if use_rr:
        with console.status("Loading the re-ranker..."):
            reranker.warm_up()
    keyword_search.get_index()

    scores = {name: [] for name in FOLLOW_UP_STRATEGIES}       # positions (None = miss)
    rewrite_ms: list[float] = []
    detail = Table(title="Follow-up questions", show_lines=True)
    detail.add_column("Follow-up", style="bold")
    detail.add_column("Rewritten by Qwen", style="yellow")
    for name in FOLLOW_UP_STRATEGIES:
        detail.add_column(name.split(" (")[0], justify="center")

    for number, item in enumerate(items, start=1):
        with console.status(f"Conversation {number}/{len(items)}: chatting through the setup..."):
            mem = memory.ConversationMemory(rewrite_mode="context")
            for question in item["setup"]:                       # a REAL conversation first
                plan = pipeline.find_sources(question, mem, top_k=top_k, mode="hybrid", use_reranker=use_rr)
                answer = "".join(generator.generate_answer(question, plan.results, mem.as_messages(), plan.hint))
                mem.add(question, plan.search_query, answer, plan.results)

            follow_up = item["follow_up"]
            started = time.perf_counter()
            rewritten = mem.rewrite(follow_up)                   # measure the Qwen rewrite itself
            rewrite_ms.append((time.perf_counter() - started) * 1000)
            queries = {
                "raw": follow_up,
                "concat": f"{mem.turns[-1].search_query} {follow_up}",
                "rewrite": rewritten,
                "combo": f"{mem.turns[-1].search_query} {rewritten}",
            }
            row = [follow_up, rewritten]
            for name, key in FOLLOW_UP_STRATEGIES.items():
                if key == "pipeline":
                    plan = pipeline.find_sources(follow_up, mem, top_k=top_k, mode="hybrid", use_reranker=use_rr)
                    results = plan.results
                else:
                    results = retriever.retrieve(queries[key], top_k=top_k, mode="hybrid", use_reranker=use_rr)
                position = _position(results, item)
                scores[name].append(position)
                cell = f"[green]#{position}[/]" if position else "[red]miss[/]"
                if key == "pipeline" and plan.origin != "search":
                    cell += f" [dim]({plan.origin})[/]"
                row.append(cell)
            detail.add_row(*row)

    console.print(detail)
    summary = Table(title=f"Follow-up exam - {len(items)} conversations, top-{top_k}, hybrid"
                          + (" + rerank" if use_rr else ""))
    summary.add_column("How the follow-up was searched", style="cyan")
    summary.add_column("Hit@1", justify="right")
    summary.add_column(f"Hit@{top_k}", justify="right")
    summary.add_column("MRR", justify="right")
    n = len(items)
    for name, positions in scores.items():
        hit1 = sum(p == 1 for p in positions)
        hitk = sum(p is not None for p in positions)
        mrr = sum(1 / p for p in positions if p) / n
        summary.add_row(name, f"{hit1 / n:.0%}", f"{hitk / n:.0%}", f"{mrr:.3f}")
    console.print(summary)
    console.print(f"[dim]Qwen rewrite cost: {sum(rewrite_ms) / len(rewrite_ms):.0f} ms per follow-up on average.[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how well retrieval finds the right pages.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="let Qwen write exam questions from your chunks")
    gen.add_argument("--n", type=int, default=25, help="how many questions (default 25)")
    gen.add_argument("--seed", type=int, default=42, help="change to pick different chunks")

    run_cmd = sub.add_parser("run", help="run the exam with every setup")
    run_cmd.add_argument("--k", type=int, default=config.TOP_K, help="top-K to check (default TOP_K)")
    run_cmd.add_argument("--misses", action="store_true", help="list the questions each setup failed")

    fu_cmd = sub.add_parser("followups", help="test follow-up questions (conversation memory)")
    fu_cmd.add_argument("--k", type=int, default=config.TOP_K, help="top-K to check (default TOP_K)")

    args = parser.parse_args()

    if not ollama_client.is_running():
        console.print("[red]Ollama is not running.[/] Start the Ollama app (or run `ollama serve`).")
        sys.exit(1)
    if vector_store.count() == 0:
        console.print("[yellow]Your knowledge base is empty.[/] Run ingest.py first.")
        sys.exit(1)

    if args.command == "generate":
        generate(args.n, args.seed)
    elif args.command == "followups":
        run_followups(args.k)
    else:
        run(args.k, args.misses)


if __name__ == "__main__":
    main()
