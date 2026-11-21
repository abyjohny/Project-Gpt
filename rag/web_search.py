"""
web_search.py - PHASE 4: Let the AI look things up on the live internet ("Cowork" mode).
=======================================================================================

Your documents only know what is IN them. Questions like
    "What is the latest NHCX circular?"   or   "How does FHIR R5 differ from R4?"
need fresh, outside information. So in Cowork mode we ALSO search the web:

    question --> [1] web search (DuckDuckGo, no API key)   -> titles + short snippets
             --> [2] open the best pages, read their text  -> real paragraphs
             --> [3] keep only the paragraphs that match the question (BM25-style word overlap)
             --> [4] hand them to Qwen as extra SOURCES, numbered after your document chunks

Why step 2? Search snippets are tiny (~150-400 characters) - often too thin to
answer anything. Reading the page itself gives the AI real material to cite.

Privacy note: this is the ONLY part of the project that talks to the internet,
and it only runs when you switch the UI to "Cowork". Your documents never leave
your computer - only the search query does.

Learn more: docs/learning/16_web_ui.md
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

import requests

from rag.keyword_search import tokenize
from rag.vector_store import SearchResult

MAX_RESULTS = 5            # search results to collect
PAGES_TO_READ = 3          # how many of them we actually open
FETCH_TIMEOUT = 6          # seconds per page - a slow site must not freeze the chat
PAGE_TEXT_CHARS = 1500     # how much of each page we keep for the AI
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class WebSearchError(RuntimeError):
    """Raised when the web search itself fails (offline, rate limited...)."""


# ----------------------------------------------------------------
# [1] Search
# ----------------------------------------------------------------
def search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """[{title, url, snippet}, ...] from DuckDuckGo."""
    try:
        from ddgs import DDGS   # the package formerly known as duckduckgo-search
    except ImportError as exc:
        raise WebSearchError("Web search needs the `ddgs` package:  pip install ddgs") from exc
    try:
        raw = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:   # network errors, rate limits, ...
        raise WebSearchError(f"Web search failed: {exc}") from exc
    return [
        {"title": r.get("title", "").strip(), "url": r.get("href", ""), "snippet": r.get("body", "").strip()}
        for r in raw if r.get("href")
    ]


# ----------------------------------------------------------------
# [2] Read a page (plain standard-library HTML parsing, nothing to install)
# ----------------------------------------------------------------
class _TextExtractor(HTMLParser):
    """Collects the readable text blocks of a page, skipping menus, scripts and styles."""

    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button"}
    BLOCKS = {"p", "li", "h1", "h2", "h3", "h4", "td", "th", "pre", "blockquote", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buffer: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCKS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCKS:
            self._flush()

    def handle_data(self, data):
        if not self._skip_depth:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
        self._buffer = []
        if len(text) >= 40:        # shorter = menu items, buttons, captions
            self.blocks.append(text)


def fetch_page_blocks(url: str) -> list[str]:
    """Download one page and return its readable paragraphs ([] if anything goes wrong)."""
    try:
        response = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
        if response.status_code != 200 or "html" not in response.headers.get("content-type", ""):
            return []
        parser = _TextExtractor()
        parser.feed(response.text[:2_000_000])   # cap giant pages
        parser.close()
        return parser.blocks
    except Exception:
        return []


# ----------------------------------------------------------------
# [3] Keep only the paragraphs that talk about the question
# ----------------------------------------------------------------
def best_passages(question: str, blocks: list[str], limit: int = PAGE_TEXT_CHARS) -> str:
    """Pick the paragraphs sharing the most words with the question, in page order."""
    wanted = set(tokenize(question))
    if not blocks or not wanted:
        return ""
    scored = [(len(wanted & set(tokenize(b))), i, b) for i, b in enumerate(blocks)]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    chosen, used = [], 0
    for _, index, block in scored:
        if used + len(block) > limit and chosen:
            break
        chosen.append((index, block[:limit]))
        used += len(block)
    return "\n".join(block for _, block in sorted(chosen))


# ----------------------------------------------------------------
# [4] Everything together -> SearchResult objects the generator understands
# ----------------------------------------------------------------
def search_and_read(question: str, query: str | None = None) -> list[SearchResult]:
    """Search the web for `query` (default: the question) and return readable sources."""
    hits = search(query or question)
    if not hits:
        return []

    to_read = hits[:PAGES_TO_READ]
    with ThreadPoolExecutor(max_workers=len(to_read)) as pool:   # open pages in parallel
        pages = list(pool.map(fetch_page_blocks, [h["url"] for h in to_read]))

    results: list[SearchResult] = []
    for position, hit in enumerate(hits):
        text = hit["snippet"]
        if position < len(pages):
            passages = best_passages(question, pages[position])
            if passages:
                text = f"{hit['snippet']}\n{passages}" if hit["snippet"] else passages
        if not text:
            continue
        results.append(SearchResult(
            text=text,
            metadata={"source": hit["title"] or hit["url"], "url": hit["url"], "type": "web", "page": 0},
            id=hit["url"],
            found_by=["web"],
        ))
    return results
