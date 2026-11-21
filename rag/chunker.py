"""
chunker.py - STEP 2: Cut long text into small, overlapping pieces ("chunks").
============================================================================

WHY?
  - A 200-page PDF is far too big to give to the AI in one go.
  - If we search a whole book, we get back... the whole book. Not helpful.
  - Small pieces let us find the EXACT paragraph that answers the question.

HOW (the "recursive" idea, explained like cutting a cake):
  1. First try to cut at the NATURAL borders: blank lines (paragraphs).
  2. If a paragraph is still too big, cut it at line breaks.
  3. Still too big? Cut at sentence ends (". ").
  4. Still too big? Cut at spaces between words.
  5. Last resort: cut by raw character count.
  Then we glue small neighbouring pieces back together until each chunk
  is close to CHUNK_SIZE.

OVERLAP:
  Each chunk repeats the last ~150 characters of the previous chunk
  (as WHOLE sentences - we never repeat half a sentence).
  So if an important sentence sits on the border, it is not lost.
  To make this possible, we first break the text into sentence-sized
  "bricks", then build chunks from the bricks. The last brick or two of
  one chunk become the first bricks of the next.

      chunk 1: [==========|ooo]
      chunk 2:            [ooo|==========|ooo]
      chunk 3:                           [ooo|==========]
                           ^ shared "overlap" text

PHASE 3 - STRUCTURE-AWARE CHUNKING  (config.CHUNKING_STRATEGY = "structure")
  Cutting purely by size is like tearing pages out of a book at random:
  a chunk may start in the middle of one topic and end in another, and a
  table can be torn in half (row 7 without its column names is meaningless).

  So we first follow the document's OWN structure, then cut by size inside it:

    1. SECTIONS  - split at headings. A chunk never mixes two topics.
    2. ADDRESS   - every chunk starts with its location in the document:
                     "Location: Section 4: Workflow Codes > 4.1 Provider-Side Codes"
                   A chunk that just says "| 3 | QUERIED | ... |" now ALSO says
                   what it is about -> much easier to find by meaning AND keywords.
    3. TABLES    - kept whole. A big table is cut BETWEEN rows, and the header
                   row (the column names) is repeated at the top of every piece.
    4. GLUE      - tiny neighbouring sections (5.6.1, 5.6.2, 5.6.3) are glued
                   into one chunk, so we don't end up with 40-character chunks.
    5. Inside a long section we still use the Phase 1 recursive splitter.

Learn more: docs/learning/03_chunking.md, docs/learning/14_structure_aware_chunking.md
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from rag import config
from rag.loader import Document

# Natural borders, from "best place to cut" to "worst place to cut".
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Used for the second, finer pass that makes sentence-sized "bricks" for overlap.
# We stop at sentences on purpose: we never want to repeat half a sentence.
SENTENCE_SEPARATORS = ["\n\n", "\n", ". "]

# "## 4.1 Provider-Side Workflow Codes"  ->  level 2, title "4.1 Provider-Side Workflow Codes"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")

# The line under a table's column names:  "| --- | :---: | --- |"
_TABLE_SEPARATOR = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+$")


@dataclass
class Chunk:
    """A small piece of a document, ready to be turned into a vector."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def clean_text(text: str) -> str:
    """Tidy up messy text that PDFs often produce."""
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)          # many spaces   -> one space
    text = re.sub(r" *\n *", "\n", text)          # spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ blank lines -> one blank line
    return text.strip()


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces that are each <= chunk_size characters."""
    if len(text) <= chunk_size or not separators:
        # Small enough - or we ran out of places to cut, so keep it whole.
        return [text]

    separator, remaining_separators = separators[0], separators[1:]

    # Last resort: hard cut every chunk_size characters.
    if separator == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Put the separator back so we don't lose full stops / line breaks.
        piece = part + (separator if i < len(parts) - 1 else "")
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            # Still too big -> try the next, finer separator.
            pieces.extend(_split_recursive(piece, chunk_size, remaining_separators))
    return pieces


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Glue small pieces together into chunks, repeating a little text between them."""
    chunks: list[str] = []
    window: list[str] = []   # pieces that make up the chunk being built
    window_length = 0

    for piece in pieces:
        # Adding this piece would make the chunk too big -> close the chunk.
        if window and window_length + len(piece) > chunk_size:
            chunks.append("".join(window).strip())

            # Keep only the TAIL of the old chunk to start the new one (the overlap).
            while window and (window_length > overlap or window_length + len(piece) > chunk_size):
                window_length -= len(window[0])
                window.pop(0)

        window.append(piece)
        window_length += len(piece)

    if window:
        chunks.append("".join(window).strip())
    return chunks


def split_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_length: int | None = None,
) -> list[str]:
    """Clean -> split recursively -> merge with overlap. Returns a list of strings."""
    # None = "use the value from config.py" (read now, so experiments can change config).
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    min_length = config.MIN_CHUNK_LENGTH if min_length is None else min_length
    # Pass 1: make sure no piece is bigger than a whole chunk.
    pieces = _split_recursive(clean_text(text), chunk_size, SEPARATORS)

    # Pass 2: break pieces into sentence-sized "bricks" (<= overlap if possible),
    # so the tail we repeat between chunks is made of whole sentences.
    # (Very long sentences stay whole - they just won't be repeated.)
    bricks: list[str] = []
    for piece in pieces:
        bricks.extend(_split_recursive(piece, overlap, SENTENCE_SEPARATORS))

    # Pass 3: glue bricks back together into chunks, with overlap.
    chunks = _merge_with_overlap(bricks, chunk_size, overlap)
    return [c for c in chunks if len(c) >= min_length]


# ----------------------------------------------------------------
# PHASE 3: structure-aware chunking
# ----------------------------------------------------------------

# A heading path is a list of (level, title), e.g.
#   [(1, "Section 4: Workflow Codes Reference"), (2, "4.1 Provider-Side Workflow Codes")]
HeadingPath = list[tuple[int, str]]


@dataclass
class Section:
    """Everything written under one heading (until the next heading)."""

    path: HeadingPath
    blocks: list[tuple[str, str]] = field(default_factory=list)   # ("text" | "table", content)

    @property
    def size(self) -> int:
        return sum(len(content) for _, content in self.blocks)


def parse_sections(text: str, start_path: HeadingPath | None = None) -> tuple[list[Section], HeadingPath]:
    """Read Markdown-style text and cut it into Sections at every heading.

    Returns the sections AND the heading path at the very end, so the next PDF
    page can continue inside the same section ("section 4.1 continues on page 8").
    """
    path: HeadingPath = list(start_path or [])
    sections: list[Section] = []
    current = Section(path=list(path))
    text_lines: list[str] = []
    table_lines: list[str] = []
    in_code = False   # inside a ``` code block, a "#" line is code, not a heading

    def flush_text() -> None:
        content = "\n".join(text_lines).strip()
        if content:
            current.blocks.append(("text", content))
        text_lines.clear()

    def flush_table() -> None:
        if table_lines:
            current.blocks.append(("table", "\n".join(table_lines)))
        table_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_table()
            in_code = not in_code
            text_lines.append(line)
            continue
        heading = None if in_code else _HEADING.match(stripped)
        if heading:
            flush_text()
            flush_table()
            if current.blocks:
                sections.append(current)
            level, title = len(heading.group(1)), heading.group(2).strip()
            # A new level-2 heading closes the previous level-2 (and deeper) sections.
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
            current = Section(path=list(path))
        elif not in_code and stripped.startswith("|") and stripped.endswith("|"):
            flush_text()
            table_lines.append(stripped)
        else:
            flush_table()
            text_lines.append(line)

    flush_text()
    flush_table()
    if current.blocks:
        sections.append(current)
    return sections, path


def _heading_line(level: int, title: str) -> str:
    return "#" * level + " " + title


def merge_small_sections(sections: list[Section], max_size: int | None = None) -> list[Section]:
    """Glue tiny neighbouring sections together (only siblings / children of the same parent).

        5.6 Policy Discovery API          (intro, 120 chars)
          5.6.1 Internal Endpoint         (180 chars)       ->  ONE chunk about "5.6"
          5.6.2 Supported Identifiers     (150 chars)           with the sub-headings inside
    """
    max_size = config.SECTION_MERGE_MAX if max_size is None else max_size
    if max_size <= 0:
        return sections
    merged: list[Section] = []
    for section in sections:
        group = merged[-1] if merged else None
        if group is not None and group.size + section.size <= max_size and section.path:
            parent = section.path[:-1]
            if parent == group.path:                 # child of the group -> append it
                group.blocks.append(("text", _heading_line(*section.path[-1])))
                group.blocks.extend(section.blocks)
                continue
            if group.path and parent == group.path[:-1]:  # sibling -> group now covers the parent
                group.blocks.insert(0, ("text", _heading_line(*group.path[-1])))
                group.path = parent
                group.blocks.append(("text", _heading_line(*section.path[-1])))
                group.blocks.extend(section.blocks)
                continue
        merged.append(Section(path=list(section.path), blocks=list(section.blocks)))
    return merged


def split_table(table: str, max_size: int, overlap_rows: int | None = None) -> list[str]:
    """Cut a big Markdown table BETWEEN rows; every piece repeats the header row.

    Like text overlap, the last `overlap_rows` rows of a piece are repeated at the
    top of the next piece, so a row near the border appears with its neighbours.
    """
    overlap_rows = config.TABLE_OVERLAP_ROWS if overlap_rows is None else overlap_rows
    lines = table.splitlines()
    has_header = len(lines) >= 2 and set(lines[1].replace("|", "").strip()) <= set("-: ")
    header = lines[:2] if has_header else []
    rows = lines[2:] if has_header else lines
    header_size = sum(len(h) + 1 for h in header)

    pieces: list[str] = []
    current: list[str] = []
    current_size = header_size
    for row in rows:
        if current and current_size + len(row) + 1 > max_size:
            pieces.append("\n".join(header + current))
            carried = current[-overlap_rows:] if overlap_rows > 0 else []
            # Only carry rows over if they leave room for new ones.
            if header_size + sum(len(r) + 1 for r in carried) + len(row) + 1 > max_size:
                carried = []
            current = list(carried)
            current_size = header_size + sum(len(r) + 1 for r in current)
        if header_size + len(row) > max_size:
            # One single row is bigger than a chunk (a cell full of text): split the row itself.
            for part in split_text(row, max(max_size - header_size, 200), min_length=1):
                pieces.append("\n".join(header + [part]))
            current, current_size = [], header_size
            continue
        current.append(row)
        current_size += len(row) + 1
    if current:
        pieces.append("\n".join(header + current))
    return pieces


def _pack(pieces: list[str], max_size: int) -> list[str]:
    """Put consecutive small pieces together until a chunk is full."""
    packed: list[str] = []
    for piece in pieces:
        if packed and len(packed[-1]) + 2 + len(piece) <= max_size:
            packed[-1] += "\n\n" + piece
        else:
            packed.append(piece)
    return packed


def section_label(path: HeadingPath) -> str:
    """[(1, 'Section 4'), (2, '4.1 Codes')]  ->  'Section 4 > 4.1 Codes'"""
    return " > ".join(title for _, title in path)


def location_header(path: HeadingPath, style: str | None = None) -> str:
    """The 'address' line written on top of every chunk (see config.CHUNK_LOCATION).

        "full" -> "Location: Section 4: Workflow Codes > 4.1 Provider-Side Codes"
        "last" -> "Location: 4.1 Provider-Side Codes"
        "none" -> ""   (the address is still saved in metadata["section"])
    """
    style = style or config.CHUNK_LOCATION
    if not path or style == "none":
        return ""
    label = section_label(path[-1:] if style == "last" else path)
    return f"Location: {label}\n\n"


def compact_table(table: str) -> str:
    """Drop the '| --- | --- |' line. It means nothing to a search engine, but it costs tokens."""
    return "\n".join(line for line in table.splitlines() if not _TABLE_SEPARATOR.match(line.strip()))


def chunk_section(section: Section, chunk_size: int | None = None) -> list[str]:
    """Turn one section into chunks, each starting with the section's 'address'."""
    chunk_size = chunk_size or config.SECTION_CHUNK_SIZE
    header = location_header(section.path)
    budget = max(chunk_size - len(header), 300)   # room left for the real content

    pieces: list[str] = []
    for kind, content in section.blocks:
        if kind == "table":
            parts = split_table(content, budget)
            if config.TABLE_COMPACT:
                parts = [compact_table(p) for p in parts]
            pieces.extend(parts)
        else:
            pieces.extend(split_text(content, chunk_size=budget, min_length=1))

    chunks = [header + body for body in _pack(pieces, budget)]
    return [c for c in chunks if len(c) - len(header) >= config.MIN_CHUNK_LENGTH]


def split_structured(text: str, start_path: HeadingPath | None = None) -> tuple[list[tuple[str, str]], HeadingPath]:
    """Structure-aware splitting. Returns ([(chunk_text, section_label), ...], end_path)."""
    sections, end_path = parse_sections(text, start_path)
    results: list[tuple[str, str]] = []
    for section in merge_small_sections(sections):
        label = section_label(section.path)
        results.extend((chunk, label) for chunk in chunk_section(section))
    return results, end_path


def _make_chunk_id(metadata: dict, index: int, text: str) -> str:
    """A stable "fingerprint" ID: same file + same position + same text = same ID.

    Stable IDs mean re-running ingestion UPDATES chunks instead of duplicating them.
    """
    raw = f"{metadata.get('path')}|{metadata.get('page')}|{index}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


# ----------------------------------------------------------------
# Public function
# ----------------------------------------------------------------

def chunk_documents(documents: list[Document], strategy: str = config.CHUNKING_STRATEGY) -> list[Chunk]:
    """Turn loaded Documents into Chunks, copying the metadata onto every chunk."""
    all_chunks: list[Chunk] = []
    # PDFs arrive page by page. Remember the heading path at the end of each page,
    # so a section that continues on the next page keeps its "address".
    open_sections: dict[str, HeadingPath] = {}

    for doc in documents:
        if strategy == "structure" and doc.metadata.get("structured"):
            file_key = str(doc.metadata.get("path", doc.metadata.get("source")))
            pieces, open_sections[file_key] = split_structured(doc.text, open_sections.get(file_key))
        else:
            pieces = [(piece, "") for piece in split_text(doc.text)]

        for index, (piece, section) in enumerate(pieces):
            metadata = {**doc.metadata, "chunk_index": index, "char_count": len(piece)}
            if section:
                metadata["section"] = section
            all_chunks.append(
                Chunk(id=_make_chunk_id(metadata, index, piece), text=piece, metadata=metadata)
            )
    return all_chunks
