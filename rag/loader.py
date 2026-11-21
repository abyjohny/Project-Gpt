"""
loader.py - STEP 1: Read files and pull out plain text.
======================================================

The AI can only understand TEXT. A PDF or a Word file is a "box"
that contains text plus a lot of formatting (fonts, colours, layout).
The loader opens the box and takes out just the words.

We also remember WHERE each piece of text came from (file name, page number).
That information is called "metadata". Later, the AI uses it to cite
its sources, e.g.  "[NHCX_Spec.pdf, page 42]".

PHASE 3 - KEEP THE STRUCTURE
    Documents are not just a bag of words. They have CHAPTERS, HEADINGS and TABLES.
    Instead of flattening everything, the loader now writes the text as simple
    Markdown, which keeps that structure visible to the chunker:

        # Section 4: Workflow Codes Reference        <- heading level 1
        ## 4.1 Provider-Side Workflow Codes           <- heading level 2
        | Code | Constant | Description |             <- a table, row by row
        | --- | --- | --- |
        | 1 | SUBMITTED | Claim sent to payer |

    Word files: headings come from the paragraph style ("Heading 1", "Heading 2"...),
                and tables are kept IN PLACE (Phase 1 dumped them all at the end!).
    PDFs:       there are no heading labels, so we GUESS: text in a bigger font
                than normal is probably a heading. Tables are found by PyMuPDF.

Learn more: docs/learning/02_loading_documents.md
            docs/learning/14_structure_aware_chunking.md
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import docx            # python-docx  -> reads .docx
import pymupdf         # PyMuPDF      -> reads .pdf
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from rag import config, ocr


@dataclass
class Document:
    """One piece of loaded text plus the information about where it came from."""

    text: str
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------
# Tables -> Markdown (shared by Word and PDF)
# ----------------------------------------------------------------

def _clean_cell(value) -> str:
    """One table cell -> one tidy line of text."""
    text = " ".join(str(value or "").split())   # newlines / double spaces -> single spaces
    return text.replace("|", "/")              # "|" would break the Markdown table


def rows_to_markdown(rows: list[list]) -> str:
    """[["Code", "Meaning"], ["1", "Submitted"]]  ->  a Markdown table.

    The FIRST row is treated as the header row. A 1-column or 1-row "table" is
    usually just a framed box of text (e.g. a code sample), so it stays plain text.
    """
    rows = [[_clean_cell(c) for c in row] for row in rows]
    rows = [row for row in rows if any(row)]             # drop empty rows
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    if width == 1 or len(rows) == 1:
        return "\n".join(" ".join(c for c in row if c) for row in rows)

    rows = [row + [""] * (width - len(row)) for row in rows]   # make every row equally wide
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


# ----------------------------------------------------------------
# One small reader per file type
# ----------------------------------------------------------------

def _pdf_heading_levels(pdf) -> dict[float, int]:
    """Find which font sizes are 'heading sizes' in this PDF.

    1. Count how many letters are written in each font size.
    2. The most used size is the normal body text.
    3. Clearly bigger sizes are headings: biggest = level 1, next = level 2, ...
    """
    letters_per_size: Counter = Counter()
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    letters_per_size[round(span["size"], 1)] += len(span["text"].strip())
    if not letters_per_size:
        return {}
    body_size = letters_per_size.most_common(1)[0][0]
    heading_sizes = sorted(
        (s for s in letters_per_size if s >= body_size * config.PDF_HEADING_SIZE_RATIO), reverse=True
    )
    return {size: min(level, 4) for level, size in enumerate(heading_sizes, start=1)}


def _pdf_tables(page) -> list[tuple[pymupdf.Rect, str]]:
    """Find tables on a page: [(where it is, the table as Markdown), ...]."""
    if not config.PDF_DETECT_TABLES:
        return []
    try:
        found = page.find_tables()
    except Exception:     # very unusual page layouts can confuse the table finder
        return []
    tables = []
    for table in found.tables:
        markdown = rows_to_markdown(table.extract())
        if markdown:
            tables.append((pymupdf.Rect(table.bbox), markdown))
    return tables


def _pdf_page_to_markdown(page, heading_levels: dict[float, int]) -> str:
    """Rebuild one PDF page as Markdown: headings, paragraphs and tables in reading order."""
    tables = _pdf_tables(page)
    items: list[tuple[float, str]] = []    # (vertical position on the page, text)
    heading_bottom: float | None = None    # bottom edge of the heading we just wrote

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:          # 0 = text. (1 = picture -> skip)
            continue
        paragraph: list[str] = []
        paragraph_top = 0.0
        for line in block["lines"]:
            box = pymupdf.Rect(line["bbox"])
            middle = pymupdf.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
            if any(middle in rect for rect, _ in tables):
                continue                    # this text is inside a table -> the table already has it
            spans = [s for s in line["spans"] if s["text"].strip()]
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not spans or not text:
                continue

            level = heading_levels.get(max(round(s["size"], 1) for s in spans))
            if level and len(text) <= 120:  # a big-font, short line = heading
                if paragraph:
                    items.append((paragraph_top, "\n".join(paragraph)))
                    paragraph = []
                    heading_bottom = None
                prefix = "#" * level + " "
                # A long heading can wrap onto a 2nd line. Glue it back ONLY if this line
                # sits directly under the previous heading (not a table or gap further down).
                wrapped = (
                    heading_bottom is not None
                    and items and items[-1][1].startswith(prefix)
                    and box.y0 - heading_bottom < box.height * 0.6
                )
                if wrapped:
                    items[-1] = (items[-1][0], items[-1][1] + " " + text)
                else:
                    items.append((box.y0, prefix + text))
                heading_bottom = box.y1
            else:
                if not paragraph:
                    paragraph_top = box.y0
                paragraph.append(text)
                heading_bottom = None
        if paragraph:
            items.append((paragraph_top, "\n".join(paragraph)))
            heading_bottom = None

    # Put each table where it sits on the page (before the first text that is lower down).
    for rect, markdown in tables:
        position = next((i for i, (top, _) in enumerate(items) if top >= rect.y0), len(items))
        items.insert(position, (rect.y0, markdown))

    return "\n\n".join(text for _, text in items)


def _load_pdf(path: Path) -> list[Document]:
    """A PDF becomes ONE Document PER PAGE, so we can cite page numbers.

    Pages with (almost) no real text are probably SCANNED pictures.
    Phase 2: instead of skipping them, we let Qwen "read" the picture (OCR).
    Phase 3: normal pages are rebuilt as Markdown (headings + tables kept).
    """
    documents = []
    with pymupdf.open(path) as pdf:
        heading_levels = _pdf_heading_levels(pdf) if config.PDF_DETECT_HEADINGS else {}
        for page_index, page in enumerate(pdf):
            metadata = {"source": path.name, "page": page_index + 1, "structured": True}

            if len(page.get_text("text").strip()) < config.OCR_MIN_TEXT_CHARS and config.OCR_ENABLED:
                text = ocr.pdf_page_to_text(page)     # slow, but better than losing the page
                metadata["ocr"] = True                # remember: this text was "read", not copied
            else:
                text = _pdf_page_to_markdown(page, heading_levels)

            if text.strip():  # still empty? (blank page) -> skip it
                documents.append(Document(text=text, metadata=metadata))
    return documents


def _load_image(path: Path) -> list[Document]:
    """PNG/JPG/WEBP: a picture of text (screenshot, photo of a letter...) -> OCR."""
    if not config.OCR_ENABLED:
        return []
    text = ocr.image_to_text(path.read_bytes())
    return [Document(text=text, metadata={"source": path.name, "page": 0, "ocr": True})] if text else []


def _docx_heading_level(paragraph) -> int:
    """'Heading 2' -> 2,  'Title' -> 1,  anything else -> 0 (not a heading)."""
    style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    if style == "title":
        return 1
    if style.startswith("heading"):
        digits = "".join(ch for ch in style if ch.isdigit())
        return min(int(digits), 6) if digits else 1
    return 0


def _docx_table_rows(table) -> list[list[str]]:
    """Read a Word table. Merged cells appear several times in python-docx -> keep them once."""
    rows = []
    for row in table.rows:
        cells, previous = [], None
        for cell in row.cells:
            if cell._tc is not previous:        # same underlying cell = horizontally merged
                cells.append(cell.text)
            previous = cell._tc
        rows.append(cells)
    return rows


def _load_docx(path: Path) -> list[Document]:
    """Word files have no fixed pages, so the whole file is one Document.

    We walk through the document body IN ORDER - paragraphs and tables mixed,
    exactly as you see them in Word - and write it out as Markdown:
        Heading styles  -> "#", "##", "###" lines
        List paragraphs -> "- item"
        Tables          -> Markdown tables, right below the heading they belong to
    """
    word_file = docx.Document(str(path))
    parts: list[str] = []

    for element in word_file.element.body.iterchildren():
        if element.tag == qn("w:p"):                       # a paragraph
            paragraph = DocxParagraph(element, word_file)
            text = paragraph.text.strip()
            if not text:
                continue
            level = _docx_heading_level(paragraph)
            style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
            if level:
                parts.append("#" * level + " " + " ".join(text.split()))
            elif "list" in style:
                parts.append("- " + text)
            else:
                parts.append(text)
        elif element.tag == qn("w:tbl"):                   # a table
            markdown = rows_to_markdown(_docx_table_rows(DocxTable(element, word_file)))
            if markdown:
                parts.append(markdown)

    text = "\n\n".join(parts)
    metadata = {"source": path.name, "page": 0, "structured": True}
    return [Document(text=text, metadata=metadata)] if text.strip() else []


def _load_text(path: Path) -> list[Document]:
    """TXT, Markdown and code files are already plain text - just read them."""
    for encoding in ("utf-8", "utf-16", "latin-1"):   # try common encodings
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return []
    # Markdown already HAS structure (# headings, | tables |). Code files do NOT:
    # in Python a "#" line is a comment, not a heading - so only .md counts as structured.
    metadata = {"source": path.name, "page": 0, "structured": path.suffix.lower() == ".md"}
    return [Document(text=text, metadata=metadata)] if text.strip() else []


# ----------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------

def load_file(path: Path) -> list[Document]:
    """Pick the right reader based on the file extension."""
    extension = path.suffix.lower()
    if extension == ".pdf":
        docs = _load_pdf(path)
    elif extension == ".docx":
        docs = _load_docx(path)
    elif extension in config.IMAGE_EXTENSIONS:
        docs = _load_image(path)
    else:
        docs = _load_text(path)

    # Remember the full path too (useful for re-ingesting a changed file)
    for doc in docs:
        doc.metadata["path"] = str(path)
        doc.metadata["file_type"] = extension.lstrip(".")
    return docs


def find_files(folder: Path = config.DOCUMENTS_DIR) -> list[Path]:
    """List every supported file inside the folder (including sub-folders)."""
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and not p.name.startswith(".")
    )
