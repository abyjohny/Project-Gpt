"""
ocr.py - PHASE 2: Read text out of PICTURES (scanned PDFs, photos, screenshots).
===============================================================================

A normal PDF stores real letters:   "P", "r", "e", "a", "u", "t", "h"  -> easy to copy.
A SCANNED PDF stores a PHOTO of a page. To the computer it is just coloured dots.
There is nothing to copy - someone has to LOOK at the picture and type it out.

That "look and type" job is called OCR (Optical Character Recognition).

Classic OCR tools (like Tesseract) are separate programs you must install.
But Qwen3.5 has "eyes" (it is a vision model), so we simply show it the
picture and ask: "type out every word you see". No extra install needed.

    PDF page --(photograph at 150 DPI)--> PNG picture --(Qwen looks at it)--> text

Trade-offs to know:
    + handles messy scans, tables and handwriting surprisingly well
    - slow: roughly 5-30 seconds per page on your GPU
    - an AI CAN misread or (rarely) invent text - for critical numbers,
      double-check against the original page (the citation tells you which one!)

Learn more: docs/learning/12_ocr_scanned_pdfs.md
"""

from __future__ import annotations

import base64
import re

from rag import config, ollama_client

OCR_PROMPT = """Transcribe ALL the text in this image exactly as written.
Rules:
- Copy the text word for word. Do not summarise, translate, explain or describe the image.
- Keep the reading order and line breaks.
- For tables, write one row per line with " | " between cells.
- If the image contains no readable text, reply with exactly: [NO TEXT]"""


def image_to_text(image_bytes: bytes) -> str:
    """Show a picture to Qwen and get back the text it reads."""
    message = {
        "role": "user",
        "content": OCR_PROMPT,
        "images": [base64.b64encode(image_bytes).decode("ascii")],   # pictures travel as base64 text
    }
    text = "".join(ollama_client.chat(
        [message],
        model=config.OCR_MODEL,
        temperature=0.0,                          # no creativity - just copy
        stream=False,
        extra_options={"presence_penalty": 0},    # allow repeated words (tables repeat a lot!)
    ))
    text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text.strip())   # drop ``` fences if the model adds them
    return "" if "[NO TEXT]" in text else text.strip()


def pdf_page_to_text(page) -> str:
    """Photograph one PyMuPDF page and OCR it."""
    picture = page.get_pixmap(dpi=config.OCR_DPI)
    return image_to_text(picture.tobytes("png"))
