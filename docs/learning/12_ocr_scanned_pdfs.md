# 1️⃣2️⃣ OCR: reading scanned PDFs and images

> **Phase 2 · Lesson 4** · Code: [`rag/ocr.py`](../../rag/ocr.py), [`rag/loader.py`](../../rag/loader.py)

## Two kinds of PDF that look the same

Open two PDFs, and they can look identical on screen. Inside, they're completely different:

| "Real" PDF (digital) | Scanned PDF |
|---|---|
| Made by Word, a website, a billing system | Made by a scanner or a phone camera |
| Stores actual **letters**: `P` `r` `e` `a` `u` `t` `h` | Stores a **photo** of the page: millions of coloured dots |
| You can select and copy text | Selecting text does nothing |
| Our loader copies the text instantly ✅ | Our loader finds **nothing** ❌ |

Hospital discharge summaries, signed letters, old policy documents and ID cards are very often scanned.
In Phase 1, those pages were simply **skipped**.

## OCR = "look at the picture and type it out"

**OCR** stands for *Optical Character Recognition*. Like a person reading a photo of a letter
and typing it out, the computer looks at the dots and works out which letters they form.

Classic OCR programs (like Tesseract) must be installed separately.
But we already have something better on your computer: **Qwen3.5 can see images** (it's a *vision* model).
So we just show it the picture and say: *"Type out every word you see."*

```
 scanned PDF page
       │  ① take a "photo" of the page (150 DPI picture)
       ▼
   PNG image
       │  ② send it to Qwen with the instruction "transcribe exactly"
       ▼
   plain text  ──►  chunk  ──►  embed  ──►  store   (same as any other page)
```

## How does the loader know a page is scanned?

Simple test: it tries to copy the text first.
If a page gives **fewer than 30 real letters** (`OCR_MIN_TEXT_CHARS`), it's probably a picture → OCR it.

So a mixed PDF (some typed pages, some scanned) works automatically: typed pages are copied (fast),
and only the scanned pages are OCR'd (slow).

Image files (`.png`, `.jpg`, `.jpeg`, `.webp`) such as screenshots or photos of documents are always OCR'd.

OCR'd chunks get a label `ocr: True` in their metadata, so you can always tell their text was *read*, not copied.

## Things to know ⚠️

| Topic | What to expect |
|---|---|
| **Speed** | About 5–30 seconds per page. A 50-page scan can take 10+ minutes. Ingest it once, then it's saved. |
| **Accuracy** | Very good on clean scans and tables; weaker on blurry photos and messy handwriting. |
| **AI mistakes** | A vision AI can occasionally misread a number. For important numbers (amounts, IDs), check the original page. The citation tells you which page! |
| **Quality vs speed** | `OCR_DPI = 150` is a good balance. Try `200` for tiny text, `100` for speed. |

## Settings (in `rag/config.py`)

| Setting | Default | What it does |
|---|---|---|
| `OCR_ENABLED` | `True` | Switch OCR off to skip scanned pages (faster ingest) |
| `OCR_MODEL` | `qwen3.5:4b` | Any Ollama model with "vision" ability |
| `OCR_MIN_TEXT_CHARS` | `30` | Below this many letters, a page counts as "scanned" |
| `OCR_DPI` | `150` | Sharpness of the page photo |

## Read the code

- [`rag/ocr.py`](../../rag/ocr.py) → `image_to_text()`: the picture is sent as **base64** (a way to write binary data as plain text) inside the chat message's `images` field.
  The prompt says "copy word for word, don't describe", and `temperature=0` means no creativity.
- [`rag/loader.py`](../../rag/loader.py) → `_load_pdf()`: the "fewer than 30 letters?" test, and `_load_image()`.

## 🧪 Try it yourself

1. Take a photo of any printed page with your phone (or a screenshot of a document) and save it as a `.jpg` or `.png`.
2. Put it in `data\documents` and run:
   ```powershell
   python ingest.py
   ```
   Look for **"(1 read with OCR)"** in the output.
3. Ask the chat a question about something in the photo. Does it find it and cite the image?
4. Try a blurry photo. Where does OCR start making mistakes?

> 🎯 **Key lesson:** the AI can only search what it can *read*. OCR turns pictures of text into real text,
> so scanned documents become searchable like everything else.
