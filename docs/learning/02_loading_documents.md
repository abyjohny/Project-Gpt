# 2️⃣ Loading documents — getting the words out

## The idea

The AI can only read **plain text**. But your files are "boxes":

- A **PDF** stores letters at exact positions on a page, plus fonts, images, lines…
- A **Word file (.docx)** is actually a zip folder of XML files describing paragraphs and styles.
- A **.txt / .md / .py** file is already plain text 🎉

The **loader** opens each box and takes out only the words.

```
 📄 NHCX_spec.pdf  ──►  loader  ──►  "Pre-authorization is a request sent by..."
                                      + { source: "NHCX_spec.pdf", page: 12 }
```

## Metadata — remembering where text came from

Along with the text, we save a few **labels** called **metadata**:

| Label | Example | Why we need it |
|-------|---------|----------------|
| `source` | `NHCX_spec.pdf` | So the AI can say *which file* |
| `page` | `12` | So the AI can say *which page*, and you can check it yourself |
| `path` | `D:\...\NHCX_spec.pdf` | So we can update a file later without duplicates |
| `file_type` | `pdf` | Useful for filtering later ("only search PDFs") |

Without metadata, the AI could answer but never tell you **where** the answer came from.
With it, you get answers like: *"Send a cancel request with the preauth ID [NHCX_spec.pdf, page 12]"*.

## How each file type is read

Open [`rag/loader.py`](../../rag/loader.py):

| File type | Library | What we do |
|-----------|---------|-----------|
| `.pdf` | **PyMuPDF** (`fitz`) | One piece **per page**, so we know page numbers |
| `.docx` | **python-docx** | All paragraphs **and tables** (API docs are full of tables!) |
| `.txt .md .py .js …` | plain Python | Just read the file |

Tables in Word are turned into lines like:
```
Field | Type | Required
preauth_id | string | yes
```
so the AI can still understand the rows.

## ⚠️ Things that can go wrong

| Problem | What you'll see | Why |
|---------|-----------------|-----|
| **Scanned PDF** (a photo of paper) | "No text found. Skipped." | There are no letters, only an image. It needs **OCR** (reading text from images), which we'll add later. |
| Weird spacing / broken words | Chunks look messy | PDFs store letters by position, not as sentences. Our chunker cleans up most of this. |
| Headers/footers on every page | "Page 3 of 200 – Confidential" repeated | Harmless mostly; we'll filter these in Phase 2. |

> 💡 **Quick test for a PDF**: open it and try to **select text with your mouse**.
> If you can select it, we can read it. If it selects like a picture, it's scanned.

## 🧪 Try it yourself

After installing, run this in the project folder (put one PDF in `data/documents` first):

```powershell
.\.venv\Scripts\python.exe -c "from rag.loader import *; docs = load_file(find_files()[0]); print(len(docs), 'pages'); print(docs[0].metadata); print(docs[0].text[:500])"
```

Look at the output. That raw, slightly messy text is **exactly** what the rest of the pipeline works with.
