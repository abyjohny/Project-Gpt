"""
build_learning_pdf.py
Compiles all Project GPT learning documentation into a single, comprehensive,
beautifully typeset master PDF textbook.
"""

import os
import re
import sys
import glob
import subprocess
import tempfile
from pathlib import Path
import markdown
from pygments.formatters import HtmlFormatter
import pymupdf

WORKSPACE_ROOT = Path(r"d:\dev\personal\Project GPT")
DOCS_DIR = WORKSPACE_ROOT / "docs" / "learning"
PLAYGROUND_DIR = WORKSPACE_ROOT / "playground"
OUTPUT_HTML = WORKSPACE_ROOT / "docs" / "Project_GPT_Learning_Guide.html"
TEMP_PDF = WORKSPACE_ROOT / "docs" / "temp_guide.pdf"
FINAL_PDF = WORKSPACE_ROOT / "docs" / "Project_GPT_Learning_Guide.pdf"
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# 18 Chapters in exact logical sequence
CHAPTERS = [
    {
        "id": "chapter-00",
        "file": "00_setup_and_ollama.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "00",
        "title": "Setup & Ollama — Running an AI on Your Own Laptop",
        "summary": "What models and weights are, how Ollama serves models locally via HTTP, parameter sizing (4b vs 9b), and solving the 2-second IPv6 localhost latency bug.",
        "code": "rag/ollama_client.py"
    },
    {
        "id": "chapter-01",
        "file": "01_what_is_rag.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "01",
        "title": "What is RAG? — The Open-Book Exam",
        "summary": "Why pasting entire documents into prompts fails, the open-book exam mental model, and the two fundamental phases: Library Ingestion vs Real-Time Querying.",
        "code": "rag/config.py, ingest.py"
    },
    {
        "id": "chapter-02",
        "file": "02_loading_documents.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "02",
        "title": "Loading Documents — Getting the Words Out",
        "summary": "Extracting clean plain text from PDFs (PyMuPDF) and Word documents (python-docx), attaching page metadata, and handling encoding issues.",
        "code": "rag/loader.py"
    },
    {
        "id": "chapter-03",
        "file": "03_chunking.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "03",
        "title": "Chunking — Cutting Documents into Pieces",
        "summary": "Why LLMs need small passages, chunk size vs search precision, sentence-boundary splitting, and sliding window overlap to prevent split facts.",
        "code": "rag/chunker.py"
    },
    {
        "id": "chapter-04",
        "file": "04_embeddings.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "04",
        "title": "Embeddings — Turning Meaning into Numbers",
        "summary": "High-dimensional vector representations, how text becomes a coordinate in 768-dimensional space, and measuring cosine distance between ideas.",
        "code": "rag/embedder.py"
    },
    {
        "id": "chapter-05",
        "file": "05_vector_database.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "05",
        "title": "The Vector Database — Storing and Finding Meaning",
        "summary": "ChromaDB internals, persistent collections, metadata filtering, Euclidean distance vs Cosine distance, and why vector search is fast.",
        "code": "rag/vector_store.py"
    },
    {
        "id": "chapter-06",
        "file": "06_retrieval.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "06",
        "title": "Retrieval — Finding the Right Pages",
        "summary": "Transforming user questions into vectors, Top-K nearest neighbour search, similarity thresholds, and formatting chunks into context passages.",
        "code": "rag/retriever.py"
    },
    {
        "id": "chapter-07",
        "file": "07_generation.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "07",
        "title": "Generation — The AI Writes the Answer",
        "summary": "Prompt engineering for grounded answers, strict anti-hallucination system instructions, source citation tokens ([1], [2]), and streaming responses.",
        "code": "rag/generator.py"
    },
    {
        "id": "chapter-08",
        "file": "08_run_and_experiment.md",
        "phase": "Phase 1: RAG From Scratch",
        "number": "08",
        "title": "Run It & Experiment — Hands-on Testing & Debugging",
        "summary": "End-to-end execution of ingest.py and chat.py, debugging retrieval failures, understanding hallucination scenarios, and performance tuning.",
        "code": "ingest.py, chat.py"
    },
    {
        "id": "chapter-09",
        "file": "09_keyword_search.md",
        "phase": "Phase 2: Better Search",
        "number": "09",
        "title": "Keyword Search (BM25) — Smart Exact Matching",
        "summary": "Where vector embeddings fail (exact part numbers, error codes, rare names), BM25 term frequency / inverse document frequency, and building an inverted index.",
        "code": "rag/keyword_search.py"
    },
    {
        "id": "chapter-10",
        "file": "10_hybrid_search.md",
        "phase": "Phase 2: Better Search",
        "number": "10",
        "title": "Hybrid Search — Combining Meaning & Keywords with RRF",
        "summary": "Reciprocal Rank Fusion (RRF) algorithm: merging BM25 and vector search rankings without scale normalisation problems, achieving best-of-both-worlds retrieval.",
        "code": "rag/retriever.py"
    },
    {
        "id": "chapter-11",
        "file": "11_reranking.md",
        "phase": "Phase 2: Better Search",
        "number": "11",
        "title": "Re-ranking — The Careful Cross-Encoder Judge",
        "summary": "Bi-encoders vs Cross-encoders (BGE-Reranker-v2-m3), evaluating query and passage simultaneously with full cross-attention, shortlisting top 20 and selecting top 5.",
        "code": "rag/reranker.py"
    },
    {
        "id": "chapter-12",
        "file": "12_ocr_scanned_pdfs.md",
        "phase": "Phase 2: Better Search",
        "number": "12",
        "title": "OCR — Reading Scanned PDFs and Images",
        "summary": "Handling image-only scanned PDFs, rendering pages to pixmaps with PyMuPDF, running local OCR engines (Tesseract / EasyOCR), and embedding image text.",
        "code": "rag/loader.py"
    },
    {
        "id": "chapter-13",
        "file": "13_measuring_quality.md",
        "phase": "Phase 2: Better Search",
        "number": "13",
        "title": "Measuring Quality — Hit@K, MRR & Systematic Evaluation",
        "summary": "How to score your RAG pipeline scientifically: Mean Reciprocal Rank (MRR), Hit@1, Hit@5, creating ground-truth evaluation datasets, and benchmarking improvements.",
        "code": "evaluate.py"
    },
    {
        "id": "chapter-14",
        "file": "14_structure_aware_chunking.md",
        "phase": "Phase 3: Structure & Memory",
        "number": "14",
        "title": "Structure-Aware Chunking — Following Document Shape",
        "summary": "Why character-count chunking cuts tables and headers in half, parsing Markdown headings, keeping table rows intact with repeated headers, and breadcrumb locations.",
        "code": "rag/chunker.py, rag/loader.py"
    },
    {
        "id": "chapter-15",
        "file": "15_conversation_memory.md",
        "phase": "Phase 3: Structure & Memory",
        "number": "15",
        "title": "Conversation Memory — Multi-Turn Dialogues",
        "summary": "Solving pronoun follow-ups ('and what about the second one?'), query rewriting with Qwen, chat history sliding buffers, and session state persistence.",
        "code": "rag/memory.py, chat.py"
    },
    {
        "id": "chapter-16",
        "file": "16_web_ui.md",
        "phase": "Phase 4: Web UI & Agents",
        "number": "16",
        "title": "The Web UI — Streaming, Sources, Sessions & Cowork",
        "summary": "FastAPI backend, Server-Sent Events (SSE) token streaming, thread queues for non-blocking inference, clickable source citations, DOMPurify security, and live GPU stats.",
        "code": "server.py, web/app.js, rag/sessions.py"
    },
    {
        "id": "chapter-17",
        "file": "17_agent_mode.md",
        "phase": "Phase 4: Web UI & Agents",
        "number": "17",
        "title": "Agent Mode — Tool Use, Sandboxing & Step Execution",
        "summary": "From single-shot RAG to multi-step autonomous tool execution, Ollama JSON function calling schema, sandboxed filesystem tools (grep, find, read_file), and loop safeguards.",
        "code": "rag/agent.py, rag/agent_tools.py"
    },
]

def render_markdown(text: str) -> str:
    # Rewrite relative .md links to internal chapter anchors
    for ch in CHAPTERS:
        filename = ch["file"]
        cid = ch["id"]
        text = re.sub(r'\[([^\]]+)\]\(' + re.escape(filename) + r'\)', rf'[\1](#{cid})', text)
    
    # Rewrite README.md links
    text = re.sub(r'\[([^\]]+)\]\(README\.md\)', r'[\1](#curriculum-overview)', text)
    text = re.sub(r'\[([^\]]+)\]\(docs/learning/README\.md\)', r'[\1](#curriculum-overview)', text)

    # Rewrite code links to bold styled code tags instead of relative dead links
    text = re.sub(r'\[`?([^\]]+\.py)`?\]\(\.\./\.\./[^\)]+\)', r'<code>\1</code>', text)
    text = re.sub(r'\[`?([^\]]+\.py)`?\]\([^\)]+\.py\)', r'<code>\1</code>', text)

    md = markdown.Markdown(
        extensions=['fenced_code', 'tables', 'codehilite', 'attr_list'],
        extension_configs={
            'codehilite': {
                'guess_lang': False,
                'use_pygments': True,
                'css_class': 'codehilite'
            }
        }
    )
    html = md.convert(text)

    # Transform blockquotes into rich callout cards
    def replace_blockquote(match):
        inner = match.group(1).strip()
        if "💡" in inner:
            return f'<div class="callout callout-tip"><div class="callout-icon">💡</div><div class="callout-content">{inner}</div></div>'
        elif "⚠️" in inner:
            return f'<div class="callout callout-warning"><div class="callout-icon">⚠️</div><div class="callout-content">{inner}</div></div>'
        elif "🧪" in inner:
            return f'<div class="callout callout-lab"><div class="callout-icon">🧪</div><div class="callout-content">{inner}</div></div>'
        elif "👉" in inner or "Code:" in inner or "Related Code:" in inner:
            return f'<div class="callout callout-code"><div class="callout-icon">📂</div><div class="callout-content">{inner}</div></div>'
        else:
            return f'<div class="callout callout-note"><div class="callout-icon">ℹ️</div><div class="callout-content">{inner}</div></div>'

    html = re.sub(r'<blockquote>([\s\S]*?)</blockquote>', replace_blockquote, html)
    return html

def build_html(page_map: dict = None):
    page_map = page_map or {}

    # Read Root README architecture section
    root_readme_path = WORKSPACE_ROOT / "README.md"
    root_readme_content = ""
    if root_readme_path.exists():
        with open(root_readme_path, "r", encoding="utf-8") as f:
            root_readme_content = f.read()

    # Read learning README
    learning_readme_path = DOCS_DIR / "README.md"
    learning_readme_content = ""
    if learning_readme_path.exists():
        with open(learning_readme_path, "r", encoding="utf-8") as f:
            learning_readme_content = f.read()

    # Extract Mini Dictionary from learning README
    mini_dict_match = re.search(r'(## 📖 Mini dictionary[\s\S]+)', learning_readme_content)
    mini_dict_content = mini_dict_match.group(1) if mini_dict_match else ""

    # Playground Files Content for Appendix
    playground_files = sorted(glob.glob(str(PLAYGROUND_DIR / "*.py")))
    playground_sections = []
    for pf in playground_files:
        p_name = os.path.basename(pf)
        with open(pf, "r", encoding="utf-8") as f:
            p_code = f.read()
        playground_sections.append((p_name, p_code))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Project GPT — Complete Learning Guide</title>
<style>
/* ==========================================================================
   PAGE SETUP & PRINT MEDIA STYLES
   ========================================================================== */
@page {{
    size: A4 portrait;
    margin: 18mm 16mm 20mm 16mm;
}}

* {{
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    line-height: 1.62;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}}

/* ==========================================================================
   COVER PAGE
   ========================================================================== */
.cover-page {{
    page-break-after: always;
    min-height: 98vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 40px 30px;
    background: linear-gradient(135deg, #090d16 0%, #111827 45%, #1e1b4b 100%);
    color: #ffffff;
    border-radius: 12px;
    position: relative;
    box-shadow: inset 0 0 100px rgba(0,0,0,0.5);
}}

.cover-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    padding-bottom: 20px;
}}

.cover-badge {{
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid #3b82f6;
    color: #60a5fa;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

.cover-edition {{
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}

.cover-hero {{
    margin: auto 0;
    padding: 20px 0;
}}

.cover-icon {{
    font-size: 54px;
    margin-bottom: 20px;
}}

.cover-title {{
    font-size: 42px;
    font-weight: 900;
    line-height: 1.15;
    letter-spacing: -0.5px;
    margin: 0 0 16px 0;
    background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.cover-subtitle {{
    font-size: 18px;
    font-weight: 400;
    color: #94a3b8;
    line-height: 1.5;
    max-width: 650px;
    margin: 0 0 32px 0;
}}

.cover-features-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    max-width: 650px;
}}

.cover-feature-item {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #e2e8f0;
}}

.cover-feature-item strong {{
    color: #38bdf8;
    display: block;
    margin-bottom: 2px;
    font-size: 12.5px;
}}

.cover-footer {{
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    padding-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    color: #64748b;
    font-size: 11px;
}}

.cover-meta strong {{
    color: #cbd5e1;
}}

/* ==========================================================================
   PAGE BREAK & SECTION STYLES
   ========================================================================== */
.page-break {{
    page-break-before: always;
    break-before: page;
}}

.chapter {{
    page-break-before: always;
    break-before: page;
    padding-top: 10px;
}}

.avoid-break {{
    page-break-inside: avoid;
    break-inside: avoid;
}}

/* ==========================================================================
   TYPOGRAPHY
   ========================================================================== */
h1, h2, h3, h4, h5, h6 {{
    color: #0f172a;
    font-weight: 700;
    line-height: 1.3;
    page-break-after: avoid;
    break-after: avoid;
}}

h1 {{
    font-size: 24px;
    margin-top: 0;
    margin-bottom: 14px;
    letter-spacing: -0.3px;
}}

h2 {{
    font-size: 17px;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 28px;
    margin-bottom: 12px;
}}

h3 {{
    font-size: 14.5px;
    margin-top: 20px;
    margin-bottom: 8px;
    color: #1e293b;
}}

p {{
    margin: 0 0 12px 0;
}}

ul, ol {{
    margin: 0 0 14px 0;
    padding-left: 24px;
}}

li {{
    margin-bottom: 4px;
}}

a {{
    color: #2563eb;
    text-decoration: none;
}}

a:hover {{
    text-decoration: underline;
}}

hr {{
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 24px 0;
}}

/* ==========================================================================
   CHAPTER HEADER BANNER
   ========================================================================== */
.chapter-header {{
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-left: 6px solid #2563eb;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 24px;
    page-break-inside: avoid;
    break-inside: avoid;
}}

.chapter-phase-badge {{
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    font-size: 10.5px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 8px;
}}

.chapter-number-badge {{
    display: inline-block;
    background: #0f172a;
    color: #ffffff;
    font-size: 10.5px;
    font-weight: 800;
    letter-spacing: 0.8px;
    padding: 3px 8px;
    border-radius: 4px;
    margin-left: 6px;
}}

.chapter-title {{
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    margin: 6px 0 10px 0;
    line-height: 1.25;
}}

.chapter-code-bar {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11.5px;
    color: #475569;
    background: #ffffff;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    width: fit-content;
    margin-top: 6px;
}}

.chapter-code-bar code {{
    color: #0284c7;
    font-weight: 700;
}}

/* ==========================================================================
   TABLE OF CONTENTS (COMPACT 2-PAGE LAYOUT)
   ========================================================================== */
.toc-container {{
    page-break-before: always;
    page-break-after: always;
    padding: 0;
}}

.toc-title {{
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
    border-bottom: 2px solid #0f172a;
    padding-bottom: 6px;
    margin-bottom: 12px;
}}

.toc-phase-block {{
    margin-bottom: 12px;
    page-break-inside: avoid;
    break-inside: avoid;
}}

.toc-phase-title {{
    font-size: 12.5px;
    font-weight: 800;
    color: #1e3a8a;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    background: #eff6ff;
    padding: 4px 10px;
    border-radius: 4px;
    border-left: 3.5px solid #3b82f6;
    margin-bottom: 6px;
}}

.toc-list {{
    list-style: none;
    padding: 0;
    margin: 0;
}}

.toc-item {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 4px 6px;
    border-bottom: 1px dotted #e2e8f0;
    font-size: 12px;
}}

.toc-item-left {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    max-width: 72%;
}}

.toc-num {{
    font-weight: 800;
    color: #0284c7;
    font-family: monospace;
    font-size: 12px;
    width: 22px;
    flex-shrink: 0;
}}

.toc-link {{
    color: #0f172a;
    font-weight: 600;
    text-decoration: none;
}}

.toc-desc {{
    font-size: 10.5px;
    color: #64748b;
    margin-top: 1px;
    line-height: 1.35;
}}

.toc-item-right {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}}

.toc-code {{
    font-family: "Cascadia Code", Consolas, monospace;
    font-size: 10.5px;
    color: #475569;
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 4px;
}}

.toc-page-badge {{
    background: #e0f2fe;
    color: #0369a1;
    font-weight: 700;
    font-size: 10.5px;
    padding: 1px 6px;
    border-radius: 4px;
    border: 1px solid #bae6fd;
    min-width: 32px;
    text-align: center;
}}

/* ==========================================================================
   CALLOUT BOXES
   ========================================================================== */
.callout {{
    display: flex;
    gap: 14px;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 16px 0;
    font-size: 12.5px;
    line-height: 1.55;
    page-break-inside: avoid;
    break-inside: avoid;
    border-left: 4px solid;
}}

.callout-icon {{
    font-size: 18px;
    flex-shrink: 0;
    margin-top: 1px;
}}

.callout-content {{
    flex-grow: 1;
}}

.callout-content p:last-child {{
    margin-bottom: 0;
}}

.callout-tip {{
    background: #f0f9ff;
    border-color: #0284c7;
    color: #0c4a6e;
}}

.callout-tip strong {{
    color: #0369a1;
}}

.callout-warning {{
    background: #fffbeb;
    border-color: #f59e0b;
    color: #78350f;
}}

.callout-warning strong {{
    color: #b45309;
}}

.callout-lab {{
    background: #ecfdf5;
    border-color: #10b981;
    color: #064e3b;
}}

.callout-lab strong {{
    color: #047857;
}}

.callout-code {{
    background: #f8fafc;
    border-color: #64748b;
    color: #1e293b;
}}

.callout-note {{
    background: #f8fafc;
    border-color: #3b82f6;
    color: #1e293b;
}}

/* ==========================================================================
   CODE & SYNTAX HIGHLIGHTING
   ========================================================================== */
pre, .codehilite {{
    background: #0f172a !important;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: "Cascadia Code", "Fira Code", Consolas, "Courier New", monospace !important;
    font-size: 11.5px !important;
    line-height: 1.48;
    color: #f8fafc !important;
    overflow-wrap: break-word;
    white-space: pre-wrap !important;
    word-break: break-word;
    page-break-inside: avoid;
    break-inside: avoid;
    margin: 14px 0;
}}

pre code, .codehilite code {{
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    font-family: inherit !important;
    font-size: inherit !important;
    color: inherit !important;
}}

code {{
    font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
    font-size: 11.5px;
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
}}

/* Pygments Token Overrides for Dark Box */
.codehilite .k, .codehilite .kn, .codehilite .kd, .codehilite .kc {{ color: #38bdf8 !important; font-weight: bold; }}
.codehilite .s, .codehilite .s1, .codehilite .s2, .codehilite .sa {{ color: #86efac !important; }}
.codehilite .c, .codehilite .c1, .codehilite .ch, .codehilite .cm {{ color: #94a3b8 !important; font-style: italic; }}
.codehilite .nf, .codehilite .nc, .codehilite .fm {{ color: #818cf8 !important; font-weight: bold; }}
.codehilite .nb, .codehilite .bp {{ color: #f472b6 !important; }}
.codehilite .m, .codehilite .mi, .codehilite .mf {{ color: #fde047 !important; }}
.codehilite .o, .codehilite .ow {{ color: #cbd5e1 !important; }}
.codehilite .nt {{ color: #f87171 !important; }}
.codehilite .na {{ color: #38bdf8 !important; }}

/* ==========================================================================
   TABLES
   ========================================================================== */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 12px;
    page-break-inside: avoid;
    break-inside: avoid;
    border: 1px solid #cbd5e1;
}}

th {{
    background: #0f172a;
    color: #ffffff;
    font-weight: 700;
    text-align: left;
    padding: 8px 12px;
    border: 1px solid #334155;
    font-size: 11.5px;
    letter-spacing: 0.4px;
}}

td {{
    padding: 8px 12px;
    border: 1px solid #e2e8f0;
    vertical-align: top;
}}

tr:nth-child(even) td {{
    background-color: #f8fafc;
}}

/* ==========================================================================
   APPENDIX & OVERVIEW CARDS
   ========================================================================== */
.overview-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
    page-break-inside: avoid;
    break-inside: avoid;
}}

.overview-card-title {{
    font-size: 15px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
}}
</style>
</head>
<body>

<!-- ==========================================================================
     COVER PAGE
     ========================================================================== -->
<div class="cover-page">
    <div class="cover-header">
        <span class="cover-badge">PROJECT GPT · ENGINEERING MANUAL</span>
        <span class="cover-edition">OCTOBER 2026 EDITION</span>
    </div>

    <div class="cover-hero">
        <div class="cover-icon">🤖📚</div>
        <h1 class="cover-title">PROJECT GPT<br>THE COMPLETE LEARNING GUIDE</h1>
        <p class="cover-subtitle">
            A comprehensive, architectural textbook on building your own local ChatGPT from raw Python scratch:
            RAG pipelines, vector math, hybrid BM25 search, cross-encoder reranking, document structure parsing,
            conversational memory, and autonomous agent loops.
        </p>

        <div class="cover-features-grid">
            <div class="cover-feature-item">
                <strong>Phase 1 · Core RAG From Scratch</strong>
                Ollama, Chunks, Embeddings, ChromaDB, Top-K, Generator
            </div>
            <div class="cover-feature-item">
                <strong>Phase 2 · Better Search Engines</strong>
                BM25 Keywords, RRF Hybrid Fusion, Cross-Encoders, OCR, MRR
            </div>
            <div class="cover-feature-item">
                <strong>Phase 3 · Document Structure & Memory</strong>
                Heading/Table chunking, Query rewriting, Follow-up memory
            </div>
            <div class="cover-feature-item">
                <strong>Phase 4 · Full-Stack App & Agents</strong>
                SSE token streaming, Source verification chips, Multi-step Tools
            </div>
        </div>
    </div>

    <div class="cover-footer">
        <div class="cover-meta">
            <div>Author & Project: <strong>Project GPT Core Team</strong></div>
            <div>Runtime Environment: <strong>100% Local & Private (Ollama + Python)</strong></div>
        </div>
        <div>18 Comprehensive Chapters · 5 Standalone Labs · Complete Reference Glossary</div>
    </div>
</div>

<!-- ==========================================================================
     TABLE OF CONTENTS
     ========================================================================== -->
<div class="toc-container">
    <div class="toc-title">Table of Contents</div>

    <div class="toc-phase-block">
        <div class="toc-phase-title">System Architecture & Foundations</div>
        <ul class="toc-list">
            <li class="toc-item">
                <div class="toc-item-left">
                    <span class="toc-num">✦</span>
                    <a class="toc-link" href="#system-architecture">Project GPT Architecture & End-to-End Pipeline Blueprint</a>
                </div>
                <div class="toc-item-right">
                    <span class="toc-code">README.md</span>
                    <span class="toc-page-badge">p. {page_map.get('arch', 4)}</span>
                </div>
            </li>
            <li class="toc-item">
                <div class="toc-item-left">
                    <span class="toc-num">✦</span>
                    <a class="toc-link" href="#curriculum-overview">Curriculum Roadmap & Study Methodology</a>
                </div>
                <div class="toc-item-right">
                    <span class="toc-code">docs/learning/README.md</span>
                    <span class="toc-page-badge">p. {page_map.get('roadmap', 6)}</span>
                </div>
            </li>
        </ul>
    </div>
"""

    # Group chapters by Phase
    current_phase = ""
    for ch in CHAPTERS:
        if ch["phase"] != current_phase:
            if current_phase != "":
                html += "</ul></div>\n"
            current_phase = ch["phase"]
            html += f"""
    <div class="toc-phase-block">
        <div class="toc-phase-title">{current_phase}</div>
        <ul class="toc-list">
"""
        ch_p = page_map.get(ch["number"], 8)
        html += f"""
            <li class="toc-item">
                <div class="toc-item-left">
                    <span class="toc-num">{ch['number']}</span>
                    <div>
                        <a class="toc-link" href="#{ch['id']}">{ch['title']}</a>
                        <div class="toc-desc">{ch['summary']}</div>
                    </div>
                </div>
                <div class="toc-item-right">
                    <span class="toc-code">{ch['code']}</span>
                    <span class="toc-page-badge">p. {ch_p}</span>
                </div>
            </li>
"""

    dict_p = page_map.get('dict', 67)
    lab_p = page_map.get('lab', 68)

    html += f"""
        </ul>
    </div>

    <div class="toc-phase-block">
        <div class="toc-phase-title">Reference & Playground Labs</div>
        <ul class="toc-list">
            <li class="toc-item">
                <div class="toc-item-left">
                    <span class="toc-num">📖</span>
                    <div>
                        <a class="toc-link" href="#mini-dictionary">Comprehensive AI & RAG Mini-Dictionary (Glossary)</a>
                        <div class="toc-desc">Every essential term defined in plain English: Vectors, Tokens, RRF, Top-K, Hallucination...</div>
                    </div>
                </div>
                <div class="toc-item-right">
                    <span class="toc-code">Glossary</span>
                    <span class="toc-page-badge">p. {dict_p}</span>
                </div>
            </li>
            <li class="toc-item">
                <div class="toc-item-left">
                    <span class="toc-num">🧪</span>
                    <div>
                        <a class="toc-link" href="#playground-appendix">Appendix: Standalone Playground Hands-On Lab</a>
                        <div class="toc-desc">Standalone companion scripts to see chunking, embeddings, NumPy vector DBs, and reranking.</div>
                    </div>
                </div>
                <div class="toc-item-right">
                    <span class="toc-code">playground/*.py</span>
                    <span class="toc-page-badge">p. {lab_p}</span>
                </div>
            </li>
        </ul>
    </div>
</div>
"""

    # System Architecture Section
    html += """
<div class="page-break" id="system-architecture">
    <h1>🏛️ Project GPT — System Architecture & Data Flow</h1>
    <div class="overview-card">
        <div class="overview-card-title">The Big Picture: Local AI Assistant Architecture</div>
        <p>
            Project GPT is built from scratch without massive abstractions (like LangChain or LlamaIndex)
            so that every mathematical and software layer is visible and tunable.
            The entire stack runs <strong>100% locally on your laptop</strong> without sending data outside.
        </p>
    </div>
"""
    clean_root_readme = re.sub(r'# 🤖 Project GPT.*?\n', '', root_readme_content)
    clean_root_readme = re.sub(r'## 📚 Learn[\s\S]*?## Project layout', '## Project Layout', clean_root_readme)
    html += render_markdown(clean_root_readme)
    html += "</div>\n"

    # Curriculum Roadmap Section
    html += """
<div class="page-break" id="curriculum-overview">
    <h1>🗺️ Curriculum Roadmap & How to Learn</h1>
    <div class="callout callout-tip">
        <div class="callout-icon">💡</div>
        <div class="callout-content">
            <strong>Recommended Study Workflow:</strong>
            <ol style="margin-top: 6px; padding-left: 20px;">
                <li>Read the note carefully to understand the core mental model.</li>
                <li>Open the indicated code file in your editor and inspect the comments.</li>
                <li>Run the "🧪 Try it yourself" experiments in terminal or in the <code>playground/</code> folder.</li>
                <li>Progress to the next concept only after seeing it work with your own eyes.</li>
            </ol>
        </div>
    </div>
"""
    clean_learn_readme = re.sub(r'# 📚 Learning Path.*?\n', '', learning_readme_content)
    clean_learn_readme = re.sub(r'## 📖 Mini dictionary[\s\S]+$', '', clean_learn_readme)
    html += render_markdown(clean_learn_readme)
    html += "</div>\n"

    # Main Chapters (00 through 17)
    for ch in CHAPTERS:
        file_path = DOCS_DIR / ch["file"]
        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Strip existing H1 line
        content_lines = content.splitlines()
        clean_lines = []
        for line in content_lines:
            if line.startswith("# ") and len(clean_lines) == 0:
                continue
            clean_lines.append(line)
        content_body = "\n".join(clean_lines)

        rendered_body = render_markdown(content_body)

        html += f"""
<div class="chapter" id="{ch['id']}">
    <div class="chapter-header">
        <div>
            <span class="chapter-phase-badge">{ch['phase']}</span>
            <span class="chapter-number-badge">CHAPTER {ch['number']}</span>
        </div>
        <div class="chapter-title">{ch['title']}</div>
        <div class="chapter-code-bar">
            <span>📂 Primary Implementation:</span>
            <code>{ch['code']}</code>
        </div>
    </div>

    <div class="chapter-body">
        {rendered_body}
    </div>
</div>
"""

    # Mini Dictionary Section
    html += f"""
<div class="page-break" id="mini-dictionary">
    <h1>📖 AI & RAG Mini-Dictionary — Core Terminology</h1>
    <p>Keep this glossary handy. Whenever a technical term causes confusion, review its plain-English definition here.</p>
    {render_markdown(mini_dict_content)}
</div>
"""

    # Appendix: Playground Lab
    html += """
<div class="page-break" id="playground-appendix">
    <h1>🧪 Appendix: Playground Hands-On Lab</h1>
    <p>
        These 5 standalone Python scripts let you observe each core mechanic in isolation without
        spinning up the entire application server. Run them directly in your terminal to develop intuition.
    </p>
"""
    for p_name, p_code in playground_sections:
        html += f"""
    <div class="overview-card" style="margin-top: 24px;">
        <div class="overview-card-title">🔬 Lab Script: <code>playground/{p_name}</code></div>
        <pre><code class="language-python">{p_code}</code></pre>
    </div>
"""
    html += "</div>\n"

    html += """
</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated master HTML at: {OUTPUT_HTML} ({len(html)} bytes)")

def convert_html_to_pdf_with_edge():
    temp_profile = tempfile.mkdtemp(prefix="edge_pdf_")
    args = [
        EDGE_EXE,
        "--headless=new",
        f"--user-data-dir={temp_profile}",
        "--no-pdf-header-footer",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={TEMP_PDF}",
        str(OUTPUT_HTML.resolve().as_uri())
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if not TEMP_PDF.exists():
        raise RuntimeError(f"Edge PDF conversion failed! {res.stderr}")

def detect_pages_from_pdf():
    doc = pymupdf.open(TEMP_PDF)
    pages = {}
    for p_idx in range(len(doc)):
        p_num = p_idx + 1
        txt = doc[p_idx].get_text()
        
        for ch in CHAPTERS:
            num = ch['number']
            if f"CHAPTER {num}" in txt and num not in pages:
                pages[num] = p_num

        if p_num > 3:
            if 'System Architecture & Data Flow' in txt and 'arch' not in pages:
                pages['arch'] = p_num
            if 'Curriculum Roadmap & How to Learn' in txt and 'roadmap' not in pages:
                pages['roadmap'] = p_num
            if 'AI & RAG Mini-Dictionary' in txt and 'dict' not in pages:
                pages['dict'] = p_num
            if 'Appendix: Playground Hands-On Lab' in txt and 'lab' not in pages:
                pages['lab'] = p_num
                
    doc.close()
    return pages

def post_process_pdf_with_mupdf(page_map: dict):
    print(f"Post-processing PDF with PyMuPDF...")
    doc = pymupdf.open(TEMP_PDF)
    total_pages = len(doc)
    print(f"Total pages rendered: {total_pages}")

    # Step 1: Build Native PDF Bookmarks (Table of Contents Tree)
    toc = []
    toc.append([1, "Cover Page", 1])
    toc.append([1, "Table of Contents", 2])
    toc.append([1, "System Architecture & Flow", page_map.get('arch', 4)])
    toc.append([1, "Curriculum Roadmap & Study Guide", page_map.get('roadmap', 6)])
    
    current_phase = ""
    for ch in CHAPTERS:
        if ch["phase"] != current_phase:
            current_phase = ch["phase"]
            phase_page = page_map.get(ch["number"], 1)
            toc.append([1, current_phase, phase_page])
        
        ch_page = page_map.get(ch["number"], 1)
        toc.append([2, f"{ch['number']} · {ch['title']}", ch_page])
        
    if "dict" in page_map:
        toc.append([1, "AI & RAG Mini-Dictionary", page_map["dict"]])
    if "lab" in page_map:
        toc.append([1, "Appendix: Playground Hands-On Lab", page_map["lab"]])

    doc.set_toc(toc)
    print(f"Injected {len(toc)} interactive bookmarks into PDF outline.")

    # Step 2: Insert Running Headers and Footers
    # Skip Cover Page (Page 1) and Table of Contents (Pages 2-3)
    for i in range(1, total_pages):
        page = doc[i]
        page_num = i + 1
        rect = page.rect
        
        header_text = "Project GPT — The Complete Learning Guide"
        header_font_size = 8.5
        header_y = 32
        
        page.draw_line(
            pymupdf.Point(45, header_y + 10),
            pymupdf.Point(rect.width - 45, header_y + 10),
            color=(0.85, 0.88, 0.92),
            width=0.5
        )
        page.insert_text(
            pymupdf.Point(45, header_y),
            header_text,
            fontsize=header_font_size,
            color=(0.4, 0.45, 0.55),
            fontname="helv"
        )
        
        footer_text = f"Page {page_num} of {total_pages}"
        footer_font_size = 8.5
        footer_y = rect.height - 30
        
        page.draw_line(
            pymupdf.Point(45, footer_y - 12),
            pymupdf.Point(rect.width - 45, footer_y - 12),
            color=(0.85, 0.88, 0.92),
            width=0.5
        )
        
        text_width = pymupdf.get_text_length(footer_text, fontname="helv", fontsize=footer_font_size)
        footer_x = (rect.width - text_width) / 2
        page.insert_text(
            pymupdf.Point(footer_x, footer_y),
            footer_text,
            fontsize=footer_font_size,
            color=(0.4, 0.45, 0.55),
            fontname="helv"
        )

    # Save finalized PDF
    doc.save(FINAL_PDF, garbage=4, deflate=True)
    doc.close()
    
    if TEMP_PDF.exists():
        TEMP_PDF.unlink()

    print(f"SUCCESS! Master PDF created at: {FINAL_PDF}")
    print(f"File size: {FINAL_PDF.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Total Pages: {total_pages}")

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== PASS 1: Generate initial layout & map page numbers ===")
    build_html()
    convert_html_to_pdf_with_edge()
    page_map = detect_pages_from_pdf()
    print("Detected page mapping:")
    for k, v in sorted(page_map.items()):
        print(f"  {k} -> Page {v}")

    print("\n=== PASS 2: Re-generate HTML with exact page numbers & finalise PDF ===")
    build_html(page_map)
    convert_html_to_pdf_with_edge()
    final_page_map = detect_pages_from_pdf()
    post_process_pdf_with_mupdf(final_page_map)

if __name__ == "__main__":
    main()
