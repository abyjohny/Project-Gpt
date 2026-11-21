"""
config.py - ALL the knobs of our RAG system live here.
=====================================================

Think of this file as the "settings page" of your AI.
Every other file reads its settings from here, so when you want to
experiment (bigger chunks? a different model?), you only change ONE place.

Learn more: docs/learning/01_what_is_rag.md
"""

from pathlib import Path

# ----------------------------------------------------------------
# 1. FOLDERS
# ----------------------------------------------------------------
# Path(__file__) = this file. .parent = the "rag" folder. .parent again = project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"   # <- put your PDFs/DOCX/TXT here
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"   # <- the "library index" is saved here

# ----------------------------------------------------------------
# 2. OLLAMA (the program that runs AI models on your computer)
# ----------------------------------------------------------------
# Ollama runs a tiny web server on your own machine at this address.
# Nothing goes to the internet - 127.0.0.1 means "this computer".
#
# Why 127.0.0.1 and not "localhost"? On Windows, "localhost" first tries
# the newer IPv6 address (::1). Ollama only listens on the older IPv4
# address, so EVERY request silently waited ~2 seconds before falling back.
# Writing the IPv4 address directly removes that hidden delay.
OLLAMA_URL = "http://127.0.0.1:11434"

# The "brain" that writes answers. 4b = 4 billion parameters.
# It fits comfortably inside your 6 GB RTX 3050.
CHAT_MODEL = "qwen3.5:4b"

# The "translator" that turns text into meaning-numbers (embeddings).
# It is a separate, much smaller model than the chat model.
EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text works best when we tell it WHAT the text is:
# a document being stored, or a question being asked.
# (Other embedding models may not need these - set both to "" then.)
EMBED_DOCUMENT_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "

# How many chunks we send to the embedding model in one request.
# Bigger = faster ingestion, but uses more memory.
EMBED_BATCH_SIZE = 32

# ----------------------------------------------------------------
# 3. CHUNKING (cutting documents into small pieces)
# ----------------------------------------------------------------
# Sizes are in CHARACTERS (letters). Roughly 4 characters = 1 token.
# 1000 characters is about 2-3 paragraphs.
CHUNK_SIZE = 1000

# How many characters neighbouring chunks share, so a sentence cut
# at the border still appears whole in at least one chunk.
CHUNK_OVERLAP = 150

# Chunks shorter than this are usually junk (page numbers, headers).
MIN_CHUNK_LENGTH = 50

# ----------------------------------------------------------------
# 3b. SMART CHUNKING  (Phase 3 - see docs/learning/14_structure_aware_chunking.md)
# ----------------------------------------------------------------
# How we cut documents into chunks:
#   "recursive" -> Phase 1: cut by size at paragraph / sentence borders
#   "structure" -> Phase 3: FIRST follow the document's own structure
#                  (chapters, headings, tables), THEN cut by size inside each section.
#                  Every chunk also gets its "address" written on top, e.g.
#                  "Section 4: Workflow Codes > 4.1 Provider-Side Workflow Codes"
# After changing this, rebuild:  python ingest.py --reset
CHUNKING_STRATEGY = "structure"

# MEASURED (Phase 3 experiment, 80 exam questions, hybrid + re-rank):
#   old loader + recursive 1000 ........ Hit@1 90.0%  MRR 0.930
#   structure 1000, defaults v1 ........ Hit@1 81.3%  MRR 0.858   <- first try was WORSE
#   structure 1500 + settings below .... Hit@1 91.3%  MRR 0.931   <- the winner
# Details: docs/learning/14_structure_aware_chunking.md

# Max chunk size (characters) for STRUCTURED documents (Word, PDF with headings).
# Bigger than CHUNK_SIZE because a section + its address + a few table rows
# needs more room than a plain paragraph. (CHUNK_SIZE is still used for .txt/code.)
SECTION_CHUNK_SIZE = 1500

# Small neighbouring sections (e.g. 5.6.1, 5.6.2, 5.6.3) are glued into one chunk
# while the result stays under this size. 0 = never glue sections together.
SECTION_MERGE_MAX = 1500

# The "address" written on top of every chunk:
#   "full" -> "Location: Section 4: Workflow Codes > 4.1 Provider-Side Workflow Codes"
#   "last" -> "Location: 4.1 Provider-Side Workflow Codes"   (only the nearest heading)
#   "none" -> no address in the text (it is still kept in the chunk's metadata)
CHUNK_LOCATION = "full"

# Remove the "| --- | --- |" line under table column names. It carries no
# meaning but costs tokens for the embedder, BM25 and the re-ranker.
TABLE_COMPACT = True

# When a big table is cut into pieces, repeat this many rows from the end of one
# piece at the top of the next (like CHUNK_OVERLAP, but for table rows).
TABLE_OVERLAP_ROWS = 1

# PDFs have no real "heading" labels like Word files do. We GUESS headings from
# the font size: a line this much bigger than normal text is treated as a heading.
PDF_DETECT_HEADINGS = True
PDF_HEADING_SIZE_RATIO = 1.15     # 15% bigger than the normal text

# Find tables inside PDFs and keep them as proper tables (slower ingest).
PDF_DETECT_TABLES = True

# ----------------------------------------------------------------
# 4. RETRIEVAL (finding the right chunks for a question)
# ----------------------------------------------------------------
COLLECTION_NAME = "my_knowledge"   # like a "table name" inside the vector DB

TOP_K = 5   # how many chunks we hand to the AI for each question

# Similarity goes from 0 (unrelated) to 1 (same meaning).
# Chunks scoring below this are thrown away as "not relevant enough".
MIN_SIMILARITY = 0.35

# ----------------------------------------------------------------
# 4b. SMART SEARCH  (Phase 2 - see docs/learning/09..11)
# ----------------------------------------------------------------
# How we search for chunks:
#   "vector"  -> meaning search only        (Phase 1 behaviour)
#   "keyword" -> exact-word search only     (BM25, like a classic search engine)
#   "hybrid"  -> BOTH, then merge the lists (best of both worlds)
RETRIEVAL_MODE = "hybrid"

# Each search method brings this many candidates to the "shortlist".
# More candidates = better chance the right chunk is in there, but slower re-ranking.
CANDIDATES_PER_SEARCH = 20

# A constant used when merging the two lists (Reciprocal Rank Fusion).
# 60 is the value from the original research paper - you rarely need to touch it.
RRF_K = 60

# The RE-RANKER is a second, smarter "judge" model that reads the question
# and each shortlisted chunk TOGETHER and gives a relevance score.
USE_RERANKER = True
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"    # lighter option: "BAAI/bge-reranker-base"
RERANKER_DEVICE = "auto"     # "auto" (GPU if available), "cuda" or "cpu"
RERANKER_MAX_LENGTH = 512    # tokens per (question + chunk) pair; longer text is cut
RERANKER_BATCH_SIZE = 8      # pairs judged at once

# Re-ranker scores go from 0 (irrelevant) to 1 (perfect match).
# Chunks below this are thrown away, even if the search liked them.
MIN_RERANK_SCORE = 0.02

# ...but if the BEST chunk passes that bar (so the question IS about your documents),
# always keep at least this many chunks. The re-ranker can be very strict: for
# "what is the search API used for?" it kept only 1 of 33 chunks and the answer
# missed half the story (Payer Discovery was there, Policy Discovery was cut).
# Nonsense questions still get 0 chunks, because their BEST chunk fails the bar.
# 1 = old behaviour (strict cut-off).
RERANK_MIN_KEEP = 3

# ----------------------------------------------------------------
# 5. GENERATION (the AI writing the answer)
# ----------------------------------------------------------------
# Temperature = creativity. 0 = strict and repeatable, 1 = creative.
# For answering from documents we want strict.
TEMPERATURE = 0.2

# How much text (in tokens) the model can "see" at once:
# instructions + retrieved chunks + your question + its answer.
CONTEXT_WINDOW = 8192

# Qwen3.5 can "think out loud" before answering. That is slower and
# not needed for document Q&A, so we switch it off by default.
ENABLE_THINKING = False

# Agent mode (Phase 5) reads files over several steps, and every tool result stays in
# the conversation. It needs more room than one-shot Q&A. (Switching between modes with
# different sizes makes Ollama reload the model once - a few seconds.)
AGENT_CONTEXT_WINDOW = 16384

# ----------------------------------------------------------------
# 5b. CONVERSATION MEMORY  (Phase 3 - see docs/learning/15_conversation_memory.md)
# ----------------------------------------------------------------
# How many earlier question/answer pairs the assistant remembers.
MEMORY_TURNS = 3

# Follow-ups like "and what about the second one?" are useless as a SEARCH query.
# So before searching we give them the context they need:
#   "context" -> search  previous question + follow-up                   (free, DEFAULT)
#   "auto"    -> search  previous question + Qwen's stand-alone rewrite  (~1.5 s slower)
#   "always"  -> like auto, but rewrite EVERY question once there is history
#   "off"     -> never (Phase 1/2 behaviour)
# Measured on data/eval/followups.json: context 88% Hit@1, Qwen rewrite alone 62-75%,
# no memory 50%. See docs/learning/15_conversation_memory.md
QUERY_REWRITE = "context"

# Old answers are shortened to this many characters when we remind the AI of them,
# so the history doesn't crowd the retrieved sources out of the context window.
HISTORY_ANSWER_MAX_CHARS = 1200

# ----------------------------------------------------------------
# 6. OCR - reading scanned PDFs and images  (Phase 2 - docs/learning/12)
# ----------------------------------------------------------------
# A scanned PDF page is just a PHOTO of text - there are no letters to copy out.
# OCR (Optical Character Recognition) = "look at the picture and type out the text".
# Qwen3.5 can SEE images, so we use it as our OCR engine (nothing extra to install).
OCR_ENABLED = True
OCR_MODEL = CHAT_MODEL

# A PDF page with fewer real letters than this is treated as "scanned" and OCR'd.
OCR_MIN_TEXT_CHARS = 30

# Picture quality when we photograph a PDF page. Higher = sharper but slower.
OCR_DPI = 150

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Which file types we know how to read.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    # code files are just text too
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cs", ".go",
    ".sql", ".json", ".yaml", ".yml", ".html", ".css", ".xml", ".csv",
} | IMAGE_EXTENSIONS   # images are read with OCR

# ----------------------------------------------------------------
# 7. EVALUATION - measuring quality  (Phase 2 - docs/learning/13)
# ----------------------------------------------------------------
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_QUESTIONS_FILE = EVAL_DIR / "questions.json"
