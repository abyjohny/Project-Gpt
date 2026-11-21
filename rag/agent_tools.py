"""
agent_tools.py - PHASE 5: The agent's "hands" (read-only, sandboxed).
=====================================================================

In Agent mode the model does not just answer - it can LOOK at your computer:
list folders, find files, search inside them, read them, measure disk usage.

Each tool is a plain Python function. The model never touches the disk itself:
it asks for a tool by name ("read_file", {"path": "server.py"}), WE run it and
hand back the text result. That gives us one place to enforce the rules:

    1. SANDBOX   - every path must be inside a folder YOU allowed
                   (data/agent_folders.json, managed from the web UI).
                   "..", symlinks and absolute paths elsewhere are rejected.
    2. READ-ONLY - Phase 1 has no tool that writes, deletes or runs anything.
    3. SECRETS   - files like .env, *.pem, id_rsa are never read.
    4. LIMITS    - results are capped so one tool call cannot flood the
                   model's small context window (8k tokens on a 4B model).

Learn more: docs/learning/17_agent_mode.md
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import threading
import time
from pathlib import Path

from rag import config

FOLDERS_FILE = config.PROJECT_ROOT / "data" / "agent_folders.json"

# Folders that are almost never interesting and can be HUGE.
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".idea", ".vs", ".next", ".nuxt", "dist", "build",
    ".gradle", "target", "bin", "obj", ".cache", ".tox", "coverage", ".terraform",
}
# Never read these, even inside an allowed folder.
SECRET_PATTERNS = ("*.env", ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12",
                   "id_rsa*", "id_ed25519*", "*.kdbx", "credentials*.json", "secrets.*")
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".7z", ".rar",
               ".gz", ".tar", ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".class", ".jar",
               ".mp3", ".mp4", ".mov", ".avi", ".woff", ".woff2", ".ttf", ".otf", ".db", ".sqlite",
               ".onnx", ".pt", ".safetensors", ".gguf", ".parquet", ".docx", ".xlsx", ".pptx"}

MAX_RESULT_CHARS = 6000        # what one tool call may hand back to the model
MAX_READ_LINES = 250
MAX_LIST_ENTRIES = 150
MAX_FIND_RESULTS = 80
MAX_GREP_MATCHES = 50
MAX_GREP_FILE_BYTES = 1_000_000
SCAN_TIME_LIMIT_S = 20         # disk_usage / grep give up (with partial results) after this


class ToolError(Exception):
    """A problem the MODEL should see and react to (bad path, not allowed, ...)."""


# ----------------------------------------------------------------
# Allowed folders (the sandbox)
# ----------------------------------------------------------------
_folders_lock = threading.Lock()


def list_folders() -> list[str]:
    try:
        data = json.loads(FOLDERS_FILE.read_text(encoding="utf-8"))
        return [f for f in data.get("folders", []) if isinstance(f, str)]
    except (OSError, ValueError):
        return []


def _save_folders(folders: list[str]) -> None:
    FOLDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FOLDERS_FILE.write_text(json.dumps({"folders": folders}, indent=1), encoding="utf-8")


def add_folder(path: str) -> str:
    raw = (path or "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Empty path.")
    folder = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not folder.is_absolute():
        raise ValueError("Please give a full path, e.g. D:\\dev\\my-project")
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    if folder == Path(folder.anchor):
        raise ValueError("Allowing a whole drive is too broad. Pick a specific folder.")
    with _folders_lock:
        folders = list_folders()
        if str(folder) not in folders:
            folders.append(str(folder))
            _save_folders(folders)
    return str(folder)


def remove_folder(path: str) -> bool:
    with _folders_lock:
        folders = list_folders()
        if path not in folders:
            return False
        folders.remove(path)
        _save_folders(folders)
        return True


def _roots() -> list[Path]:
    return [Path(f) for f in list_folders()]


def resolve_path(path: str | None) -> Path:
    """Turn what the model asked for into a real path INSIDE an allowed folder - or refuse."""
    roots = _roots()
    if not roots:
        raise ToolError("No folders are allowed yet. Ask the user to add one with the 'Folders' button.")
    raw = (path or "").strip().strip('"').strip("'")
    candidate = Path(raw) if raw else roots[0]
    if not candidate.is_absolute():
        candidate = roots[0] / candidate          # relative paths start in the first allowed folder
    resolved = candidate.resolve()                # follows "..", symlinks and junctions
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    allowed = ", ".join(str(r) for r in roots)
    raise ToolError(f"Access denied: {resolved} is outside the allowed folders ({allowed}).")


def _display(path: Path) -> str:
    """Short path for the model: relative to its allowed folder when possible."""
    for root in _roots():
        if path == root:
            return str(root)
        if root in path.parents:
            return str(path.relative_to(root)).replace("\\", "/")
    return str(path)


def _is_secret(path: Path) -> bool:
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, p) for p in SECRET_PATTERNS)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _walk(start: Path, deadline: float | None = None):
    """os.walk that skips ignored folders, never follows links and respects a time limit."""
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.startswith("."))
        yield Path(dirpath), dirnames, sorted(filenames)
        if deadline and time.monotonic() > deadline:
            return


# ----------------------------------------------------------------
# The tools
# ----------------------------------------------------------------
def tool_list_allowed_folders() -> str:
    roots = list_folders()
    if not roots:
        return "No folders are allowed yet."
    return "Allowed folders:\n" + "\n".join(f"- {r}" for r in roots)


def tool_list_dir(path: str = "") -> str:
    folder = resolve_path(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {_display(folder)}")
    entries = []
    try:
        items = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise ToolError(f"Permission denied: {_display(folder)}") from exc
    for item in items[:MAX_LIST_ENTRIES]:
        try:
            if item.is_dir():
                note = " (skipped by default)" if item.name in IGNORED_DIRS else ""
                entries.append(f"[dir]  {item.name}/{note}")
            else:
                entries.append(f"[file] {item.name}  ({_human(item.stat().st_size)})")
        except OSError:
            entries.append(f"[?]    {item.name}")
    more = f"\n... and {len(items) - MAX_LIST_ENTRIES} more" if len(items) > MAX_LIST_ENTRIES else ""
    return f"{folder}  ({len(items)} entries)\n" + "\n".join(entries) + more


def tool_find_files(pattern: str, path: str = "") -> str:
    folder = resolve_path(path)
    pattern = (pattern or "*").strip()
    if not any(ch in pattern for ch in "*?["):
        pattern = f"*{pattern}*"                    # "config" -> "*config*"
    hits = []
    deadline = time.monotonic() + SCAN_TIME_LIMIT_S
    for dirpath, _, filenames in _walk(folder, deadline):
        for name in filenames:
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                hits.append(_display(dirpath / name))
                if len(hits) >= MAX_FIND_RESULTS:
                    return "\n".join(hits) + f"\n... stopped at {MAX_FIND_RESULTS} results, narrow the pattern."
    return "\n".join(hits) if hits else f"No files matching '{pattern}' in {_display(folder)}."


def tool_grep(pattern: str, path: str = "", file_glob: str = "*") -> str:
    folder = resolve_path(path)
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)   # model sent plain text with ( or [
    files = [folder] if folder.is_file() else None
    matches: list[str] = []
    deadline = time.monotonic() + SCAN_TIME_LIMIT_S

    def scan(file: Path) -> bool:
        if file.suffix.lower() in BINARY_EXTS or _is_secret(file):
            return True
        try:
            if file.stat().st_size > MAX_GREP_FILE_BYTES:
                return True
            with open(file, encoding="utf-8", errors="ignore") as handle:
                for number, line in enumerate(handle, start=1):
                    if regex.search(line):
                        matches.append(f"{_display(file)}:{number}: {line.strip()[:200]}")
                        if len(matches) >= MAX_GREP_MATCHES:
                            return False
        except OSError:
            pass
        return True

    if files:
        scan(files[0])
    else:
        for dirpath, _, filenames in _walk(folder, deadline):
            for name in filenames:
                if fnmatch.fnmatch(name.lower(), (file_glob or "*").lower()) and not scan(dirpath / name):
                    return "\n".join(matches) + f"\n... stopped at {MAX_GREP_MATCHES} matches."
    return "\n".join(matches) if matches else f"No matches for '{pattern}'."


def tool_read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    file = resolve_path(path)
    if not file.is_file():
        raise ToolError(f"Not a file: {_display(file)}")
    if _is_secret(file):
        raise ToolError(f"Refused: {file.name} looks like a secrets file and is never read.")
    if file.suffix.lower() in BINARY_EXTS:
        raise ToolError(f"{file.name} is a binary file ({file.suffix}); it cannot be read as text.")
    try:
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ToolError(f"Cannot read {_display(file)}: {exc}") from exc
    start = max(1, int(start_line or 1))
    end = min(len(lines), int(end_line) if end_line else start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
    body = "\n".join(f"{n:>4}| {lines[n - 1]}" for n in range(start, end + 1))
    note = f"\n... file has {len(lines)} lines; call read_file again with start_line={end + 1} for more." \
        if end < len(lines) else ""
    return _clip(f"{_display(file)} (lines {start}-{end} of {len(lines)})\n{body}{note}")


def tool_repo_overview(path: str = "") -> str:
    """Folder tree + file-type counts + detected stack + README start. The best FIRST call on a codebase."""
    folder = resolve_path(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {_display(folder)}")
    tree: list[str] = []
    ext_counts: dict[str, int] = {}
    total_files = 0
    deadline = time.monotonic() + SCAN_TIME_LIMIT_S
    for dirpath, dirnames, filenames in _walk(folder, deadline):
        depth = len(dirpath.relative_to(folder).parts)
        for name in filenames:
            total_files += 1
            ext = Path(name).suffix.lower() or "(none)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        if depth <= 2 and len(tree) < 120:
            indent = "  " * depth
            label = dirpath.name if depth else str(folder)
            tree.append(f"{indent}{label}/")
            shown = filenames[:12] if depth < 2 else filenames[:5]
            tree.extend(f"{indent}  {n}" for n in shown)
            if len(filenames) > len(shown):
                tree.append(f"{indent}  ... (+{len(filenames) - len(shown)} files)")
        if depth >= 2:
            dirnames[:] = dirnames if depth < 3 else []

    markers = {
        "package.json": "Node.js / JavaScript", "tsconfig.json": "TypeScript", "requirements.txt": "Python",
        "pyproject.toml": "Python", "setup.py": "Python", "Pipfile": "Python", "go.mod": "Go",
        "Cargo.toml": "Rust", "pom.xml": "Java (Maven)", "build.gradle": "Java/Kotlin (Gradle)",
        "Dockerfile": "Docker", "docker-compose.yml": "Docker Compose", "Gemfile": "Ruby",
        "composer.json": "PHP", "next.config.js": "Next.js", "vite.config.ts": "Vite", "vite.config.js": "Vite",
    }
    stack = sorted({label for marker, label in markers.items() if (folder / marker).exists()})
    if any(folder.glob("*.sln")) or any(folder.glob("*.csproj")):
        stack.append(".NET / C#")

    top_ext = sorted(ext_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    parts = [f"Project: {folder}",
             f"Detected stack: {', '.join(stack) or 'unknown'}",
             f"Files (excluding node_modules/.git/venv etc.): {total_files}",
             "Most common file types: " + ", ".join(f"{e} x{c}" for e, c in top_ext),
             "", "Tree (depth 2):", *tree]
    for readme in ("README.md", "readme.md", "README.txt", "README"):
        if (folder / readme).is_file():
            head = (folder / readme).read_text(encoding="utf-8", errors="replace")[:1200]
            parts += ["", f"{readme} (start):", head]
            break
    for manifest in ("package.json", "requirements.txt", "pyproject.toml"):
        if (folder / manifest).is_file():
            parts += ["", f"{manifest}:", (folder / manifest).read_text(encoding="utf-8", errors="replace")[:800]]
    return _clip("\n".join(parts))


def tool_disk_usage(path: str = "") -> str:
    """Size of a folder and of each item directly inside it (largest first). Includes ignored folders."""
    folder = resolve_path(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {_display(folder)}")
    deadline = time.monotonic() + SCAN_TIME_LIMIT_S
    timed_out = False

    def size_of(p: Path) -> int:
        nonlocal timed_out
        if p.is_file():
            return p.stat().st_size
        total = 0
        for dirpath, _, filenames in os.walk(p, followlinks=False):
            for name in filenames:
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    pass
            if time.monotonic() > deadline:
                timed_out = True
                break
        return total

    rows = []
    for item in folder.iterdir():
        try:
            rows.append((size_of(item), item))
        except OSError:
            continue
        if timed_out:
            break
    rows.sort(key=lambda r: r[0], reverse=True)
    total = sum(r[0] for r in rows)
    lines = [f"{folder}: {_human(total)} total{' (PARTIAL - scan time limit reached)' if timed_out else ''}"]
    for size, item in rows[:30]:
        pct = (size / total * 100) if total else 0
        lines.append(f"{_human(size):>10}  {pct:5.1f}%  {item.name}{'/' if item.is_dir() else ''}")
    if len(rows) > 30:
        lines.append(f"... and {len(rows) - 30} smaller items")
    return "\n".join(lines)


def tool_find_temp_junk(path: str = "") -> str:
    """Scan for well-known junk/temp/cache folders and files. Returns a summary table."""
    folder = resolve_path(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {_display(folder)}")

    # Well-known junk folder names (case-insensitive on Windows)
    JUNK_DIRS = {
        "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".cache", ".tox", "coverage", ".next", ".nuxt", "dist", "build", ".gradle",
        "target", "bin", "obj", ".terraform", ".angular", ".parcel-cache",
        ".sass-cache", "tmp", "temp", ".temp", "thumbs.db", "desktop.ini",
    }
    # Junk file patterns
    JUNK_FILE_PATTERNS = (
        "*.pyc", "*.pyo", "*.o", "*.obj", "*.log", "*.bak", "*.swp", "*.swo",
        "*~", "*.orig", "*.tmp", "*.temp", "Thumbs.db", "desktop.ini", ".DS_Store",
        "*.map",  # source maps can be huge
    )

    results: list[dict] = []          # {path, type, size, reason}
    total_junk = 0
    deadline = time.monotonic() + SCAN_TIME_LIMIT_S
    timed_out = False

    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        if time.monotonic() > deadline:
            timed_out = True
            break
        current = Path(dirpath)
        # Check if any subdirectory is a junk dir
        junk_here = []
        keep = []
        for d in dirnames:
            if d.lower() in JUNK_DIRS or d.lower() in {".venv", "venv", "env", ".git"}:
                junk_path = current / d
                try:
                    size = sum(
                        os.lstat(os.path.join(dp, f)).st_size
                        for dp, _, fns in os.walk(junk_path, followlinks=False)
                        for f in fns
                        if time.monotonic() < deadline
                    )
                except OSError:
                    size = 0
                if size > 0:
                    results.append({"path": str(junk_path), "type": "dir", "size": size, "reason": d})
                    total_junk += size
                junk_here.append(d)
            else:
                keep.append(d)
        dirnames[:] = keep       # don't descend into junk dirs

        # Check files in this folder
        for name in filenames:
            if any(fnmatch.fnmatch(name.lower(), p.lower()) for p in JUNK_FILE_PATTERNS):
                fp = current / name
                try:
                    sz = fp.stat().st_size
                except OSError:
                    sz = 0
                results.append({"path": str(fp), "type": "file", "size": sz, "reason": name})
                total_junk += sz

    results.sort(key=lambda r: r["size"], reverse=True)
    lines = [f"Junk/temp scan of {folder}: {_human(total_junk)} total "
             f"in {len(results)} items{' (PARTIAL - time limit)' if timed_out else ''}"]
    if not results:
        lines.append("No junk or temp items found.")
    for r in results[:40]:
        tag = "DIR " if r["type"] == "dir" else "FILE"
        lines.append(f"  {_human(r['size']):>10}  [{tag}]  {r['path']}")
    if len(results) > 40:
        lines.append(f"  ... and {len(results) - 40} smaller items")
    lines.append("")
    lines.append("NOTE: These are common cache/build/temp items. Deleting them is usually safe,")
    lines.append("but tell the user what you found and let THEM decide. You cannot delete anything.")
    return "\n".join(lines)


def tool_find_large_files(path: str = "", min_size_mb: int = 10, top_n: int = 30) -> str:
    """Find the largest files in a folder tree (recursively). Default: files >= 10 MB."""
    folder = resolve_path(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {_display(folder)}")
    min_bytes = max(1, int(min_size_mb or 10)) * 1024 * 1024
    top_n = min(max(5, int(top_n or 30)), 60)
    hits: list[tuple[int, Path]] = []
    deadline = time.monotonic() + SCAN_TIME_LIMIT_S
    timed_out = False
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))   # skip hidden
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                sz = os.lstat(fp).st_size
            except OSError:
                continue
            if sz >= min_bytes:
                hits.append((sz, fp))
        if time.monotonic() > deadline:
            timed_out = True
            break
    hits.sort(key=lambda h: h[0], reverse=True)
    total = sum(h[0] for h in hits)
    lines = [f"Large files (>= {min_size_mb} MB) in {folder}: {len(hits)} found, "
             f"{_human(total)} total{' (PARTIAL)' if timed_out else ''}"]
    if not hits:
        lines.append("No files that large.")
    for sz, fp in hits[:top_n]:
        lines.append(f"  {_human(sz):>10}  {_display(fp)}")
    if len(hits) > top_n:
        lines.append(f"  ... and {len(hits) - top_n} more")
    return "\n".join(lines)


def tool_find_duplicates(path: str = "", min_size_kb: int = 100) -> str:
    """Find duplicate files (same content, different names) by hashing. Default: files >= 100 KB."""
    import hashlib
    folder = resolve_path(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {_display(folder)}")
    min_bytes = max(1, int(min_size_kb or 100)) * 1024
    # Phase 1: group by (size) - only files with the same size can be duplicates
    size_groups: dict[int, list[Path]] = {}
    deadline = time.monotonic() + SCAN_TIME_LIMIT_S
    timed_out = False
    for dirpath, dirnames, filenames in _walk(folder, deadline):
        for name in filenames:
            fp = Path(dirpath) / name
            if fp.suffix.lower() in BINARY_EXTS or _is_secret(fp):
                continue      # skip binary and secrets (can't read, or shouldn't)
            try:
                sz = os.lstat(fp).st_size
            except OSError:
                continue
            if sz >= min_bytes:
                size_groups.setdefault(sz, []).append(fp)

    # Phase 2: hash files that share a size
    hash_groups: dict[str, list[Path]] = {}
    for sz, files in size_groups.items():
        if len(files) < 2:
            continue
        if time.monotonic() > deadline:
            timed_out = True
            break
        for fp in files:
            try:
                h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
            except OSError:
                continue
            hash_groups.setdefault(h, []).append(fp)

    dupes = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}
    if not dupes:
        return f"No duplicate files (>= {min_size_kb} KB) found in {_display(folder)}."

    wasted = sum(paths[0].stat().st_size * (len(paths) - 1) for paths in dupes.values())
    lines = [f"Duplicate files in {_display(folder)}: {len(dupes)} group(s), "
             f"~{_human(wasted)} wasted{' (PARTIAL scan)' if timed_out else ''}"]
    for h, paths in sorted(dupes.items(), key=lambda kv: -kv[1][0].stat().st_size)[:15]:
        sz = paths[0].stat().st_size
        lines.append(f"\n  [{_human(sz)}] identical x{len(paths)}:")
        for fp in paths[:6]:
            lines.append(f"    {_display(fp)}")
        if len(paths) > 6:
            lines.append(f"    ... and {len(paths) - 6} more copies")
    return "\n".join(lines)


# ----------------------------------------------------------------
# Registry: what the model sees (JSON schema) + how we run it
# ----------------------------------------------------------------
def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []}}}


_PATH = {"type": "string", "description": "Folder or file path. Relative paths start in the first allowed folder."}

TOOLS = {
    "list_allowed_folders": (tool_list_allowed_folders, _schema(
        "list_allowed_folders", "Show which folders you are allowed to access.", {})),
    "repo_overview": (tool_repo_overview, _schema(
        "repo_overview", "Overview of a project folder: tree, file types, tech stack, README. "
                         "Call this FIRST when asked about a codebase.", {"path": _PATH})),
    "list_dir": (tool_list_dir, _schema(
        "list_dir", "List the files and folders directly inside a folder, with sizes.", {"path": _PATH})),
    "find_files": (tool_find_files, _schema(
        "find_files", "Find files by name pattern (recursive), e.g. '*.py', 'config*', 'auth'.",
        {"pattern": {"type": "string", "description": "Glob or part of a file name"}, "path": _PATH}, ["pattern"])),
    "grep": (tool_grep, _schema(
        "grep", "Search text inside files (recursive, case-insensitive regex). Returns file:line: text.",
        {"pattern": {"type": "string", "description": "Text or regex to search for"}, "path": _PATH,
         "file_glob": {"type": "string", "description": "Only search files matching this, e.g. '*.py'"}},
        ["pattern"])),
    "read_file": (tool_read_file, _schema(
        "read_file", f"Read a text file with line numbers (max {MAX_READ_LINES} lines per call).",
        {"path": _PATH, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"])),
    "disk_usage": (tool_disk_usage, _schema(
        "disk_usage", "How much disk space a folder uses, and its largest items.", {"path": _PATH})),
    "find_temp_junk": (tool_find_temp_junk, _schema(
        "find_temp_junk", "Scan a folder for well-known junk/temp/cache items (node_modules, __pycache__, "
                          ".git, venv, *.pyc, *.log, *.bak, etc.). Reports what could be cleaned.",
        {"path": _PATH})),
    "find_large_files": (tool_find_large_files, _schema(
        "find_large_files", "Find the largest files in a folder tree. Use to find space hogs.",
        {"path": _PATH,
         "min_size_mb": {"type": "integer", "description": "Minimum file size in MB (default 10)"},
         "top_n": {"type": "integer", "description": "How many results (default 30)"}},
    )),
    "find_duplicates": (tool_find_duplicates, _schema(
        "find_duplicates", "Find duplicate files (identical content) by hashing. Reports wasted space.",
        {"path": _PATH,
         "min_size_kb": {"type": "integer", "description": "Minimum file size in KB (default 100)"}},
    )),
}


def schemas() -> list[dict]:
    return [schema for _, schema in TOOLS.values()]


def run_tool(name: str, arguments: dict | str | None) -> tuple[bool, str]:
    """Run one tool. Returns (ok, text). Never raises: errors become text the model can read."""
    if name not in TOOLS:
        return False, f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except ValueError:
            arguments = {}
    func = TOOLS[name][0]
    allowed = func.__code__.co_varnames[:func.__code__.co_argcount]
    kwargs = {k: v for k, v in (arguments or {}).items() if k in allowed and v not in (None, "")}
    try:
        return True, _clip(func(**kwargs))
    except ToolError as exc:
        return False, f"Error: {exc}"
    except TypeError as exc:
        return False, f"Error: bad arguments for {name}: {exc}"
    except Exception as exc:          # a broken tool must not kill the agent loop
        return False, f"Error: {name} failed: {exc}"
