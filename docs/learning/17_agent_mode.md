# 17 - Agent Mode: letting the AI look at your folders

Chat and Cowork answer in **one shot**: search → write. Agent mode works in **steps**:
the model picks a tool, we run it, it looks at the result, and decides what to do next.

```
you    : "Where is the sandbox check implemented?"
model  : → repo_overview()                 we run it, send back the tree
model  : → grep("resolve_path")            we run it, send back file:line matches
model  : → read_file("rag/agent_tools.py", 95, 130)
model  : "It's in resolve_path() in rag/agent_tools.py:110 ..."   ← no tool call = final answer
```

## The pieces

| File | Job |
|---|---|
| `rag/agent_tools.py` | The tools (plain Python functions) + the **sandbox** |
| `rag/agent.py` | The loop: call model → run tools → repeat (max 12 steps) |
| `server.py` | `mode="agent"` in `/api/chat`, and `/api/agent/folders` |
| `web/app.js` | Agent mode button, folder chips, live step timeline |

## How tool calling works (Ollama)

We send the tool descriptions as JSON schemas along with the messages:

```json
{"type": "function", "function": {"name": "read_file",
  "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
```

The model replies either with normal text (the answer) or with:

```json
{"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "server.py"}}}]}
```

We run the tool and append `{"role": "tool", "content": "<result>"}`, then ask again.

## Tools (Phase 1 - all read-only)

| Tool | What it does |
|---|---|
| `repo_overview` | Tree, file types, tech stack, README - the best first call |
| `list_dir` | Files/folders in one folder, with sizes |
| `find_files` | Find by name pattern (`*.py`, `config*`) |
| `grep` | Search inside files (regex) → `file:line: text` |
| `read_file` | Read up to 250 lines, with line numbers |
| `disk_usage` | Folder size + largest items |

## Safety rules

1. **Sandbox** - every path is resolved (`..`, symlinks) and must be inside a folder you
   added in the UI (`data/agent_folders.json`). Whole drives (`C:\`) are refused.
2. **Read-only** - no tool can write, delete or run commands (yet).
3. **Secrets** - `.env`, `*.pem`, `*.key`, `id_rsa*` are never read.
4. **Limits** - results are capped (6000 chars) and scans stop after 20 s, so one call
   cannot flood the small model's context window.

## Small-model tricks

- `AGENT_CONTEXT_WINDOW = 16384` (bigger than Chat's 8192 - tool results pile up).
- Old tool results are shortened; only the newest 3 stay complete.
- The same call twice in a row gets a "you already have this" nudge instead of re-running.
- Out of steps → tools are switched off and the model must answer with what it has.

## What's next

- Phase 2: storage analysis (junk/temp/duplicates, report only)
- Phase 3: approval cards in the UI (Approve / Reject)
- Phase 4: write actions - file edits, Recycle-Bin cleanup, run commands - all approval-gated
- Phase 5: code-aware indexing for large repos
