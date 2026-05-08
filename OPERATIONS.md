# OPERATIONS — Sandbox, logging, and file layout

> Version: v0.3 · 2026-05-08
> How the build environment is set up and how the agent communicates state to the user via files.

---

## 1. Sandbox

### 1.1 Strategy

Lightweight. **No Docker.** The agent uses a Python virtual environment in a dedicated project directory. This is enough isolation for the user's intent ("not a scary program, just keep it light") and avoids the overhead and platform-juggling of containers.

### 1.2 Project root

The agent works in a single project directory. Suggested location: `~/sherlock-build/` or wherever the user clones this spec package. The directory layout after PHASE 0 should look like:

```
sherlock-build/
├── README.md                       # from spec package
├── SPEC.md
├── AGENTS_AND_LOOP.md
├── EVALUATION_PROTOCOL.md
├── OPERATIONS.md                   # this file
├── INTENT_DEVIATIONS.md
│
├── .venv/                          # the Python virtual env (gitignored)
├── pyproject.toml                  # created by agent in M1
├── requirements.txt                # or use pyproject [project.dependencies]
│
├── sherlock/                       # the library source (created by agent)
│   ├── __init__.py
│   ├── providers/
│   ├── memory/
│   ├── bootstrap/
│   ├── inference/
│   ├── rag/
│   ├── tools/
│   ├── evolution/
│   └── cli/
│
├── tests/                          # pytest tests
│   ├── unit/
│   ├── integration/
│   └── milestones/                 # tests that gate each milestone's exit
│
├── prompts/
│   ├── main_system_prompt.md       # the test main prompt the agent writes
│   └── meta_context.md             # built-in Sherlock meta-context
│
├── evaluation/                     # see EVALUATION_PROTOCOL.md § 4
│
└── logs/                           # see § 3 below
```

### 1.3 Bootstrap script

The agent writes `bootstrap.sh` (or `.ps1` for Windows users) early in PHASE 0. Suggested content:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Detect Python (require 3.11+)
if ! command -v python3.11 >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.11+ required" >&2
    exit 1
fi

PY=$(command -v python3.11 || command -v python3)

# Create venv if absent
if [ ! -d ".venv" ]; then
    "$PY" -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
    pip install -e .
fi

# Sanity
python -c "import sys; print('Python:', sys.version)"

echo "Bootstrap complete. Activate with: source .venv/bin/activate"
```

The agent runs this at PHASE 0 and again any time `requirements.txt` or `pyproject.toml` changes.

### 1.4 Verifying the cli-wrapper-unified

In PHASE 0, the agent verifies the evaluation tool is reachable. The verification:

1. Look for the wrapper at `~/Desktop/cli-wrapper-unified` and similar standard locations on the user's home Desktop. The exact name and path should be confirmed by the agent — it is the user's tool.
2. Run the wrapper's `--help` (or equivalent) to confirm it executes.
3. Run a tiny test invocation calling `gemini-3.1-flash-lite-preview` (or the closest valid id the wrapper accepts) with a trivial prompt to confirm it produces output.
4. Document the verified path and exact CLI shape in `logs/curated.md` for future reference.

If verification fails, write SOS. Do not proceed to Phase 3 evaluation without a working evaluator.

### 1.5 Running on macOS, Linux, Windows

The agent should not assume any particular OS. The user has Mac and Windows machines (per memory). If a step is OS-specific, the agent writes both variants and detects which to use at runtime. Do not silently assume macOS just because that is the user's primary dev machine.

---

## 2. Dependencies

The agent installs only what is needed. Suggested initial set (refine during M1):

```
# Core LLM SDKs
anthropic
openai
google-generativeai

# Optionally use unified provider abstraction
# litellm  # decide in M1

# Memory / RAG
chromadb
rank-bm25
sentence-transformers
sqlmodel
pydantic
pyyaml

# CLI / config / logging
typer
structlog

# Async
anyio  # or rely on stdlib asyncio

# Web search
requests
# tavily-python or equivalent — install when M3 begins

# Test
pytest
pytest-asyncio

# Lint
ruff
black

# UI (M8)
streamlit  # only when M8 begins

# Dev convenience
ipython
```

Each import the agent adds to the codebase implies an entry in `requirements.txt` or `pyproject.toml`. Keep the dependency surface lean.

---

## 3. Logging

### 3.1 Two logs, on purpose

| File | Writer | Reader | Format |
|------|--------|--------|--------|
| `logs/raw.jsonl` | The code being built (Sherlock itself + the test harness) | The orchestrator agent + user | JSON Lines |
| `logs/curated.md` | The orchestrator agent (Claude Code) | The agent itself in next loop + user | Markdown |

Why two:
- Raw is for machines and post-hoc analysis. Anything emitted by code goes here, structured.
- Curated is for the agent's working memory and for the user's quick check-in. It is what the agent reads at the start of each loop to remember "where am I, what just happened, what should I try next".

The user can read either. In practice the curated one is friendlier.

### 3.2 Raw log schema

`logs/raw.jsonl` is append-only JSON Lines. One event per line:

```json
{
  "ts": "2026-05-08T14:32:11.123Z",
  "level": "info|warning|error|debug",
  "loop_id": 47,
  "milestone": "M5",
  "phase": "build|eval|bootstrap|setup",
  "component": "providers.anthropic|memory.decay|bootstrap.engine|...",
  "event": "short_event_name",
  "data": { /* arbitrary structured payload */ },
  "duration_ms": 1234,           // optional, when measuring
  "tokens_used": { "in": 0, "out": 0 },  // optional, for cost tracking
  "cost_usd": 0.0023             // optional, when applicable
}
```

Required fields: `ts`, `level`, `event`. All others optional.

Rotation: when `raw.jsonl` exceeds 100 MB, rotate to `raw.<date>.jsonl.gz` and start fresh. **Do not delete** rotated logs.

### 3.3 Curated log schema

`logs/curated.md` is a markdown document, append-only at the bottom. Each loop appends a section:

```markdown
## Loop 47 — 2026-05-08 14:32

**Milestone:** M5 (Async pipeline)
**Phase:** build

**What I attempted:**
Implemented `asyncio.gather` parallelism for LLM 2 + LLM 3 + search.
Added integration test `tests/integration/test_async_parallel.py`.

**What happened:**
Test fails: web search task crashes with `ConnectionError` because Tavily
key is not set in the test environment.

**Root cause:**
The test fixture for "all components running" did not stub the search client.
Search was firing real API calls in tests.

**What I will do next:**
Add a search-client stub to `tests/conftest.py` and re-run.

**Open question:**
Should integration tests use real Tavily (slow, costs money) or always stub?
Tentatively going with stub by default + a separate `--live` pytest flag for
on-demand real-API runs.

---
```

The agent reads the **last few sections** of `curated.md` at the start of each loop to recall what was happening. Long curated logs can be summarized periodically (the agent writes a `## Summary as of <date>` section at the top and clears below; old details are still in raw.jsonl).

### 3.4 Other log files (informational, non-blocking)

| File | Written when | Purpose |
|------|--------------|---------|
| `logs/CHECKPOINT.md` | After each milestone, or every 50 loops | Short status for the user |
| `logs/SOS.md` | Agent stuck — see AGENTS_AND_LOOP.md § 4 | Block until resolved |
| `logs/SOS_RESOLVED.md` | Created by the user | Signal to resume |
| `logs/AWAITING_PHASE1_APPROVAL.md` | Phase 1 done | Block until approval |
| `logs/AWAITING_PHASE2_APPROVAL.md` | Phase 2 done | Block until approval |
| `logs/DONE.md` | Final 80% achieved | Project complete |

---

## 4. Cost and time tracking

The agent tracks total cost per run in `logs/cost.json`:

```json
{
  "started_at": "2026-05-08T10:00:00Z",
  "by_model": {
    "claude-opus-4-7": {"in_tokens": 1234567, "out_tokens": 234567, "cost_usd": 12.34},
    "gpt-5": {"in_tokens": 89000, "out_tokens": 12000, "cost_usd": 5.67},
    "gemini-3.1-flash-lite-preview": {"in_tokens": 50000, "out_tokens": 5000, "cost_usd": 0.50}
  },
  "total_cost_usd": 18.51,
  "cost_cap_usd": 50.00,
  "remaining_budget_usd": 31.49
}
```

Updated after each LLM call. When `remaining_budget_usd <= 0`, the agent stops and writes SOS.

Time tracking is similar: `logs/time.json`:

```json
{
  "started_at": "2026-05-08T10:00:00Z",
  "elapsed_hours": 4.2,
  "time_cap_hours": 12.0,
  "remaining_hours": 7.8
}
```

---

## 5. Restart resilience

The agent must be killable and resumable. State that survives restarts:

- All files under the project directory (logs, evaluation, sherlock/, tests/, etc.)
- A `state/current.json` file the agent maintains:

```json
{
  "phase": "build",
  "milestone": "M5",
  "loop_id": 47,
  "last_evaluation_score": 62,
  "last_user_approval": {"phase1": true, "phase2": true},
  "blocked_on": null
}
```

Updated at the start of each loop. On restart, the agent reads this file and resumes from the right place.

---

## 6. Git

The agent commits frequently. Every successful loop that produces working code gets at least one commit. Suggested commit-message format:

```
[M5/loop-47] Add async parallel pipeline for LLM 2/3/search

- Implements asyncio.gather in pipeline.py
- Adds tests/integration/test_async_parallel.py
- Fixes search-client stub in conftest

Refs: SPEC.md § 4.3
```

Do not commit:
- `.venv/`
- `logs/raw.jsonl` and rotations (gitignore them; they are user-local artifacts)
- API keys, `.env` files, anything in `secrets/`
- `evaluation/runs/*/` (gitignore; these are run artifacts)

A `.gitignore` template the agent should create in PHASE 0:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env
.env.*
secrets/
logs/raw.jsonl
logs/raw.*.jsonl.gz
logs/cost.json
logs/time.json
state/current.json
evaluation/runs/
sherlock.db
sherlock_vectors/
```

---

## 7. The user's view in three files

If the user only reads three files to know what's happening:

1. **`logs/curated.md`** — recent sections show what the agent is doing
2. **`logs/CHECKPOINT.md`** — quick "where are we" summary
3. **`logs/SOS.md`** (if it exists) — the agent needs you

Plus `evaluation/runs/<latest>/score.txt` for the most recent eval result.

The agent should design its logging assuming the user will glance at it for 30 seconds, not read it line by line.
