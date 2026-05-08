# Sherlock Build — Curated Log

This file is the agent's working memory. Each loop appends a section.
The user can skim this to see what's happening without reading raw.jsonl.

---

## Bootstrap — 2026-05-08

**Phase:** PHASE 0 (sandbox bootstrap)
**Build directory:** `/Users/minwoo_mini/Desktop/claude files/project_sherlock_spec/` (in-place; spec files coexist with build artifacts per OPERATIONS.md § 1.2)

**What I did this session:**

1. Read all 6 spec files in the order the user specified (README → SPEC → AGENTS_AND_LOOP → EVALUATION_PROTOCOL → OPERATIONS → INTENT_DEVIATIONS).
2. Verified environment: `~/Desktop/cli-wrapper-unified/` exists; Python 3.12.13, 3.13, 3.14 available locally; default Gemini model in the wrapper is exactly `gemini-3.1-flash-lite-preview` (matches `EVALUATION_PROTOCOL.md` § 3.3).
3. Created the directory skeleton: `sherlock/`, `tests/{unit,integration,milestones}/`, `prompts/`, `evaluation/runs/`, `logs/`, `state/`.
4. Wrote `.gitignore` (verbatim from `OPERATIONS.md` § 6 plus `.DS_Store`).
5. Wrote `bootstrap.sh` (per `OPERATIONS.md` § 1.3 with the python detection block prioritising `python3.12 → 3.13 → 3.11 → python3` because chromadb / sentence-transformers wheels are not yet guaranteed on 3.14).
6. Wrote skeleton `pyproject.toml` (no runtime deps yet — they grow per milestone).
7. Ran `bash bootstrap.sh` → venv created with **Python 3.12.13**, pip 26.1.1, `sherlock` installed editable. Bootstrap exited 0.
8. Verified the evaluator:
   - `~/Desktop/cli-wrapper-unified/.venv/bin/unified-cli --help` → printed Korean help, exit 0.
   - Smoke chat: `unified-cli chat "Reply with exactly: SHERLOCK_BOOTSTRAP_OK" -m gemini-3.1-flash-lite-preview --new` → returned `SHERLOCK_BOOTSTRAP_OK` (latency 5195 ms; tokens in/out = 8947/9; session `c713ff26-...`).
   - Python-import path: `pip install -e ~/Desktop/cli-wrapper-unified` → `from unified_cli import create; create('gemini', model='gemini-3.1-flash-lite-preview').chat('...')` returns `Response.text` and `Response.session_id`. Smoke call returned `PYIMPORT_OK` (session `9e4686b6-...`).
9. Initialized logs (`logs/raw.jsonl`, `logs/cost.json`, `logs/time.json`) and `state/current.json`.
10. Logged the import-path-vs-CLI choice in `INTENT_DEVIATIONS.md` (DEVIATION-001).

**Verified evaluator invocation shapes:**

CLI form (matches `EVALUATION_PROTOCOL.md` § 3.3 reasonable shape):
```
~/Desktop/cli-wrapper-unified/.venv/bin/unified-cli chat <prompt> \
    -m gemini-3.1-flash-lite-preview --new
```
The wrapper does not expose `--system` / `--user-file` / `--output` flags as the spec template suggested. To inject a system prompt, the agent will (a) prepend the rubric inline to the prompt, or (b) use the Python API with `UnifiedConversation` if/when needed. This concrete CLI shape will be re-checked at PHASE 3 evaluation time.

Python form (preferred for evaluation):
```python
from unified_cli import create
r = create("gemini", model="gemini-3.1-flash-lite-preview").chat(
    full_prompt_string,  # GOLD + CANDIDATE + rubric inline
)
score_json = r.text  # parse as JSON per evaluator_system_prompt rubric
```

**Open questions (non-blocking):**

- Whether the wrapper supports a true `--system` flag in chat-mode for cleaner system/user separation. Will inspect `src/unified_cli/cli.py` when PHASE 3 evaluation arrives. Current plan: prepend rubric to user message — this still gets scored deterministically because the rubric is in-band.
- API keys for Anthropic/OpenAI/Gemini (Sherlock runtime in PHASE 3): the wrapper rides on subscription auth, but Sherlock itself will need real provider keys. Not needed for PHASE 0–2.

**Next:**

Enter PHASE 1 — generate the long synthetic dummy conversation per `EVALUATION_PROTOCOL.md` § 1, self-judge, save to `evaluation/dummy_conversation.md`, write `logs/AWAITING_PHASE1_APPROVAL.md`, stop and wait for `evaluation/PHASE1_APPROVED`.

---
