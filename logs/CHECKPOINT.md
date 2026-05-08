# CHECKPOINT — 2026-05-08

**Status:** PHASE 0 complete. Ready to enter PHASE 1.

- ✅ Sandbox: `.venv/` on **Python 3.12.13**, `sherlock` installed editable.
- ✅ Evaluator: `~/Desktop/cli-wrapper-unified/` reachable via both CLI and Python import. Default Gemini id `gemini-3.1-flash-lite-preview` matches the spec exactly. Smoke calls returned expected text.
- ✅ Layout, logs, state, git baseline all in place.
- ⏭ **Next:** PHASE 1 — generate the long synthetic dummy conversation. Will write to `evaluation/dummy_conversation.md`, then create `logs/AWAITING_PHASE1_APPROVAL.md` and stop.
- 🔑 **User action needed before PHASE 3 only:** API keys for Anthropic / OpenAI / Gemini in env vars (the spec's YAML config wires them up). PHASE 1–2 generation does not need them.

See `logs/curated.md` for the full bootstrap log and `INTENT_DEVIATIONS.md` for the two PHASE 0 deviations (wrapper-via-Python-import, python3.12-preference).
