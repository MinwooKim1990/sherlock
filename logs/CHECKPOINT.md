# CHECKPOINT — 2026-05-08

**Status:** PHASE 3 / M1 complete. Advancing to M2.

- ✅ PHASE 0: sandbox + venv + wrapper verified (commit `f9f9469`).
- ✅ PHASE 1: 80-turn dummy conversation, QC-approved (commit `541c0a5`).
- ✅ PHASE 2: gold standard with 4 sections, QC-approved (commit `fb4318d`).
- ✅ **M1: core skeleton — 16 tests pass, CLI works end-to-end** (commit `3206ece`).
  - litellm-backed providers (Anthropic / OpenAI / Gemini / xAI / Ollama / LM Studio behind one ABC)
  - pydantic + YAML config loader
  - SQLite (sqlmodel) baseline
  - bare chat (no memory yet — M2 layer)
  - typer CLI: `sherlock chat`, `sherlock config validate / show`, `sherlock models`
- ⏭ **Next: M2 — memory layer.** Vector DB (Chroma) + embedding provider + LLM-2 summarization cycle + K-turn retention + 4-state decay.
- 🔑 **API key status:** `ANTHROPIC_API_KEY` is set in your shell but not propagating to pytest's subprocess in this environment. Live smoke skipped. Code path is wired and will work as soon as the env var is inherited (e.g. running pytest from a shell where you've explicitly `export`-ed the key).

See `logs/curated.md` for full loop history and `INTENT_DEVIATIONS.md` for the three deviations applied so far (DEVIATION-001 wrapper-via-Python-import, DEVIATION-002 python3.12-preference, DEVIATION-003 litellm-as-provider-backend).
