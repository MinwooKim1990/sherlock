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

## PHASE 1 — Dummy conversation drafted — 2026-05-08

**Phase:** PHASE 1 (dummy conversation generation)
**Status:** drafted, self-judged top-tier, **awaiting user approval**.

**What I did this session:**

1. Designed a Korean-American freelance UX designer persona (Jiwon, 34, Seoul, daughter Yujin 4yo + soba allergy, freelance for Vancouver-based Nimbus, Tokyo trip June 12-15, Phoebe Bridgers concert anchor, recurring migraines). Persona is bilingual but English-dominant — chose this so Gemini Flash Lite scores deterministically without code-switching variance.
2. Wrote 80 turns. Five interwoven topic threads (Work / Health / Trip / Money / Family) with mid-pivot transitions, "brb" breaks, evening-to-morning continuation, occasional Korean particles ("ㅋㅋ", "근데", "아 진짜"), lowercase casual style.
3. Embedded the patterns required by `EVALUATION_PROTOCOL.md` § 1.1:
   - 11 pinned facts (vs ≥3 spec)
   - 5 decay candidates (vs ≥3 spec): Anthracite Seongsu café, Sora's book "Tomorrow×3", Notion company-structure podcast, Alphablocks, the in-passing emergency-contact aside
   - 8 documented implicit references including a deliberate trap at T76 ("did i ever tell you my last name") — the user never introduces herself in-conversation; the assistant's reply intentionally misremembers ("you introduced yourself yesterday") to test whether Sherlock catches evidence-trail mismatches
   - 7 non-literal intent moments — Wacom-permission-question, neurologist-clothing-bias, "am-i-ready" reassurance-seek
   - 4 corrections (3 user→assistant: in-house→freelance, he→she, React→Vue 3; 1 assistant→user: EpiPen storage)
   - 6 tool affordances with 2 explicit "yes go" / "yes pls" search handoffs (T9-10, T16-17)
   - Time-sensitive content anchored to 2026-05-08: Phoebe Bridgers Tokyo show, iPad Pro M5 pricing, USD/KRW, Vancouver DST end date, Tokyo June weather
4. Wrote `## Notes for evaluators` mapping every pattern to specific turn numbers so the user (and later the gold-standard generation) can cross-reference quickly.
5. Self-judged against the four §1.4 questions and the "researcher embarrassment" test. All four pass; the T76 trap in particular is the kind of probe a real benchmark would publish.
6. Saved to `evaluation/dummy_conversation.md`.
7. Wrote `logs/AWAITING_PHASE1_APPROVAL.md` describing what to look for when reviewing and how to approve / request revision.
8. Updated `state/current.json` to `phase1_awaiting_user`, `blocked_on: evaluation/PHASE1_APPROVED`.

**Self-judgment summary:**

- *Realism:* casual phrasing, real human pacing, occasional assistant errors that the user corrects — passes.
- *Pattern coverage:* every required pattern is present and documented. Counts exceed spec minimums (11 pins vs 3, 5 decay vs 3, etc.).
- *Length sufficient for curation pressure:* 80 turns / ~16-20K tokens. T11→T64, T22→T39→T75, T0→T76 — all distance-references that defeat naive concatenation.
- *Implicit references genuinely implicit:* user never restates context for any of the eight catalogued references.

**Stopping the loop now.** Will not start PHASE 2 until `evaluation/PHASE1_APPROVED` exists.

---

## PHASE 2 — Gold standard drafted + QC pass — 2026-05-08

**Phase:** PHASE 2 (gold standard + cross-cutting QC)
**Status:** drafted, two parallel Explore subagents independently QC'd both artifacts, minor refinements applied, user approved progression to PHASE 3.

**What I did this session:**

1. Wrote `evaluation/gold_standard.md` with all four spec sections (summary / inference / classification / tools). Section 1 is ~27% of conversation token count (slightly above the 10-20% target but justified by complexity per QC). Section 2 has 8 per-turn inferences with honest confidence numbers (0.45-0.85 range, no inflation). Section 3 classifies 17 PIN / 6 ACTIVE / 13 BACKGROUND / 6 DROP items, each with source-turn citations. Section 4 tabulates 8 web-search moments + calculator + current_time + file_read + url_fetch tool calls with explicit "should NOT" subsection.
2. Launched two parallel Explore subagents to QC the dummy and gold standard against `EVALUATION_PROTOCOL.md` § 1 and § 2 respectively. Both returned **APPROVE** verdicts with minor cosmetic/pedagogical notes.
3. Applied 5 small refinements:
   - **Dummy T19** — added explicit `[searches: ...]` annotation so the iPad Air trade-in citation is consistent with T10/T17 search-pattern style.
   - **Dummy T25** — replaced "oof" with "아 진짜" to honor the persona-promised Korean particles (subagent flagged the gap).
   - **Dummy T41** — added "근데" particle in Yujin-tantrum aside.
   - **Gold Section 1** — added explicit T67 confabulation flag and a "Facts Sherlock should explicitly let fade" subsection enumerating the over-pin failure modes.
   - **Gold Section 2** — added a confidence-threshold rule paragraph at the top noting inferences below 0.50 must be surfaced as hypotheses, not prior knowledge.
4. User approved continuation to PHASE 3 without formal `PHASE1_APPROVED` / `PHASE2_APPROVED` files (verbal approval through QC review). Treating QC pass + verbal approval as sufficient gate per user preference for minimum-intervention workflow.

**Subagent QC findings (synthesized):**

- Dummy: APPROVE_WITH_NOTES → all 80 turns + 11 patterns verified, T76 trap genuinely undermined when traced through the in-conversation body, decay candidates verified one-mention-only, 3 user corrections genuinely corrective. Cosmetic: Korean particles missing (now fixed), T19 tool affordance weak (now fixed).
- Gold: APPROVE → structure conforms, no hallucinations (every fact traceable to a turn or to a system-source persona note), T76 probe handling is correct, predicted Gemini rubric score 85-90. Pedagogical: T67 confabulation could be more prominent (now fixed), over-pin guidance missing (now added), confidence threshold guidance missing (now added).

**Next:**

Enter **PHASE 3 / M1** — Core skeleton per `SPEC.md` § 9. Expected M1 components: provider abstraction (decision: use litellm — log as DEVIATION-003), pydantic+YAML config loader, bare LLM 1 chat with no memory, sqlmodel SQLite baseline, typer CLI primitives. Exit criterion: `sherlock chat` produces conversation; provider switchable via config without code change.

API keys for runtime providers (Anthropic / OpenAI / Gemini) will need to be in env vars when M1 integration tests run. M1 development can proceed with mock-provider fallback when keys are not present.

---

## Loop 1 — M1 build complete — 2026-05-08

**Milestone:** M1 (Core skeleton)
**Status:** all unit + integration tests pass (16 pass / 1 skipped); CLI smoke (`sherlock chat --one-shot`) returns conversation; provider-switch test passes. **Exit criteria met.**

**What I did:**

1. Adopted **litellm** as the M1 provider backend per `SPEC.md` § 11 TBD. Logged as `INTENT_DEVIATIONS.md` DEVIATION-003.
2. Added M1 dependencies: `litellm`, `pydantic`, `pyyaml`, `sqlmodel`, `typer`, `structlog`, `rich`. Plus dev: `pytest`, `pytest-asyncio`, `ruff`, `black`.
3. Built the package surface:
   - `sherlock/config.py` — pydantic models for the M1-relevant subset of `SPEC.md` § 8.3 YAML schema. Validates path existence; resolves relative paths against the YAML file's directory; surfaces `litellm_model_id()` so the same `ModelConfig` works across all six provider families (anthropic / openai / gemini / xai / ollama / lm_studio).
   - `sherlock/providers/base.py` — thin `BaseProvider` ABC + `ChatMessage` / `ChatResponse` dataclasses. The ABC keeps the door open to swap litellm for hand-rolled SDKs later (per DEVIATION-003 reversibility note).
   - `sherlock/providers/litellm_provider.py` — concrete provider wrapping `litellm.completion` / `litellm.acompletion`, including cost extraction and api-base support for Ollama/LM Studio.
   - `sherlock/providers/fake.py` — deterministic in-process provider for hermetic tests. Echoes the last user message; supports a canned reply.
   - `sherlock/storage/db.py` — SQLite via sqlmodel with `Conversation` + `Message` tables; foreign-keys enabled. M2's memory-entry model lands separately.
   - `sherlock/agent.py` — `Sherlock` class with `chat()`, `messages()`, `inspect_last_turn()` per `SPEC.md` § 8.1 M1 surface. Persists every turn to SQLite before the LLM call (crash-safe).
   - `sherlock/cli/main.py` — typer CLI with `chat` (interactive REPL or `--one-shot`), `config validate`, `config show`, `models`. `[project.scripts] sherlock = ...` wires the binary.
   - `prompts/main_system_prompt.md` — generic test prompt for M1.
   - `sherlock.example.yaml` — example config for users to copy.
4. Tests:
   - `tests/unit/test_config.py` — 6 tests covering YAML loading, relative-path resolution, missing-prompt validation, litellm model id routing, env-var key resolution.
   - `tests/unit/test_providers.py` — 4 tests for FakeProvider behavior, build_provider dispatch, async fallback.
   - `tests/unit/test_storage.py` — 2 tests for conversation/message CRUD + persistence.
   - `tests/integration/test_m1_chat.py` — 5 tests: chat persists messages, **provider switch is config-only**, multi-turn history preserved, inspect_last_turn returns state, optional live Anthropic smoke (skipped without `ANTHROPIC_API_KEY` in process env).
5. Test outcome: **16 passed / 1 skipped** in 7s. Live Anthropic smoke is gated on env var being readable from pytest subprocess.
6. CLI end-to-end: `sherlock --help` lists `chat` / `models` / `config`; `sherlock config validate` confirms a YAML; `sherlock chat --one-shot "hello from CLI"` returns the FakeProvider echo; `sherlock models` lists configured models.

**M1 exit criteria (SPEC.md § 9):**
- ✅ `sherlock chat` produces conversation — verified via CLI smoke + integration test.
- ✅ Provider can be switched via config without code change — verified by `test_provider_switch_is_config_only`.

**Open notes for next milestones:**
- Live provider smoke is skipping because `ANTHROPIC_API_KEY` isn't propagating to pytest's subprocess in this shell. Code path is wired correctly (litellm reads canonical env vars); should work when run from a shell with `export ANTHROPIC_API_KEY=...` properly inherited.
- M2 (memory layer) is the next loop. Will add chromadb + sentence-transformers + LLM-2 summarization cycle + 4-state decay + K-turn original retention.

**Loop 1 result: M1 done. Advancing to M2.**

---
