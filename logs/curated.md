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
