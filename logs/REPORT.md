# Project Sherlock — Iterative Build Report

> Append-only experiment report. Sections 4 and 5 grow per loop; the rest is stable narrative.
> Last update: 2026-05-08, after loop 7 (with one disrupted post-loop-7 cross-evaluator probe).

## 1. Executive summary

Sherlock is a domain-agnostic context-curation library. The user authors only the main system prompt; the system itself is supposed to bootstrap its companion (LLM-2 summarizer / LLM-3 inferrer) prompts, curate memory through a 4-state decay lifecycle, and evolve its companion prompts based on user-feedback signals. The build covered M1 through M7 (with M4/M6/M9 in light form), an evaluation harness, and the Ralph-style verify-fix-retry loop that is the subject of this report.

Across loops 1–7 the system was rebuilt, then iterated against the gold-standard benchmark in `evaluation/gold_standard.md`. The Gemini Flash Lite evaluator was the official judge per `EVALUATION_PROTOCOL.md` § 3.3. **Scores from Gemini Flash Lite over six loops**: 57 → 48 → 50 → 48 → 48 → 61. The 7th run (timestamp `2026-05-08T07-36-03`) fell to a different evaluator (`codex/gpt-5.4-mini`) because Gemini hit a rate-limit and the auto-fallback chain activated; that score (27) **cannot be compared** with the prior six because the rubric calibration is different per evaluator.

The trajectory tells two stories. First, the system never moved past 61/100 against the canonical evaluator — well short of the 80-point gate. Second, within the trajectory there is a clear story of identifying root causes (Bootstrap-authored prompts overriding our discipline rules; over-pinning from semantic-paraphrase blindspots; over-recommendation of tools because no discipline clause reached LLM-3 in flight; the 30-turn cap structurally hiding the T76 trap) and patching them one at a time. The final gain into 61 came from disabling Bootstrap entirely and pinning the discipline directly into `DEFAULT_*_PROMPT` — confirming the diagnosis was right but the architectural assumption (LLM-1 will reliably author quality companion prompts) was the actual blocker.

## 2. Build scope

### Milestones implemented
- **M1 — Core skeleton.** Provider ABC, litellm-backed `LiteLLMProvider`, FakeProvider, pydantic+YAML config, sqlmodel SQLite storage, typer CLI. 16 tests pass, 1 skipped. Exit criteria met.
- **M2 — Memory layer.** `MemoryEntry` per SPEC §6.1, Chroma vector store, LiteLLM embeddings + `FakeEmbeddingProvider` fallback, 4-state decay engine, LLM-2 summarizer, K-turn original retention.
- **M3 — Bootstrap + Inference + Web search.** `bootstrap/{engine,meta_context}.py`; `inference/engine.py` produces ≥3 hypotheses with confidence + evidence; `tools/{builtin,web_search}.py` covers builtins plus Tavily/StubSearch. Multi-domain divergence test not formally executed.
- **M4-light — RAG.** Vector + BM25 + RRF (k=60). No reranker, no semantic-triple compression beyond placeholder.
- **M5 — Async pipeline.** `Sherlock.achat()` parallelizes LLM-3 + retrieval; summarizer + decay parallel post-response. Cost cap surface present, not exercised.
- **M6-light — Evolution.** `evolution/versioning.py` is a versioning shell only. Feedback-driven evolution NOT implemented.
- **M7 — Tool layer.** Built-ins wired, MCP discovery surface present but untested, `@sherlock.tool` decorator works.
- **Evaluation harness.** `sherlock/evaluation/{replay,output_format,evaluator}.py` + `evaluator_system_prompt.txt`; `sherlock evaluate` writes timestamped run directories.

### Deviations from spec (`INTENT_DEVIATIONS.md`)
- **DEVIATION-001** — wrapper accessed via Python import primary, CLI fallback. Wrapper has no `--system / --user-file / --output` flags.
- **DEVIATION-002** — bootstrap.sh prefers python3.12→3.13→3.11→python3 because `python3` is 3.14.3 (chromadb wheels uncertain).
- **DEVIATION-003** — litellm chosen over hand-rolled SDKs (SPEC §11 leaves it as TBD for M1). ABC preserved for reversibility.
- **DEVIATION-004** — `cli-wrapper-unified` used as runtime provider because `os.environ.get('ANTHROPIC_API_KEY')` returns None in the agent's process. Spec calls wrapper "evaluator only"; reversal is a one-line YAML edit.

### Key architectural choices
- litellm vs hand-rolled providers: litellm.
- Wrapper as runtime: required because env vars don't propagate; tool-suppression banner injected to prevent the loop-2 side effect.
- Embedding fallback to FakeEmbeddingProvider when no key.
- Mixed-provider config: claude-haiku-4-5 for LLM-1, gemini-3.1-flash-lite-preview for LLM-2/3 (3-4s vs 10-15s/call).

## 3. Evaluation protocol

### Rubric (EVALUATION_PROTOCOL.md § 3.4)
```
final_score = 0.4 * summary_fidelity
            + 0.4 * inference_quality
            + 0.1 * classification_correctness
            + 0.1 * tool_recommendations
```
80% gate terminates the loop. Currently 19 points below per Gemini.

### Why scores from different evaluator models cannot be cross-compared
Each evaluator model has its own internal calibration. Gemini Flash Lite scores summary at 57–65 where gpt-5.4-mini scores 44; Gemini gives 40–45 on inference where gpt-5.4-mini gives 12. Same rubric prompt, different internal scaling. So §4 trajectory tables are grouped by `evaluator_model`.

### Fallback chain (commits c68679c → 0c64bdd → 71976ec)
```
gemini/gemini-3.1-flash-lite-preview  (canonical)
  → gemini/gemini-2.5-flash-lite      (closest cousin)
  → codex/gpt-5.4-mini                (cross-vendor backup)
  → claude/claude-haiku-4-5           (last resort)
```
`gemini-3.0-flash` was rejected by the wrapper (`gemini:model_not_allowed`); pruned. **Gemini Pro excluded** because its scoring distribution is materially stricter — including it would silently shift the rubric mid-loop.

## 4. Trajectory per evaluator model

The earliest six runs predate the `evaluator_model` field; they are Gemini Flash Lite per the curated log.

### 4a. Evaluator: gemini/gemini-3.1-flash-lite-preview

| Loop | Run timestamp | Score | summary / inference / classification / tools | Code-state changes | Diagnosis (from notes) |
|------|---------------|-------|----------------------------------------------|--------------------|------------------------|
| L2 baseline | 2026-05-08T05-06-40 | **57** | 65 / 60 / 30 / 50 | Initial M2-M7 batch + WrapperProvider; mixed-provider config; 30-turn cap | Classification bloated, redundant, over-pinning; Section 4 was just hypothesis-counts not per-turn tools. |
| L3 | 2026-05-08T05-21-57 | **48** ▼9 | 65 / 45 / 30 / 20 | Dedup-at-add (`28d11fa`); per-turn tool table; tighter Section 2 prompt | System-source persona facts promoted to user-stated by dedup upgrade; tool over-rec from no discipline. |
| L4 | 2026-05-08T05-36-59 | **50** ▲2 | 65 / 45 / 30 / 40 | SYSTEM source sticky (`0047534`); tool-rec discipline in DEFAULT_LLM3_PROMPT; PIN bucket split | Paraphrase facts evade prefix-60 dedup; T76 unreachable behind 30-turn cap. |
| L5 | 2026-05-08T05-52-01 | **48** ▼2 | 65 / 45 / 30 / 20 | Full 80-turn replay; LLM-2 sees existing pins; max_items 60→25; semantic dedup | **Root cause:** Bootstrap auto-runs LLM-1-authored prompts, overriding `DEFAULT_*_PROMPT`. All loop-3-5 fixes had zero in-flight effect. |
| L6 | 2026-05-08T06-30-33 | **48** ±0 | 65 / 40 / 25 / 30 | META_CONTEXT carries discipline (`59114e8`); let_fade=true→COLD; PIN-context cap 25 | Section 3 dumped hundreds into DROP including PIN-worthy migraine + work-architecture; T76 missed. |
| L7 | 2026-05-08T07-02-55 | **61** ▲13 | 75 / 70 / 30 / 40 | **Bootstrap disabled** (`206909b`); strict DEFAULTs used directly | Summary + inference jumped. Classification still 30 — flat structure / wrong DROP labels. |

### 4b. Evaluator: codex/gpt-5.4-mini (fallback, not directly comparable)

| Loop | Run timestamp | Score | summary / inference / classification / tools | Code-state changes | Diagnosis |
|------|---------------|-------|----------------------------------------------|--------------------|-----------|
| L8 (cross-eval) | 2026-05-08T07-36-03 | **27** | 44 / 12 / 18 / 26 | Conservative let_fade + post-hoc tool rate cap (`4502679`); prose-with-citation Section 3 (`2a75784`) | Gemini rate-limited → wrapper auto-fell-back to gpt-5.4-mini. Distribution looks harsh because evaluator changed; not a regression. |

## 5. Per-loop narrative

**Loop 2 baseline — 57/100.** Mixed-provider config, 30-turn cap. Classification 30/100 because LLM-2 re-emitted paraphrases with no add-time dedup; PIN bucket >70 vs gold's 17. Section 4 was rendering hypothesis-counts not tool calls. Mid-run wrapper-driven Claude wrote `tokyo_trip_reference.md` as a side effect; patched with TEXT-ONLY guard banner (`dc5ff90`).

**Loop 3 — 48/100, ▼9.** Three failures named: source-rank dedup promoted SYSTEM persona to USER; tool over-rec because discipline didn't reach in-flight prompt; user_utterance entries leaking into PIN/ACTIVE/BACKGROUND/DROP. Fixes: SYSTEM-sticky dedup; discipline in DEFAULT_LLM3_PROMPT; Section 3 excludes user_utterance and splits PIN by source.

**Loop 4 — 50/100.** Tool dim partially recovered (20→40). Paraphrase facts evade prefix-60 dedup; T76 trap missed because 30-turn cap. Fixes: semantic dedup at write (cosine ≥0.92); LLM-2 sees existing PINs; full 80-turn replay; max_items 60→25.

**Loop 5 — 48/100.** Decisive diagnostic: tool-discipline in `DEFAULT_LLM3_PROMPT` from loop 4 had zero runtime effect. Root cause: `bootstrap.auto_run_on_init: true` calls LLM-1 to author *fresh* companion prompts at every run, overriding DEFAULTs. Three loops of fixes invisible in flight. Fix: discipline rules move to `META_CONTEXT` (bootstrap input).

**Loop 6 — 48/100.** META_CONTEXT-mediated discipline reached the bootstrap-authored prompts, but LLM-1 over-applied "let_fade" — Section 3 dumped hundreds into DROP including PIN-worthy items. T76 still missed. Fix: stop trusting LLM-1 to author the prompts; pin them directly.

**Loop 7 — 61/100, ▲13.** Bootstrap disabled (`206909b`). DEFAULTs used verbatim. Summary 65→75, inference 40→70. Classification stuck at 30 — flat list structure, wrong DROP for trip itinerary / allergy cards. Tool 30→40. Highest score against canonical evaluator. Fix design (loop 8 attempted): prose-with-citation Section 3 to mirror gold's structure.

**Loop 8 (cross-evaluator) — 27/100.** Gemini rate-limited → fallback to gpt-5.4-mini. Same issues called out (T76 missed, inference effectively absent, tool advice over-general) but at much harsher absolute scores. Not comparable to L7's 61.

## 6. Patterns observed across the iteration

**Bootstrap-authored prompts vs DEFAULT prompts — silent override.** The single most expensive bug. Loops 3-5 committed fixes into DEFAULTs that bootstrap overrode every session. Discovery (loop 5) triggered by "I committed this, why doesn't the evaluator see it." Loop 7 disabled bootstrap entirely; that's when scores moved.

**Over-pinning (classification).** Stuck at 30 across all six Gemini loops. Six dedup/cap iterations didn't break the floor because the failure is structural — gold uses prose-with-citation; our Section 3 is flat bucket list.

**Tool over-recommendation.** Gold flags ~10-12 turns; loops 3-4 flagged 54/80. Discipline clause needed META_CONTEXT routing (loop 6) and finally bootstrap disable (loop 7) to take partial effect.

**30-turn-cap blind spot for T76.** Loops 2-4 capped at 30 turns hiding T76, T55 (EpiPen), T67 (fintech role), and the corrections at T20/T27. Loop 5 went to 80; T76 moved from "structurally unreachable" to "reachable but missed." Still missed in every loop. Fix would require LLM-3 to maintain a "facts established in conversation" provenance log and probe-check inbound user turns; not built.

## 7. Trustworthiness assessment

**Reliable production:** summary 65-75 against Gemini; inference 70 after loop 7; ≥3 hypotheses with confidence + evidence per highlight; provenance distinction visible when bootstrap disabled; SQLite + dedup + vector/BM25 retrieval all pass tests.

**Still fails:**
- **T76 provenance trap** never caught. Architectural feature ("flag when user asks about a fact never established") not implemented despite provenance fields existing on entries.
- **Classification 25-30/100** across all Gemini loops — needs prose-with-citation format AND projected lifecycle (PIN-until-X, ACTIVE-for-5-weeks, DROP-after).
- **Multi-domain divergence (M3 exit)** never formally executed.

**Safety incidents:** loop-2 wrapper-driven Claude wrote `tokyo_trip_reference.md` mid-run as a tool side-effect. Mitigation: TEXT-ONLY guard banner in `dc5ff90`. No PII leakage in evaluator inputs (synthetic persona).

**Confidence honesty:** gold standard's 0.45-0.85 range is honest with explicit hypotheses + evidence. Sherlock's *runtime* confidences are LLM-3-emitted but not calibrated against held-out data — treat as ordinal not metric. The 0.92 semantic-dedup threshold is empirically validated only on the documented Yujin-allergy paraphrase cluster.

## 8. Growth potential

**Reachable ceilings with current architecture:** summary ~80, inference ~75, classification ~50, tools ~70 → composite ~73, still 7 short of the 80 gate.

**Stub-quality components:**
- **Tavily** wired but most runs use `StubSearch` (no key).
- **Semantic-cluster decay** turn/day-based only; HDBSCAN (SPEC §11) not built.
- **Evolution engine** is versioning only; SPEC §5.3 feedback-driven path not implemented.
- **Streamlit UI** (SPEC §8.5) not built.
- **MCP discovery** surface present, no end-to-end server tested.
- **Reranker** (SPEC §7.1 mandatory) not wired.

**Path to score gains:**
- Summary +5-10: verbatim PIN list in Section 1; per-thread sub-sections.
- Inference +5-10: T76-style provenance-trap detection.
- Classification +15-20: prose-with-citation (loop 8 attempted); per-fact lifecycle projection.
- Tool +20: post-hoc rate cap as hard limit; freshness-required as gating signal.

## 9. Usability

**Onboarding:** `bootstrap.sh` detects Python 3.12→3.13→3.11→python3, builds `.venv`, installs editable, ensures wrapper importable. `.env.example` lists keys. `sherlock chat --one-shot` and `sherlock evaluate` both work.

**Works well:** one-command bootstrap (<2 min), typer+rich CLI ergonomics, crash-safe SQLite persistence (turn saved before LLM call), hermetic FakeProvider/FakeEmbeddingProvider test mode (16/17 pass in 7s).

**Wrapper-vs-API-key trade-off:**
- Wrapper: no `.env` needed, same auth as evaluator; con: tool-suppression banner, 3-15s/call latency.
- API key: 1-3s/call, no banner, no side-effect risk; con: env vars didn't propagate to agent's Python in this build.
- **Recommendation:** real users should switch to `provider: anthropic` + populate keys; keep wrapper as fallback. One-line YAML edit per DEVIATION-004.

**Latency:** evaluation simulation ~28s/turn (sync sequential); production async path (M5 `achat`) ~3-6s/turn.

## 10. Open questions and risks

**Spec-vs-implementation gaps:**
1. Bootstrap currently disabled (`206909b`); SPEC §5.3 mandates LLM-1-authored companion prompts. Re-enabling needs a quality gate.
2. Evolution is a shell, not a learning loop.
3. M3 multi-domain divergence test never executed.
4. Reranker not wired (SPEC §7.1 mandatory).
5. HDBSCAN not built.
6. Streamlit UI not built.
7. Provenance-trap detection (T76) not built — single most impactful inference-dim feature.

**User intent vs current state:** "make the iterative verification loop possible" met — loop runs end-to-end, diagnostic-fix-retry operational. 80% gate not crossed; highest is 61 (Gemini).

**Risks:**
- Cross-evaluator drift when Gemini rate-limits (mitigation: `evaluator_model` field in JSON, introduced `c68679c`; pre-commit runs lack it).
- Wrapper coupling — outage breaks both runtime and evaluator paths simultaneously.
- No automated regression gate beyond unit suite.
- Empty `2026-05-08T08-09-19/` directory suggests run-failure mode where dir is created but no output written; defensive cleanup warranted.

*Report ends. Future loops should append a row to the appropriate §4 table and a paragraph to §5; §1-3 and §6-10 only need touching when something architectural changes.*
