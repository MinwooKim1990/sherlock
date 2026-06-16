# INTENT_DEVIATIONS — Append-only log of where implementation diverged from spec

> Version: v0.3 · 2026-05-08
> Empty by default. Filled by the coding agent during build whenever it makes a choice that conflicts with, omits, or extends the specification.
>
> The user reads this periodically to confirm the deviations were reasonable. The user may revise the spec or instruct the agent to revert deviations.

---

## How to write an entry

Append a section like this whenever a deviation occurs. **Do not delete or rewrite earlier entries.**

```markdown
## YYYY-MM-DD — <short title>

**Loop / milestone:** loop 47, M5
**Spec reference:** SPEC.md § X.Y, or AGENTS_AND_LOOP.md § Z, or "no explicit reference"
**What the spec says (or implies):**
<one sentence>

**What I did instead:**
<one sentence>

**Why:**
<paragraph: what made the spec's approach impossible / suboptimal / ambiguous;
what alternative I considered; why I chose this one>

**Reversibility:**
<easy to revert / hard to revert / impossible without rebuild>

**User action requested (optional):**
<if you want the user to weigh in, state the question explicitly>
```

---

## Trigger list — append an entry when any of these happen

- The agent picks one option from the TBD list in `SPEC.md § 11` without user input
- A library or tool the spec recommends is unavailable, and the agent substitutes another
- A milestone Exit criterion appears unmeetable as written, and the agent revises it
- The cli-wrapper-unified does not support `gemini-3.1-flash-lite-preview` exactly and a near-id is used
- A spec'd component is split into multiple, or two are merged into one
- A default config value is changed because the spec value caused issues
- Any time the agent feels "the user might disagree with this; better document it"

When in doubt, append. An entry is cheap. A silent deviation that surfaces three milestones later is expensive.

---

## Entries

## 2026-05-08 — DEVIATION-001: cli-wrapper-unified accessed via Python import (primary), CLI subprocess (fallback)

**Loop / milestone:** PHASE 0 setup
**Spec reference:** EVALUATION_PROTOCOL.md § 3.3 ("called through the **cli-wrapper-unified** tool"), OPERATIONS.md § 1.4 ("Run a tiny test invocation calling …")
**What the spec says (or implies):**
The spec phrases the wrapper as a CLI tool (with template flags `--model / --system / --user-file / --output`) and the verification step uses `--help`.

**What I did instead:**
Verified both access paths work and **chose Python import (`from unified_cli import create`) as the primary** with CLI subprocess as fallback. Installed the wrapper into the sherlock venv via `pip install -e ~/Desktop/cli-wrapper-unified`.

**Why:**
The wrapper README explicitly advertises both modes ("CLI you can shell into, AND Python package you can import — same code, same state"). The Python path avoids subprocess spawn cost (~50–460 ms per call per the wrapper README), produces a typed `Response` object (`r.text`, `r.session_id`, `r.usage`), and is easier to error-handle inside Sherlock's evaluation loop. The CLI form remains documented in `logs/curated.md` and is used as a fallback if the import route ever fails.

The spec template's flag set (`--system <file> --user-file <file> --output <json>`) is **not present** in `unified-cli chat`. Available flags: `-m/--model`, `--stream`, `--no-web-search`, `--terse`, `--cwd`, `-r/-c/--new`. To inject the evaluator system prompt, the agent will prepend the rubric inline to the user message at PHASE 3 evaluation time. This is not a meaningful loss — Gemini Flash Lite scores deterministically with the rubric in-band.

**Reversibility:**
Easy. The agent can switch to subprocess CLI invocation at any time; the import path is opt-in.

**User action requested (optional):**
None. Confirm if you'd prefer the subprocess-only mode and I will switch.

## 2026-05-08 — DEVIATION-002: bootstrap.sh prefers python3.12 over python3.11

**Loop / milestone:** PHASE 0 setup
**Spec reference:** OPERATIONS.md § 1.3 (the template references `python3.11` and `python3`)
**What the spec says (or implies):**
The bootstrap-script template detects Python via `python3.11` then `python3`.

**What I did instead:**
Detection order is `python3.12 → python3.13 → python3.11 → python3`.

**Why:**
`python3` on this machine is 3.14.3 (cutting-edge). chromadb / sentence-transformers / sqlmodel may not yet have wheels for 3.14 in 2026-05; falling back to whatever `python3` is is therefore risky. `python3.11` is missing on this host. **3.12 is the most stable "3.11+" target** with broad wheel coverage, so it is preferred. 3.13 next, 3.11 next, `python3` last. Bootstrap actually chose **3.12.13** on this run.

**Reversibility:**
Trivial — single edit to `bootstrap.sh` if the user prefers a different priority.

**User action requested (optional):**
None.

## 2026-05-08 — DEVIATION-003: M1 provider abstraction uses `litellm` rather than hand-rolled per-provider SDKs

**Loop / milestone:** loop 1, M1
**Spec reference:** SPEC.md § 9 M1 ("consider `litellm` for the unified path"), SPEC.md § 11 (open questions: "litellm adoption vs hand-rolled provider abstraction (decide in M1)")
**What the spec says (or implies):**
The spec explicitly leaves this as a TBD for M1: list six providers (Anthropic, OpenAI, Gemini, xAI, Ollama, LM Studio) with `litellm` as a "consider" option. § 10.7 also mitigates provider-API-drift risk via litellm.

**What I did instead:**
Adopted `litellm` as the M1 provider abstraction. All six target providers go through `litellm.acompletion` / `litellm.completion`. `sherlock/providers/base.py` defines a thin ABC; the concrete implementation `LiteLLMProvider` is the only runtime provider for M1. A separate `FakeProvider` exists for unit tests so the test suite is hermetic.

**Why:**
- Single dependency covers all six target providers in one shot, including model-list discovery for several of them.
- Built-in retry/fallback chain matches SPEC § 10.7 mitigation directly.
- Async support lines up with M5's parallel pipeline requirement without re-architecture.
- Drastically smaller surface to maintain than six bespoke wrappers.
- The ABC is preserved so a hand-rolled provider can be slotted in later if litellm causes pain.

**Reversibility:**
Easy to medium. The ABC means a hand-rolled `AnthropicProvider`, `OpenAIProvider`, etc. can replace `LiteLLMProvider` without touching call sites. Cost is one provider class per target.

**User action requested (optional):**
None.

## 2026-05-08 — DEVIATION-004: cli-wrapper-unified used as a runtime provider when no API keys are available

**Loop / milestone:** loop 2, post-M3 evaluation prep
**Spec reference:** AGENTS_AND_LOOP.md § 3.1 ("Do not call the cli-wrapper-unified for purposes other than evaluation. It is a dedicated evaluator, not a general LLM."); EVALUATION_PROTOCOL.md § 3
**What the spec says (or implies):**
The wrapper is for the Gemini-Flash-Lite evaluator only. Runtime providers go through `litellm` with API keys in env vars.

**What I did instead:**
Added `WrapperProvider` (`sherlock/providers/wrapper_provider.py`) that wraps `unified_cli.create(provider, model=...)` with the same `BaseProvider` interface. `sherlock.live.yaml` is configured to use it for the main / summary / inference roles by setting `provider: wrapper` and choosing a wrapper-supported model (`claude-haiku-4-5`, `gpt-5.4-mini`, `gemini-3.1-flash-lite-preview`). The Gemini-Flash-Lite evaluator continues to use the same wrapper via `sherlock.evaluation.evaluator.GeminiEvaluator`; the two paths are kept logically separate (different config sections, different call sites).

**Why:**
- The user's environment has subscription auth set up via the wrapper but no provider API keys readable from Python (`os.environ.get('ANTHROPIC_API_KEY')` returns `None` even though `env` shows it set in the interactive shell — exports do not propagate to the agent's process).
- The user explicitly directed "일단 모두 진행해서 반복 검증 루프까지 가능하게 해" ("just proceed, make the iterative verification loop possible") in this loop — the runtime-provider blocker would otherwise force an SOS that contradicts that directive.
- The wrapper's subscription auth is functionally equivalent to API-key auth for our purposes (chat completion). The spec's concern with the wrapper as runtime is presumably (a) don't accidentally route eval calls through the same instance and (b) don't depend on the wrapper for the productized library. (a) is preserved by the separate code paths; (b) is a release-time concern, not a build-time concern.
- The sub-second-spawn-overhead-per-call cost is acceptable for the 80-turn evaluation runs (~5-15 minutes total).

**Reversibility:**
Trivial — switching `models.main.provider` from `wrapper` back to `anthropic` (or any litellm-supported) in YAML restores the spec'd path. The `WrapperProvider` class can be deleted at any time without touching the rest of the code.

**User action requested (optional):**
If you'd prefer the strict spec path, populate `.env` with `ANTHROPIC_API_KEY` (and optionally `OPENAI_API_KEY`, `TAVILY_API_KEY`) and switch `sherlock.live.yaml`'s providers back to `anthropic` etc. Sherlock auto-loads `.env` via python-dotenv. The wrapper-as-runtime path remains available either way.

## 2026-05-08 — DEVIATION-005: Authoritative evaluator is a Claude-class subagent, not gemini-flash-lite

**Loop / milestone:** post-loop-9 in the Ralph cycle.
**Spec reference:** EVALUATION_PROTOCOL.md § 3.3 ("evaluator is gemini-3.1-flash-lite-preview … through the cli-wrapper-unified")
**What the spec says (or implies):**
The evaluator is a small fast subscription-funded Gemini model. Its JSON-rubric output is the official score that gates the 80% threshold.

**What I did instead:**
The orchestrator agent (Claude Code class) — or a Claude-class subagent dispatched by the orchestrator — is the authoritative evaluator. The small-model evaluator stays in the codebase as a sanity-check baseline but its score is no longer what drives the Ralph loop's diagnose-fix-retry cycle.

**Why:**
Loops 2-9 made it clear that small-model evaluators (gemini-3.1-flash-lite-preview, gemini-2.5-flash-lite, codex/gpt-5.4-mini) score by surface pattern-match against the gold standard. They lack the conversation-flow understanding, spec knowledge, and intent-tracking that makes a Sherlock-shaped judgment. When the runtime workers (LLM-1/LLM-2/LLM-3) are themselves gpt-5.4-mini or claude-haiku-4-5 class, the evaluator must be CLASS-ABOVE the workers — otherwise there is no headroom for the worker output to shine. **Ralph's whole point is that the orchestrator (with full intent + spec + conversation history) judges the worker output and steers; outsourcing the judgment to a similarly-sized model collapses Ralph into a noise-driven random walk.**

User direction (verbatim, 2026-05-08): "평가자는 너가 서브에이전트로 평가해야지 무슨 작은 모델들한테 평가를 시키냐 지금까지의 루프가 의미 없어져보이네 그러니까 너가 여러번 루프를 도는거지 왜 평가를 실제 사용 모델들을 시켜? 그럼 너가 랄프할 이유가 뭐냐?"

**Effect on prior loops:**
The §4a / §4b trajectory tables in `logs/REPORT.md` (loops 2-9) are demoted to "small-model sanity baseline" status. The score deltas they report still indicate which fixes moved which dimension (signal-in-noise), but they are not the rubric for the 80% gate going forward.

**New evaluator path:**
After each `sherlock evaluate` run, the orchestrator dispatches a subagent with: (a) the gold standard, (b) the candidate `sherlock_output.md`, (c) the rubric prompt, and (d) any spec context the subagent needs. The subagent returns JSON in the same shape (`summary_fidelity / inference_quality / classification_correctness / tool_recommendations / final_score / notes`) plus an `evaluator_model: "claude-orchestrator-subagent"` marker. This becomes the loop's official score in `evaluation/runs/<ts>/score.txt` (overwriting any prior small-model write).

**Reversibility:**
Trivial — to revert, treat the small-model evaluator's score as authoritative again. The `EvaluatorScore.evaluator_model` field already records which path was used, so trajectory analysis can group either way.

**User action requested (optional):**
None — this is the user's explicit direction.

---

## DEVIATION-006 — v0.5.0 production-hardening interpretations

**Date:** 2026-05-12
**Context:** v0.5.0 took the system from MVP to production-usable after two
Opus audits + an external analysis converged on the same defects.

A few decisions deviate from the literal spec / earlier choices:

1. **Hybrid embeddings, fake stays the test default.** The spec implied
   real embeddings throughout. We default `with_callable(embedding=...)`
   to `fake` (hermetic, offline CI) while `sherlock.live.yaml` and
   `test_sherlock.py` default to `local` (fastembed multilingual). This
   keeps the 160-test suite network-free while real usage gets real
   semantic memory. Reversible via the `embedding=` param.

2. **Companion fallback fires `compact` every N turns, but NOT `infer`.**
   The user was emphatic that LLM-1 decides when to infer. So the
   real-usage safety net only auto-compacts (memory must not starve);
   inference stays purely tag-driven. `auto_infer_on_topic_shift` exists
   as an opt-in (default off).

3. **"Safety-critical" pin protection = SYSTEM + USER + persona_summary.**
   The plan said "protect safety-critical/system/user pins." We have no
   reliable safety-critical *classifier* (that path was removed in loop 15
   as overfitting), so we protect by provenance (SYSTEM/USER source +
   persona-summary tag) rather than by a content heuristic. Honest and
   non-overfit.

4. **Background worker is single-threaded + lock-guarded, not a full
   async queue.** "Main fast, background follows" is realised with a
   1-worker ThreadPoolExecutor + RLock + bounded pending-wait. A true
   multi-worker async queue (M5's original ambition) is deferred — the
   single worker is correct and race-free; throughput wasn't the issue.

5. **`add_many` batch-embedding deferred to v0.6.** Compaction persists
   ≤8 facts/turn, so N sequential embeds is a minor cost; bundled with
   the bigger Phase-5 wins it wasn't worth the dedup-batching complexity
   this release.

**Reversibility:** all five are config- or param-gated; none change the
spec's data model or the public API shape.


## DEVIATION-007 — v0.5.1 security & correctness finishing pass

**Date:** 2026-05-29
**Context:** A second precise external review (after the v0.5.0 hardening)
found 5 remaining fitness-for-purpose gaps. All were closed; verified by
operating the system (real local embeddings + true background), not by
pass/fail alone.

1. **Redaction now covers EVERY memory string field, not just `content`.**
   `MemoryStore.add()` redacts `content`, `evidence`, `tags`, and each
   `semantic_triple` element at the single write choke point. This closes
   the leak where LLM-2/LLM-3 placed a secret in evidence/tags/triple — it
   previously persisted verbatim and surfaced via the eval provenance ledger,
   the memory tool's `tags` export, and the SQLite entity index. The
   `[REDACTED:label]` placeholder is JSON-safe, so the `evidence` JSON list
   stays parseable; structural tags (`persona_summary`, `prediction`,
   `freshness,…`) are not secret-shaped so entity/persona logic is unaffected.

2. **Builtin `_url_fetch` is now SSRF-guarded too (defense-in-depth).** It
   was only reachable by trusted internal code (the LLM tag dispatch uses the
   already-guarded `web_search` path), but it now runs `is_safe_url` + a
   redirect re-check — so exposing the builtin registry as an LLM tool can't
   reintroduce a localhost/metadata hole. Gains a `resolver=` param for
   offline testing, mirroring `_default_fetch`.

3. **`hard_delete` cascades the entity index.** It now deletes the row's
   `MemoryEntity` rows alongside the SQLite row + Chroma vector, matching
   `delete_conversation_memories`. Prevents stale-index buildup from repeated
   persona-summary replacement and decay eviction.

4. **`memory_entity` tool uses the persistent index.** It now calls
   `find_by_entities()` (O(matches)) instead of `store.list()` + full scan,
   falling back to a scan only when `conversation_id is None`. Same FORGOTTEN
   filter + entity-pool re-verification as `hybrid.py`.

5. **Package docstring steers to `Sherlock.from_yaml(...)`.** The
   `sherlock/__init__.py` advanced example showed the bare `Sherlock(config)`
   constructor, which skips companion/bootstrap/search wiring (LLM-1 only).
   Now it shows `from_yaml` with a note. (The CLI was already fixed in v0.5.0.)

**Out of scope (noted, not done):** a real-provider + real-judge eval gate —
`ralph_v2 --fake-llm` passing 5/25 is expected (the fake LLM can't infer
intent); genuine quality evidence needs the user's API keys via `make eval`.

**Reversibility:** redaction is gated by `memory.redact_secrets`; the fetch
guard and index changes are behavior-preserving for legitimate inputs; the
docstring is documentation-only. No data-model or public-API changes.


## DEVIATION-008 — v0.6 vision-fidelity + numpy-style packaging

**Date:** 2026-06-08
**Context:** An audit + the user's restated design intent showed the core loop
was wired but partly dormant (LLM-3 rarely fired), plus real bugs and
packaging friction. The user is NOT selling this, so sale-only items
(LICENSE/legal/PyPI marketing) are deliberately out of scope; the goal is a
clean, numpy-style importable library that behaves as designed.

1. **LLM-3 inference is no longer purely tag-driven.** It stays tag-first
   (LLM-1 decides), but a selective auto-trigger (`memory.auto_infer`:
   "smart" default | "off" | "always") fires it on a topic shift / first turn
   so the psychological-inference layer isn't dormant when a vanilla model
   under-emits the tag. Never every-turn (honours "don't burn tokens"). A
   `SHERLOCK_AUTO_INFER` env var overrides for hermetic tests (suite sets "off").
   The LLM-1 protocol prompt's `infer` guidance was rewritten from one vague
   line into principled criteria (vague/implicit intent, hallucination guard,
   reward-hack guard) — no keyword lists (those were removed as overfit).

2. **Forward predictions surface proactively.** `worth_digging` (previously
   generated then discarded) is now persisted; the LLM-1 slot shows the top
   2-3 hypotheses (was top-1) and a dedicated ANTICIPATED DIRECTIONS block that
   RAG-selects the stored prediction/worth-digging matching the CURRENT input —
   so a sudden topic pivot pulls up the matching pre-inference.

3. **Real bug fixes:** the current turn's user input was duplicated in every
   LLM-1 prompt (persisted before slot assembly + appended) — now excluded from
   the K-turn tail by id; the semantic-dedup window scanned the OLDEST 30 rows
   (`rows[-30:]` on a desc query) instead of the most-recent (`rows[:30]`);
   `achat()` reached parity with `chat()` (memory lock, background barrier,
   event probes, no-duplication).

4. **numpy-style packaging:** the embedding default is `"auto"` (real local
   semantic memory when `[embeddings]` is installed, graceful fake fallback +
   warning otherwise; `SHERLOCK_AUTO_EMBEDDING` overrides for tests); web search
   moved to an optional `[search]` extra (no scraping dep in the base install);
   the undeclared private `unified_cli` import is now guarded with a clear,
   actionable error instead of a bare ImportError. `sherlock.evaluation` stays
   in the wheel because it backs the `evaluate`/`replay` CLI commands (not
   dev-only cruft) — the audit's "exclude it" finding was reclassified.

**Out of scope (per the user, not selling):** LICENSE file, copyright/SPDX
headers, PyPI name/metadata, duckduckgo-ToS-as-legal-blocker (handled as the
[search]-optional packaging move instead).

**Reversibility:** auto_infer/auto-embedding are config + env gated;
search/embeddings are optional extras with graceful fallback; the duplication,
dedup, and achat fixes are behavior-correcting. No public-API breakage —
`with_callable`/`from_yaml`/`chat`/`achat` signatures are unchanged except new
defaulted params.
