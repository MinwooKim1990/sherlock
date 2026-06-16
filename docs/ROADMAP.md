# Sherlock Upgrade Roadmap (v0.9+)

> **STATUS (2026-06-11): all 35 items shipped.**
> R1–R4, R9–R10, R12–R13, R18, R26–R28 landed in v1.0; the remainder landed in
> v1.1. Scope notes: **R6** ships as auto JSON-mode with a memoized per-provider
> fallback (never loses a completion); **R16** is satisfied by the compact
> line-oriented serializations introduced in the v1.0 waste-removal pass (no
> TOON dependency); **R17** is an optional extra (`pip install "sherlock[compress]"`,
> config `search.deep_research_compress`, plain-truncation fallback);
> **R25** sectioned synthesis engages on runs with >18 facts + a strategy
> outline (small runs keep the cheaper single call); **R32/R34** ship as the
> corrections/supersede/`invalid_at_turn` mechanism (non-destructive, no graph).
> The document below is kept as the original engineering rationale.


**Goal.** Sherlock is a bring-your-own-LLM context-curation library: any model the user wires in — including 3B–14B local models — should get measurably smarter because Sherlock feeds it better-curated context, and every model should stop paying for tokens that buy nothing. The published evidence says this is achievable: curated atomic memory beats raw-context stuffing at 1–3B scale (A-Mem), curated context lets a small open model match a GPT-4.1-based production agent (ACE), and the bigger relative lift from context curation consistently lands on the *smaller* model (Zep on gpt-4o-mini).

**Locked principles (non-negotiable, enforced in review):**

1. **The user owns model choice.** No model routing, cascading, or silent substitution. LLM-1/2/3 are wired by the user and never rerouted by code. Capability flags (e.g. "my runtime supports JSON-schema decoding") are allowed; picking a model for the user is not.
2. **No result caps.** Results are never restricted, shortened, or skipped to save money. No budget ceilings on research or retrieval.
3. **Savings = waste elimination only.** Legitimate savings: not re-paying bytes the provider already processed (caching), not re-sending information already represented elsewhere in the prompt (compaction frontier, dedup), not losing whole paid calls to a parse error (salvage), not shipping payloads nobody reads (conditional blocks, dead fields).
4. All internal prompts in English; replies in the user's language.

Every item below was checked against these principles; items are tagged where the framing matters.

---

## Where we are

Sherlock v0.8 has the right skeleton: SQLite + chroma/fastembed vector store, k-turn raw window, LLM-2 summaries and pinned facts, decay tiers (warm/cold/forgotten), hybrid RAG (vector + BM25 + entity), a text-tag tool protocol parsed only from LLM output (prompt-injection-aware), and a deep-research loop whose delta protocol (URL dedup, only-new-fragments-per-round, capped state digest) genuinely avoids re-reading old material. Whole-turn-only tail fitting, per-message token memoization, and error-response exclusion from memory are all in place.

The measured limits are equally real (all numbers from the project's own `count_tokens`, tiktoken cl100k_base):

- **Per-turn protocol overhead:** `DEFAULT_SHERLOCK_EXTENSION` = **1,308 tokens** injected every turn (`sherlock/agent.py:337-435`); empirical turn-1 system message via `with_callable()` = **1,394 tokens** with a 9-token user prompt — ~99% of system overhead is protocol, not user content. Companion prompts: LLM-3 **1,599 tokens**, LLM-2 **1,388 tokens**, every call.
- **Small windows collapse:** at ctx=8,192 the budget math yields `sys_ceiling=1000, k_pool=0` — the protocol is head-truncated mid-sentence and the model cannot see the previous turn (`sherlock/budget.py:129-145`, `sherlock/agent.py:2444-2473`). At ctx=16,384, `k_pool=334`. The architecture silently degrades hardest exactly in the regime it is supposed to help.
- **BYO path mis-budgeted:** `with_callable()` cannot declare a context window; the registry falls back to 128K (`sherlock/budget.py:54`), so an 8K local model is budgeted ~81K tokens of raw tail.
- **Double-pay history:** the k-turn tail greedily takes all leftover window (81–111K on a 200K model) and keeps raw turns that LLM-2 already compacted into TIER-2 highlights — the same information paid twice, every turn, per-turn cost growing linearly with session length.
- **Zero prompt caching:** the system prompt is flattened to one string (`sherlock/agent.py:2428-2442`); `LiteLLMProvider` emits no `cache_control` or `prompt_cache_key` (`sherlock/providers/litellm_provider.py:47-95`). The stable prefix is re-paid at full price every turn and up to 3× per turn in tool rounds (`sherlock/agent.py:2830-2849`).
- **Fragile formats:** all-or-nothing JSON parsing wastes whole 1.4–3.6K-token LLM-2/LLM-3 calls on a single malformed brace (`sherlock/inference/engine.py:286-289`, `sherlock/memory/summarizer.py:216-224`); malformed tool tags leak verbatim into user-visible replies.
- **Dead artifacts:** LLM-2's `retrieval_keywords` and LLM-3's `context_to_expand`/`context_to_exclude` are produced (paid output tokens) and never consumed.
- **Korean gaps:** whitespace tokenization breaks BM25/entity retrieval on agglutinated forms ("유진이는" ≠ "유진"); the no-tiktoken fallback undercounts Korean 3.7× (64 chars = 59 real tokens vs 16 estimated), causing real overflow.
- **Deep research trust layer missing:** fact merge is exact-lowercase-string dedup (`sherlock/agent.py:1431-1457`), citations are attributed once from 160-char snippets and never verified, convergence counts new URLs rather than new knowledge, and the state digest shows LLM-3 at most 20 facts.

The roadmap below is ordered within each theme by impact/effort ratio (quick wins first); numbering R1–R35 is global. Items marked **[waste-only]** change cost with zero possible effect on result quality.

---

## Theme 1: Small-model intelligence

The compounding failure chain today: small windows are mis-budgeted (R1, R2) → what survives the window is protocol bloat (R3) → the formats the model must emit are the fragile ones (R4–R7) → the curation levers that could compensate are dead (R8).

### R1. Let `with_callable()` declare the model's context window — [high impact / small effort]
**Problem.** `with_callable()` has no `context_window` or budget-override parameters (`sherlock/agent.py:3360-3392`); the resulting `ModelConfig(provider="callable", ...)` (`:3510`) leaves `context_window=None`, which resolves through the registry wildcard to 128,000 (`sherlock/budget.py:54`). A user wrapping an 8K Qwen-7B gets a ~81K-token raw tail budgeted — guaranteed provider-side overflow Sherlock can neither see nor control. This silently defeats the slot-budget system for the flagship BYO-LLM audience.
**Proposal.** Add `context_window: int | None` and `max_output_tokens: int | None` to `with_callable()` (and an optional `slot_budget_overrides` dict), threaded into `ModelConfig` and `select_profile_for_window`. If unset, log a one-time warning that the 128K default is being assumed. Mirror the same parameters for LLM-2/LLM-3 callables, which have the same problem.
**Evidence.** Internal measurement (k_pool ≈ 81K budgeted for an 8K window); MemGPT's premise that the manager must know the true window to manage it: https://arxiv.org/abs/2310.08560
**Acceptance criteria.**
- `Sherlock.with_callable(fn, context_window=8192)` produces prompts whose measured token total never exceeds 8,192 − declared output reserve.
- A regression test asserts no registry-wildcard fallback when the parameter is provided.

### R2. Proportional small-window budget profiles + correct token counting — [high impact / small effort]
**Problem.** Only two budget profiles exist (DEFAULT ≥200K, SMALL <200K); SMALL still reserves `output_reserve=15,000` + `floor_k_turn_budget=4,000` (`sherlock/budget.py:129-145`). The ceiling math (`sherlock/agent.py:2444-2473`) yields `sys_ceiling=1000, k_pool=0` at ctx=8,192 — the 1,308-token protocol is chopped mid-sentence and conversation history vanishes. Separately, the no-tiktoken fallback (`len/4`, `sherlock/budget.py:164-204`) undercounts Korean 3.7×, and cl100k is a miscalibrated proxy for Qwen/Llama tokenizers on CJK.
**Proposal.**
- Replace fixed reserves with proportional ones: `output_reserve = clamp(8% of window, 512, 15_000)`; `floor_k_turn` expressed as "N whole most-recent turns" (default 2) that **wins over block reserves** — history is never zero.
- Add profiles for 8K/16K/32K windows; `select_profile_for_window` interpolates rather than cliff-stepping at 200K.
- Fix the fallback counter for CJK (chars-per-token ≈ 1.1 for Hangul/Han ranges), and add an optional per-model calibration hook (`tokens_per_char` measured once from the user's own tokenizer via a callable, since model choice — and therefore tokenizer — belongs to the user).
**Evidence.** Internal measurements (k_pool=0 @8K, 334 @16K; 59 vs 16 token Korean undercount); input-length degradation makes every mis-budgeted token an accuracy loss: https://arxiv.org/abs/2402.14848
**Acceptance criteria.**
- At ctx=8,192 with the default extension: full (compact, see R3) protocol survives untruncated, ≥2 whole turns of history present, no provider overflow across a 50-turn Korean test conversation without tiktoken installed.
- Property test: for all windows 4K–200K, `sys_ceiling + k_pool + output_reserve + user ≤ ctx`.

### R3. Compact, conditional, truncation-ordered protocol extension — [high impact / small effort]
**Problem.** `DEFAULT_SHERLOCK_EXTENSION` (`sherlock/agent.py:337-435`) is 1,308 tokens of monolithic prose, injected unchanged even when both search engines are `None` (verified: the model then emits a search tag and `_do_tool_call` returns "no search engine configured", `agent.py:2595-2601` — paid output tokens for a tool that doesn't exist). It is ~16% of an 8K window. `_truncate_to_tokens` (`agent.py:2242-2262`) keeps the head, but the load-bearing rules (tag syntax discipline, citation rules, reply-language rule) sit at the tail — exactly what gets cut.
**Proposal.**
- Assemble the extension from per-capability fragments gated on what is actually wired (no search fragment when engines are `None`; no deep_research fragment without an engine + approval path).
- Provide a compact variant (target ≤450 tokens) for windows <32K: short imperative bullets, no persona prose, one aligned few-shot tag example, an explicit "most replies need no tool" line.
- Reorder so the most critical rules come first (tag grammar, reply-language, citation), and restate the single most critical rule in one line at the end.
**Evidence.** Personas don't improve accuracy (https://aclanthology.org/2024.findings-emnlp.888/); short prompts beat verbose ones on 1.5–8B models (https://arxiv.org/html/2601.22025v1); reasoning degrades by ~3K input tokens (https://arxiv.org/abs/2402.14848); instruction-position U-curve (https://medium.com/@lars.chr.wiik/llm-instruction-placement-in-prompts-it-matters-a-lot-3b57580756ee)
**Acceptance criteria.**
- With engines disabled, the extension contains no search/fetch/deep_research instructions and the model emits zero spurious tool tags across the eval probe set.
- Compact variant ≤450 tokens measured; under forced truncation to 300 tokens, tag grammar + language rule still present.

### R4. Error-tolerant parsing ladder + retry-with-error-feedback for LLM-2/LLM-3 — [high impact / small effort]
**Problem.** `infer` returns `{}` on any parse failure (`sherlock/inference/engine.py:286-289`, `_safe_parse_json` `:533-567`); the summarizer falls back to storing raw text as a "summary," losing facts/persona/predictions/worth_digging entirely (`sherlock/memory/summarizer.py:216-224`). One malformed brace discards 1.4–3.6K paid tokens. StructuredRAG shows 8B models fail composite-object JSON 75–100% of the time on some shapes — this is the highest-frequency waste path on small models.
**Proposal.** Three-stage ladder, applied in both `engine.py` and `summarizer.py`:
1. **Lenient parse** (BAML schema-aligned style): strip fences/CoT preamble, tolerate trailing commas, unquoted keys, missing closing brackets. Zero extra calls.
2. **Field-level salvage:** keep the valid fields, drop the broken one; never discard a whole response.
3. **One re-ask** containing the failed output + the specific validation error (Instructor pattern), optionally at higher temperature — i.e. sample-until-valid, capped at 2 attempts. Cost is ~1× on the happy path; the extra call fires only when the alternative is a 100% wasted call. **[waste-only]**
**Evidence.** https://boundaryml.com/blog/schema-aligned-parsing ; https://python.useinstructor.com/concepts/retrying/ ; https://arxiv.org/abs/2408.11061 ; adaptive resampling economics: https://arxiv.org/abs/2305.11860
**Acceptance criteria.**
- Replay corpus of real malformed small-model outputs (collect during eval runs): ≥90% of previously-discarded responses yield at least partial structured data; zero raw-text-as-summary fallbacks.
- Telemetry counter for salvage stage reached per call.

### R5. Schema flattening, tiering, and few-shot anchoring for companion calls — [high impact / medium effort]
**Problem.** LLM-2's schema has ~10 top-level fields with nested fact objects (7 sub-fields each); LLM-3's has 7 (`sherlock/memory/summarizer.py:23-134`, `sherlock/inference/engine.py:17-144`). This sits squarely in the demonstrated small-model failure zone (Llama-3-8B: 0% on `List[string]` in one task, 25% on composite objects). IFEval shows multi-constraint compliance compounds failure — one mega-call is the worst shape.
**Proposal.**
- Define a **core schema** (3–4 flat fields) and an **extended schema**; select by the companion model's declared window/size, not by rerouting the model.
- Split LLM-2's mega-call into 2 focused calls where the extended schema is too heavy (fact extraction vs persona/prediction update) — same user-wired model, fewer simultaneous constraints.
- Embed the exact schema text + one aligned few-shot example in each prompt (the dottxt result: aligned few-shot is what makes structure free).
- Prefer a line-keyed `field: value` output format over nested JSON where possible — cheaper and salvageable line-by-line (synergy with R4 and R16).
**Evidence.** https://arxiv.org/abs/2408.11061 ; https://blog.dottxt.ai/say-what-you-mean.html ; https://arxiv.org/abs/2311.07911
**Acceptance criteria.**
- Parse-success rate for a 7B-class model (e.g. Qwen2.5-7B) on the eval probe set ≥95% with core schema (baseline measured first).
- No regression in extracted-fact recall on the existing integration tests.

### R6. Constrained-decoding passthrough as a provider capability — [high impact / medium effort]
**Problem.** Sherlock relies on parsing luck for LLM-2/LLM-3 JSON. Grammar-constrained decoding guarantees parseable output by construction, improves downstream quality (up to +4% GSM8K), and speeds generation ~50% — validated down to Llama-3.2-1B. The provider interface (`sherlock/providers/base.py:14-21`) is text-only and cannot express it.
**Proposal.**
- `LiteLLMProvider`: pass `response_format`/`json_schema` for LLM-2/LLM-3 calls when `litellm` reports support (Ollama, vLLM, llama.cpp, OpenAI-compatible servers all support it).
- `with_callable()`: opt-in `supports_json_schema=True` flag; when set, Sherlock passes the schema alongside the prompt so the user's runtime can enable guided decoding. This is a capability flag on the user's chosen model, not routing.
- R4's ladder remains the fallback for providers without grammar support.
**Evidence.** https://arxiv.org/abs/2501.10868 ; https://blog.dottxt.ai/say-what-you-mean.html ; https://arxiv.org/pdf/2510.03847
**Acceptance criteria.**
- With a local vLLM/llama.cpp runtime in guided mode, LLM-2/LLM-3 parse failures = 0 across a 200-call soak run.
- Flag absent ⇒ behavior byte-identical to today (back-compat test).

### R7. Tag-protocol hardening: one regular grammar, repair pass, explicit no-tool path — [medium impact / medium effort]
**Problem.** Tags must match exact regexes (`_TOOL_TAG_RE` `sherlock/agent.py:71-75`, companions `:51-54`, deep_research `:141-144`). Small models mangle bespoke formats (single brackets, fenced tags, translated keywords); a malformed tag is not stripped — it leaks verbatim into the user-visible reply and the call silently never happens. Research verdict: the text-tag *paradigm* is fine for small models (BFCL: prompted Qwen3-4B ≈ fine-tuned-FC 3B; NLT: +18.4pp, −70% variance, open-weight models gain most) — the regex intolerance is what's broken. Native FC is therefore not required; keep one format everywhere.
**Proposal.**
- Near-miss detector + repair pass: recognize `<sherlock-tool: …>`, fenced tags, `<<tool: search …>>`, etc.; canonicalize and execute, or strip from the user-visible reply and log.
- Pre-execution argument validator with one corrective re-ask on invalid args (mirrors R4's ladder).
- Add the explicit "no tool needed — just answer" path and unambiguous tool names to the extension (Hammer: small models emit spurious calls without an irrelevance path; ties into R3's few-shot example).
**Evidence.** https://gorilla.cs.berkeley.edu/leaderboard.html ; https://arxiv.org/abs/2510.14453 ; https://arxiv.org/abs/2410.04587
**Acceptance criteria.**
- Corpus of real mangled tags (collected from small-model eval runs): ≥80% repaired and executed; 0% leak verbatim to the user.
- Spurious-call rate on no-tool-needed probes drops measurably vs baseline.

### R8. Consume LLM-3's curation levers; relevance-gate speculative carry-forward — [medium impact / medium effort]
**Problem.** `context_to_expand`/`context_to_exclude` are produced every infer call (`sherlock/inference/engine.py:133-134, 294-295`) and never consumed — the only sink is bookkeeping (`sherlock/agent.py:2039`). Meanwhile turn N's hypotheses + freshness search are injected into turn N+1's TIER 3 unconditionally (`agent.py:632-635, 2800-2808, 2405-2416`) even after a topic pivot — actively misleading context, and small models are the most suggestible. Note the existing asymmetry: LLM-2 predictions *are* relevance-gated via RAG (`:2214-2229`); LLM-3 output is not.
**Proposal.** This is ACE's curation-delta pattern with the consumer Sherlock never built:
- `_assemble_messages` applies `context_to_exclude` as an exclusion filter over TIER-2/3 blocks and `context_to_expand` as targeted RAG prefetch hints for the next slot.
- Gate pending hypotheses/search results with a cheap cosine check against the current `user_text` (embeddings already local); below threshold, hold them out of the slot (they remain retrievable). Excluding misleading context is curation, not result restriction.
**Evidence.** https://arxiv.org/abs/2510.04618 (ACE: small open model matched GPT-4.1-based agent via curation deltas)
**Acceptance criteria.**
- Topic-pivot probe set: stale `[INFERENCE HYPOTHESES]`/`[WEB SEARCH RESULTS]` blocks no longer appear post-pivot; answer quality on pivot turns improves on the eval harness.
- Exclusion filter covered by unit tests over block ids.

---

## Theme 2: Token efficiency without limits

Two items dominate everything else by an order of magnitude in recoverable tokens: the compaction frontier (R10) and prompt-cache alignment (R9/R11). Both are pure waste elimination.

### R9. Provider cache enablement: `cache_control` passthrough, companion-prompt pinning, `prompt_cache_key`, hit-rate telemetry — [high impact / small effort] [waste-only]
**Problem.** `LiteLLMProvider.chat` sends plain `{"role","content"}` dicts (`sherlock/providers/litellm_provider.py:47-95`): zero caching on Anthropic-style providers (explicit breakpoints required), no `prompt_cache_key` for OpenAI routing, and cache effectiveness is invisible to `TokenUsage`. The byte-identical LLM-3 (1,599 tok) and LLM-2 (1,388 tok) system prompts are re-paid at full price on every background call — both fire well within Anthropic's 5-minute TTL.
**Proposal.**
- Emit OpenAI-format content blocks with `"cache_control": {"type": "ephemeral"}`; LiteLLM translates per provider and **silently drops** it where unsupported, so it can be attached unconditionally (avoid `ttl` for Bedrock portability).
- Pin the LLM-2/LLM-3 system prompts as cached blocks immediately (no layout work needed — they are already stable).
- Pass `prompt_cache_key=<conversation_id>` on OpenAI routes (documented hit-rate jump 60%→87%); gate via `litellm.supports_prompt_caching()`.
- Surface `cached_tokens` / `cache_creation_input_tokens` into `TokenUsage` so users see their hit rate.
**Evidence.** https://platform.claude.com/docs/en/build-with-claude/prompt-caching ; https://docs.litellm.ai/docs/completion/prompt_caching ; https://developers.openai.com/cookbook/examples/prompt_caching_201 ; https://github.com/BerriAI/litellm/issues/17250
**Acceptance criteria.**
- On an Anthropic-route model, companion calls show `cache_read_input_tokens ≥ 1,300` from the second call onward.
- Telemetry reports cache hit rate per role; unsupported providers show no errors and unchanged behavior.

### R10. Compaction frontier: evict summarized raw turns; append-only K-turn tail — [high impact / medium effort] [waste-only]
**Problem.** `k_pool` = all leftover window (`sherlock/agent.py:2444-2467`; tail walk `:2264-2305`): 81–111K tokens of raw turns on a 200K model, including turns already represented in TIER-2 `[COMPACTED MEMORY HIGHLIGHTS]` and recoverable via `memory timeline/lookup`. Same information paid twice, every turn; per-turn cost grows linearly (quadratic cumulative). The sliding window also shifts message positions every few turns, killing any message-level KV/prefix cache.
**Proposal.** MemGPT's eviction boundary, which Sherlock has the storage for but never enforced:
- Raw turns older than the last LLM-2 compaction event drop out of the tail; they remain in highlights + the memory store + on-demand tools (`<<sherlock-tool: memory …>>` is exactly MemGPT's recall storage).
- Between compactions the tail is **append-only** — the prefix only re-keys at compaction events, a planned infrequent cache write instead of per-turn invalidation (composes with R11).
- Guaranteed minimum of N whole recent turns regardless of frontier (R2's floor).
- The eval evidence preempts the "losing information" objection: extracted memory beats full raw context on LoCoMo (+26% LLM-judge, >90% token savings) — keeping raw history is paying twice for *worse* answers.
**Evidence.** https://arxiv.org/abs/2310.08560 ; https://arxiv.org/abs/2504.19413 ; https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus ; https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
**Acceptance criteria.**
- 100-turn session on a 200K model: per-turn prompt tokens plateau after the first compaction instead of growing linearly (measured via existing token accounting).
- LongMemEval-style recall probes (R28) over the evicted span pass via memory tools/highlights at parity or better.
- Tail message ids are append-only between compaction events (asserted in tests).

### R11. Tier-aligned, block-structured prompt layout with cache breakpoints — [high impact / medium effort] [waste-only]
**Problem.** `_assemble_messages` already orders TIER 1 → 2 → 3 correctly in spirit, then flattens everything into one opaque `composite_system` string (`sherlock/agent.py:2428-2442`). Consequences: Anthropic-style providers cache nothing (see R9); OpenAI/Gemini implicit caches break at the first changed byte — and TIER-2 truncation (`:2393-2395`) can reflow earlier bytes; volatile TIER 3 (minute-granularity timestamp, hypotheses, search results, RAG) lives *inside* the system message, churning the whole prefix. Tool rounds re-send the full prompt up to 3× per turn (`:2830-2849`).
**Proposal.** All three major providers (and local vLLM/SGLang prefix caching) reward the same byte-identical prefix, so one layout serves everyone:
- **Block A** (breakpoint 1): protocol extension + TIER-1 user system prompt — changes only on config change.
- **Block B** (breakpoint 2): TIER-2 pinned/persona/highlights — changes only at compaction events; make truncation deterministic and append-biased so bytes before the breakpoint never reflow. Fix the anti-recency truncation bug here (head-keeping currently cuts the *newest* summaries at the bottom first).
- **Messages:** append-only tail (R10), breakpoint 3 on the last tail message (Anthropic's 20-block backward lookup then gives incremental hits as turns append).
- **Volatile TIER 3** (timestamp, hypotheses, search/RAG blocks) moves after all breakpoints — prepended to the current user turn — so churn never invalidates the prefix. Tool rounds 2–3 then re-read the prefix at 0.1× instead of 1.0×.
- Internally: `ChatMessage` grows optional labeled content blocks with `cache_hint` (`sherlock/providers/base.py`), with a default flattener for back-compat.
**Evidence.** https://platform.claude.com/docs/en/build-with-claude/prompt-caching ; https://developers.openai.com/api/docs/guides/prompt-caching ; https://developers.googleblog.com/gemini-2-5-models-now-support-implicit-caching/ ; https://docs.vllm.ai/en/latest/design/prefix_caching/
**Acceptance criteria.**
- Anthropic route: ≥80% of TIER 1+2 tokens read from cache from turn 2 onward in a 20-turn session; tool rounds within a turn read the full prefix from cache.
- Byte-stability test: serialize Block A+B across 10 consecutive turns without compaction → identical bytes.
- Estimated effect (from measurements): the 2.5–4K-token stable prefix re-sent 1–3×/turn drops to ~0.1× price on Anthropic, ~50% on OpenAI, ~90% discount on Gemini 2.5+.

### R12. Conditional provenance ledger + trimmed companion prompts — [medium impact / small effort] [waste-only]
**Problem.** Every LLM-3 infer call ships the provenance ledger (last 40 user utterances × 200 chars ≈ up to ~2K tokens) plus all system-persona entries (`sherlock/inference/engine.py:236-280`), though the ledger is load-bearing only on provenance-probe turns; `auto_infer="smart"` fires on every topic shift, so this is frequent. The ~450-token Detective/Clinical persona prose in the LLM-3 prompt is style guidance small models ignore (and personas measurably don't help any model).
**Proposal.** Ship the full ledger only when a provenance probe is detected (cheap lexical patterns: "내가 언제 말했", "when did I say", "who told you"…); otherwise a 1-line ledger digest. Alternatively/additionally, restructure the ledger as an append-only cacheable block right after LLM-3's system prompt (delta protocol Sherlock already uses for deep research). Cut the persona prose from the small-window LLM-3 prompt variant (R3/R5 share this work). Expected: 30–50% off each background call with zero result loss.
**Evidence.** https://www.letta.com/blog/sleep-time-compute ; https://arxiv.org/html/2504.13171v1 (amortize background work, don't add fixed overhead); https://aclanthology.org/2024.findings-emnlp.888/
**Acceptance criteria.**
- Mean LLM-3 input tokens per call drops ≥30% on the eval transcript replay; provenance-probe answers unchanged (existing probe tests pass).

### R13. Slot deduplication + pinned-block hygiene — [medium impact / small effort] [waste-only]
**Problem.** (a) A SUMMARY entry surfaced by hybrid search appears in `[RAG RETRIEVAL]` while also riding in TIER-2 highlights; RAG-retrieved pinned facts duplicate `[PINNED FACTS]` — exclusions only cover recent USER_UTTERANCE + DEEP_RESEARCH (`sherlock/agent.py:927-939`). (b) Pinned facts ship verbatim in insertion order with no per-fact size cap (`:2117-2141`, cap 25 in `store.py:408`); one long fact rides whole every turn.
**Proposal.** Set-difference the RAG block against ids already rendered in TIER 2 (the `all_entries` list is already shared at `:2343` — this is nearly free). Order pinned facts by relevance to the current query (embedding similarity, already computed channel) with stable fallback ordering for cache friendliness within Block B; add a per-fact render cap with a "(truncated — pinned fact #id, use memory lookup)" marker rather than silent loss.
**Evidence.** Internal audit (B6/B7); duplicate context as pure waste per https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
**Acceptance criteria.**
- No memory-entry id appears in two slot blocks in any assembled prompt (asserted in integration tests).
- Slot token total drops on transcripts with heavy summary overlap; zero behavioral regressions.

### R14. Query-aware extraction instead of blind head-truncation for fetched content — [medium impact / small effort] [waste-only]
**Problem.** Fetch bodies are cut at `[:4000]` chars (`sherlock/agent.py:2682`), research pages at `[:2500]`/`[:1000]`, snippets at `[:160]`/`[:200]`/`[:400]`. Head-keeping is content-blind; the answer is often mid-page, so the model re-searches or re-fetches — extra rounds, each a full-prompt resend.
**Proposal.** Token-neutral replacement: extract windows around query-term/entity hits (BM25-style scoring over page chunks, no LLM call), concatenate the top windows up to the same size budget, always retaining the page title + URL anchor. Applies to both the chat-loop `fetch` tool and research-round fetches (synergy with R24).
**Evidence.** Internal audit (B5); snippet-grounding threshold problem documented across deep-research systems: https://blog.promptlayer.com/how-deep-research-works/
**Acceptance criteria.**
- On a fetch-heavy probe set, re-fetch/re-search rate drops vs baseline at identical size budgets; answer accuracy non-decreasing.

### R15. Expose block structure + cache hints to BYO callables — [medium impact / medium effort] [waste-only]
**Problem.** `with_callable()` users receive one flattened system string (`sherlock/providers/callable_provider.py`), so they cannot attach Anthropic SDK `cache_control`, OpenAI `prompt_cache_key`, or local-engine hints on their side — even after R11 makes the prefix stable.
**Proposal.** Optional richer contract: the callable may accept `blocks: list[{text, label, stability, cache_hint}]` (detected by signature or an opt-in flag); the default flattener preserves full back-compat. Document that local engines get prefix reuse for free once the prefix is stable (vLLM `--enable-prefix-caching`, SGLang RadixAttention — 60–85% hit rates reported on agent loops). Sherlock's job under the locked principles is to emit a cache-friendly stable prefix and hand hints through — never to route.
**Evidence.** https://docs.vllm.ai/en/latest/design/prefix_caching/ ; https://arxiv.org/pdf/2312.07104 ; https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide
**Acceptance criteria.**
- Example in `README.md` wiring Anthropic SDK caching through a callable, with measured cache reads.
- Existing single-string callables run unchanged (back-compat suite).

### R16. Compact tabular serialization (TOON/CSV-style) for injected arrays and companion schemas — [medium impact / medium effort]
**Problem.** Sherlock injects exactly the payload shape where JSON/prose is most wasteful — uniform arrays: retrieved memories, search results, the 40-line provenance ledger, LLM-2 predictions, deep-research deltas. Benchmarks show 30–60% fewer tokens (39.6% in the 4-model benchmark) with equal-or-better accuracy (73.9% vs 69.7%).
**Proposal.** Tabular/TOON-style rendering in the slot formatters (`_format_retrieved_block`, search blocks, ledger, predictions) and line-keyed compact **output** formats for LLM-2/LLM-3 (synergy with R4/R5: line-oriented output is salvageable line-by-line). Keep field names stable for cache friendliness.
**Evidence.** https://github.com/toon-format/toon ; https://arxiv.org/abs/2603.03306 ; https://www.infoq.com/news/2025/11/toon-reduce-llm-cost-tokens/
**Acceptance criteria.**
- ≥30% token reduction measured on the rendered array blocks of the eval transcript corpus; eval answer-quality scores non-decreasing.

### R17. Optional LLMLingua-2 compression for fetched web content in deep research — [medium impact / large effort] [waste-only]
**Problem.** Multi-round fetches are deep research's largest single token sink; fetched pages are mostly boilerplate/navigation — genuinely wasted tokens.
**Proposal.** Optional dependency (same pattern as fastembed/chroma): a BERT-size local LLMLingua-2 compressor applied **only** to fetched page text before it enters research-round prompts (2–5× typical compression, up to ~15× on document-heavy prompts, fully local, CPU-friendly). Scope explicitly excludes user memories, pinned facts, and protocol prompts — compression there risks meaning loss and conflicts with provenance guarantees.
**Evidence.** https://arxiv.org/pdf/2403.12968 ; https://www.microsoft.com/en-us/research/project/llmlingua/
**Acceptance criteria.**
- With the extra installed: research-round input tokens drop ≥40% on fetch-heavy topics; fact recall and citation accuracy (R23 metrics) non-decreasing. Absent the extra: zero behavior change.

---

## Theme 3: Deep research quality

The delta protocol (only-new-fragments, URL dedup, capped digest, fetch discipline) is genuinely SOTA-aligned. The gaps are in convergence signals, evidence trust, and state representation.

### R18. Fact-gain convergence + plan-time effort scaling — [high impact / small effort]
**Problem.** The stall counter (`sherlock/agent.py:1514` area) counts new **URLs**, not new **knowledge**: a round adding 5 fresh URLs and zero facts resets the counter; a topic where few canonical pages answer many questions converges prematurely. Research shape is fixed (wide round-1 sweep, up to 20 rounds) regardless of topic complexity — `_deep_research_plan` reads static config only.
**Proposal.** Stall on facts-merged-per-round (already computable from existing state, zero extra LLM calls); later, on outline-section coverage (R25). Extend `plan_search` (`sherlock/inference/engine.py:454`) with a complexity field (simple fact-find / comparison / open-ended survey) consumed by `_run_deep_research` to set max_rounds and round-1 width — Anthropic's effort-scaling rules, embedded in the plan prompt. Scaling effort *up* for complex topics is explicitly allowed; this never caps results, it stops rounds that are no longer producing knowledge. **[waste-only on the stop side]**
**Evidence.** https://www.anthropic.com/engineering/multi-agent-research-system ; https://www.langchain.com/blog/open-deep-research
**Acceptance criteria.**
- Zero-fact-gain rounds terminate runs on the regression topic set; canonical-page topics no longer converge early (probe pair in tests).
- Simple fact-find topics complete in ≤3 rounds with unchanged answer accuracy.

### R19. Reciprocal Rank Fusion across multilingual queries + credibility-tiered selection — [medium impact / small effort] [waste-only]
**Problem.** Hit aggregation (`sherlock/agent.py:1371-1388`) is concatenation + URL dedup; the `new_hits[:8]` prompt cut and top-3 fetch picks follow engine-return order — discarding the cross-query agreement signal Sherlock's own multilingual plan already paid for. `_fact_corroboration` (`:326`) counts distinct domains but a community-only fact ranks identically to an official-source fact.
**Proposal.** Deterministic, zero-LLM-token: RRF-merge per-query result lists so URLs surfaced by multiple queries/languages rank first; apply source-type tiers (official > news > blog > community, reusing `_source_type` `:254-330`) when picking the 8 in-prompt hits and 3 fetch targets, and weight corroboration by tier.
**Evidence.** https://github.com/Raudaschl/rag-fusion ; https://openreview.net/pdf?id=lz936bYmb3 ; https://arxiv.org/pdf/2512.09483
**Acceptance criteria.**
- Unit-tested RRF over fixture result sets; on the research eval topics, fraction of official/news sources among fetched pages increases with no added tokens.

### R20. Counter-evidence and perspective seeding in plan and meta-questions — [medium impact / small effort]
**Problem.** The five FIXED_META questions (`sherlock/agent.py:1327-1333`) are process-questions; LLM-3's gap-driven questions inherit whatever angle round 1 found — confirmation bias by construction. The plan never deliberately seeds opposing-view queries (Gemini explicitly does; STORM seeds perspective diversity).
**Proposal.** One-line prompt changes: `plan_search` must emit ≥1 counterargument/conflicting-evidence query; `generate_meta_questions` (`sherlock/inference/engine.py:387`) requests perspective-diverse questions (different stakeholder/source-type angles). Pairs with R22's contradiction surfacing.
**Evidence.** https://gemini.google/overview/deep-research/ ; https://arxiv.org/abs/2402.14207
**Acceptance criteria.**
- Plans on the eval topic set contain ≥1 counter-evidence query; final reports on contested topics surface at least one dissenting source (judged in eval harness).

### R21. Research-brief scoping step — [medium impact / small effort]
**Problem.** The topic string in the `deep_research` tag is the entire specification; `user_text` is only a language/purpose hint into `plan_search` (`sherlock/agent.py:1336-1344`). Long runs drift; mid-research inbox messages merge into an unstructured append-only list.
**Proposal.** Before `plan_search`, one LLM-3 call synthesizes an explicit brief (goal, sub-topics, constraints, output expectations) from the tag + recent turns; the brief ships in every round prompt and the synthesis as the "north star" (LangChain ODR's highest-value finding). Mid-research `extra_context` merges into the brief, not a list.
**Evidence.** https://www.langchain.com/blog/open-deep-research ; https://github.com/langchain-ai/open_deep_research
**Acceptance criteria.**
- Brief present in all round prompts and synthesis prompt; drift probes (topic with an ambiguous tag, clarified by conversation) produce on-target reports vs baseline.

### R22. Contradiction-aware semantic fact merge with two-sided reporting — [high impact / medium effort]
**Problem.** Merge is exact-lowercase-string dedup (`sherlock/agent.py:1431-1457`): "X opened in 2024" / "X was opened in 2024" accumulate as duplicates (inflating round prompts and the unbounded synthesis prompt, `:1816-1893`), while "X costs $40M" / "X costs $25M" coexist silently as two confirmed facts. Models rarely surface inter-source conflicts unprompted.
**Proposal.**
- Reuse the embedding-similarity dedup already built in `MemoryStore.add` (`sherlock/storage/db.py` / `store.py:227-283`, 0.92 cosine) at fact-merge time: near-duplicates union their sources (fixes paraphrase inflation — **[waste-only]**).
- High-similarity pairs that are *not* duplicates get a cheap LLM-2 conflict check; conflicting facts are marked `disputed` and the synthesis prompt instructs two-sided presentation with per-side sources and credibility tiers (R19).
**Evidence.** https://arxiv.org/abs/2605.17301 ; https://arxiv.org/html/2506.08500v2 ; https://arxiv.org/pdf/2504.00180
**Acceptance criteria.**
- Paraphrase fixtures dedupe with source union; planted-conflict fixtures yield a `disputed` fact and a two-sided sentence in the report.
- Synthesis prompt size drops on long runs (measured).

### R23. Claim-level citation verification pass — [high impact / medium effort]
**Problem.** Citations are attributed once by LLM-1 from 160-char snippets in `_answer_research_round` (`sherlock/agent.py:1657-1664`) and propagate unverified into the final report's inline citations — the highest-risk hallucination path in the pipeline. Evidence text is not retained per fact.
**Proposal.** Anthropic CitationAgent / VeriScore pattern: retain snippet/page evidence text alongside each fact's URLs at merge time (`:1440-1457`); after synthesis, one verification pass checks each cited (claim, URL) pair against stored evidence and flags or demotes unsupported citations (footnote "unverified" rather than silent removal — results are never dropped, they are labeled).
**Evidence.** https://www.anthropic.com/engineering/multi-agent-research-system ; https://arxiv.org/abs/2406.19276 ; https://arxiv.org/html/2505.16973
**Acceptance criteria.**
- Every inline citation in the final report maps to stored evidence text; planted-misattribution fixtures get flagged.
- Citation-accuracy metric added to the research eval harness and improves vs baseline.

### R24. Evidence compression instead of hard truncation in research rounds — [medium impact / medium effort]
**Problem.** Snippets reach LLM-1 at 160 chars and fetched pages at 1,000 chars in-prompt (`sherlock/agent.py:1657-1670`) — below the grounding threshold for reliable fact extraction, which is why facts mostly derive from engine snippets. Gemini/OpenAI read full pages, not snippets.
**Proposal.** When pages are fetched, LLM-2 compresses them into fact-dense extracts with retained source anchors (LangChain `compress_research` pattern) before they enter the round prompt; R14's query-aware extraction is the no-LLM fallback. Spends background LLM-2 tokens (its designed role) to make every LLM-1 token count — better results *and* fewer retry rounds.
**Evidence.** https://www.langchain.com/blog/open-deep-research ; https://ai.google.dev/gemini-api/docs/interactions/deep-research
**Acceptance criteria.**
- Facts-per-fetched-page increases on the eval topics; rounds-to-convergence non-increasing; citation accuracy (R23) improves on page-derived facts.

### R25. Outline/draft-as-state with section-wise synthesis — [high impact / large effort]
**Problem.** Shared state is a flat fact list; `_state_digest` (`sherlock/agent.py:1593-1606`) shows LLM-3 at most 20 facts / 1,500 chars, so on long runs the question generator steers blind. Synthesis is one monolithic LLM-1 call over all facts (`:1816-1893`) — report structure is invented at the last moment and, on small models, the whole report must fit one completion.
**Proposal.** STORM + TTD-DR transplant into the existing delta protocol (no extra re-reading):
- Replace `{confirmed_facts[], open_gaps[]}` with a hierarchical outline whose sections facts attach to; thin/empty sections drive `next_queries` and meta-questions; the digest becomes per-section summaries (bounded regardless of run length).
- Synthesis writes section-by-section with per-section references — fixes the small-model output-length wall and gives `open_gaps` structural meaning.
- Section coverage becomes the mature convergence signal (extends R18).
**Evidence.** https://arxiv.org/abs/2402.14207 (+25% absolute on organization) ; https://arxiv.org/abs/2507.16075 (draft-as-state beat OpenAI Deep Research 69–74% head-to-head) ; https://research.google/blog/deep-researcher-with-test-time-diffusion/
**Acceptance criteria.**
- 15+-round runs: question generator demonstrably targets thin sections (logged); digest stays bounded.
- A 7B-class LLM-1 produces a complete multi-section cited report (impossible today in one completion); report-organization scores improve in the eval harness.

---

## Theme 4: Memory quality

### R26. Wire `retrieval_keywords`: fact-augmented key expansion + time-aware query filtering — [high impact / small effort]
**Problem.** LLM-2 produces `retrieval_keywords` every call (`sherlock/memory/summarizer.py:45`) and they are never indexed or queried — paid output tokens, discarded. This is precisely LongMemEval's fact-augmented key expansion: +9.4% recall@k, +5.4% accuracy averaged across models. Temporal questions also get no time filtering despite timestamps existing in SQLite.
**Proposal.** Index keywords as expansion keys in `sherlock/memory/store.py`; consume them as an additional deterministic channel in `sherlock/rag/hybrid.py`. Add time-range extraction from queries (deterministic patterns first) to filter/boost the search window (+6.8–11.3% temporal recall). The artifacts are already paid for — this is the cheapest high-evidence win in the codebase.
**Evidence.** https://arxiv.org/abs/2410.10813
**Acceptance criteria.**
- Recall@k on the memory probe set improves vs baseline with zero new LLM tokens per turn; temporal probes ("what did I say last week about…") hit the right entries.

### R27. Conversational query rewriting + Korean morpheme tokenization — [high impact / small effort]
**Problem.** `_retrieve_memories` passes raw `user_text` to hybrid search (`sherlock/agent.py:907-941`): elliptical follow-ups ("그럼 비행기에서는?") retrieve nothing — >60% of production follow-ups contain unresolved coreference/ellipsis. The whitespace tokenizer + exact-token entity match (`sherlock/rag/hybrid.py:40-70`, entity index `store.py:332-378`) means "유진이는" ≠ "유진": for the owner's primary language, two of three retrieval channels are dead.
**Proposal.**
- **Deterministic first (zero new LLM tokens):** augment the query with entities/keywords from the last k turns and LLM-2's stored keywords (R26).
- **One schema field:** `standalone_query` rides the existing paid LLM-2 call for the next-turn rewrite.
- **Korean:** tokenize BM25/entity channels with a morphological analyzer — kiwipiepy as an optional local dependency (consistent with the local-embeddings design) — or char-ngram BM25 for Hangul as the no-dependency fallback; normalize josa in entity extraction.
**Evidence.** https://arxiv.org/pdf/2410.07797 (+31.7% P@1, +25.2% MRR) ; https://arxiv.org/abs/2406.18960 ; https://pypi.org/project/kiwipiepy/ ; https://medium.com/@autorag/making-benchmark-of-different-tokenizer-in-bm25-134f2f0e72f8
**Acceptance criteria.**
- Korean ellipsis probe set ("그래서 걔는 어떻게 됐어?" after an entity thread): retrieval hit-rate goes from ~0 to parity with explicit queries.
- "유진이는" matches entity "유진" in the entity channel (unit test).

### R28. LongMemEval/LoCoMo-style memory regression harness — [medium impact / small effort]
**Problem.** Themes 2 and 4 restructure what enters the prompt; without category-level regression coverage (temporal reasoning, knowledge update, multi-hop, abstention), "waste elimination" claims can't be distinguished from silent recall loss. Commercial assistants drop 30–64% on LongMemEval — the headroom and the risk are both real.
**Proposal.** Extend the existing `evaluation/` harness (probes already exist under `evaluation/probes/`) with LongMemEval-style categories over scripted multi-session Korean+English conversations; track recall@k, answer accuracy, tokens-per-turn, and cache hit-rate as first-class metrics. Gate R10/R13/R32-R34 on it.
**Evidence.** https://arxiv.org/abs/2410.10813 ; https://github.com/xiaowu0162/LongMemEval ; https://snap-research.github.io/locomo/
**Acceptance criteria.**
- CI-runnable suite with per-category scores; every Theme-2/Theme-4 item lands with a before/after row.

### R29. Composite retrieval scoring + recall reinforcement + persisted reflection notes — [medium impact / small effort]
**Problem.** `sherlock/memory/decay.py` is time-only: no importance dimension, no recall-based reinforcement; `sherlock/rag/hybrid.py` fuses channels without an explicit recency×importance×relevance composite. LLM-3 hypotheses are regenerated every turn instead of persisted — paid output, no durable artifact.
**Proposal.** Generative-agents scoring: importance = one extra integer field on the already-paid LLM-2 call; reinforcement = `accessed_count`/`last_accessed` columns slowing decay on recall (MemoryBank/Ebbinghaus); composite score = arithmetic in the hybrid ranker. Write LLM-3's confirmed inferences back as first-class retrievable memories with child links (reflection notes) instead of re-deriving them — the sleep-time-compute move: background work *replaces* test-time work.
**Evidence.** https://arxiv.org/abs/2304.03442 ; https://arxiv.org/abs/2305.10250 ; https://arxiv.org/html/2504.13171v1
**Acceptance criteria.**
- Frequently-recalled facts survive decay tiers longer (unit tests on tier transitions); persisted reflection notes retrievable in later sessions; R28 multi-hop scores improve.

### R30. Typed memory split (episodic/semantic/procedural) with per-type slot policies — [medium impact / small effort]
**Problem.** Working (k-turn), semantic (pinned/summaries), and episodic-ish vector entries exist but share one policy — slot assembly can't budget or decay per type. Facts should decay slowly and dedupe aggressively; episodes should decay fast once summarized; procedural notes (user correction patterns) belong in the protocol extension.
**Proposal.** `memory_type` column on `MemoryEntry` (`sherlock/memory/entry.py`) + per-type budgets/decay rates in slot assembly. Deliberately scheduled before R32–R34: it is the prerequisite plumbing their policies hang off.
**Evidence.** https://arxiv.org/abs/2309.02427 ; https://langchain-ai.github.io/langmem/concepts/conceptual_guide/
**Acceptance criteria.**
- Migration script for existing DBs; per-type decay verified in unit tests; no R28 regression.

### R31. Structured memory reading format (timestamped records + relevance note) — [medium impact / small effort]
**Problem.** Retrieved memories are rendered as prose concatenation. LongMemEval: JSON-with-timestamps + Chain-of-Note reading improved QA up to +10 absolute points across three LLMs *with identical retrieval* — a pure formatting delta.
**Proposal.** Render `[RAG RETRIEVAL]` (and highlights) as structured timestamped records with a one-line relevance note each, in the compact tabular form from R16. Touches slot formatters in `sherlock/agent.py` / `sherlock/evaluation/output_format.py` only.
**Evidence.** https://arxiv.org/abs/2410.10813
**Acceptance criteria.**
- R28 QA accuracy improves with retrieval held constant; token delta ≈ 0 (or negative with R16).

### R32. Mem0-style write-time reconciliation (ADD/UPDATE/DELETE/NOOP) — [high impact / medium effort]
**Problem.** LLM-2 extracts facts but never reconciles: the store accumulates stale duplicates and contradictions, which then fill retrieval slots — paid tokens crowding out signal, disproportionately harmful to suggestible small models.
**Proposal.** After extraction, retrieve top-s similar stored memories and have the **same already-paid LLM-2 call** emit a per-fact op (ADD/UPDATE/DELETE/NOOP) — one extra output field, no new LLM role, no routing. Bound LLM-2's input on small windows using Mem0's recipe: rolling summary + last m raw messages instead of unbounded history (`sherlock/memory/summarizer.py:173-214` + `store.py`). LoCoMo evidence: extraction+reconciliation beats best RAG (J=66.88 vs 60.97) with ~90% token savings vs full-context.
**Evidence.** https://arxiv.org/abs/2504.19413
**Acceptance criteria.**
- Contradiction probes ("I moved from Seoul to Busan"): old fact updated/invalidated, not duplicated; R28 knowledge-update category improves; store growth rate per 100 turns drops.

### R33. A-Mem atomic notes: tags, contextual descriptions, and a links table — [high impact / medium effort]
**Problem.** Memories have content + embedding + decay but no link structure and no LLM-generated tags — multi-hop recall depends entirely on single-shot vector similarity. A-Mem is the strongest published small-model evidence: beats MemGPT-style baselines on Qwen2.5-1.5B/3B and Llama-3.2-1B/3B at ~1,200–2,500 tokens/interaction vs ~16,900 (~85% reduction).
**Proposal.** Add tags/keywords columns (largely already paid for via `retrieval_keywords`, R26) and a `links` table to `sherlock/storage/db.py`; on insert, link to top-k similar notes (embedding-only by default; optional LLM-2 link/evolve step). Retrieval gains 1-hop link expansion in `sherlock/rag/hybrid.py` — cheap multi-hop without a knowledge graph (HippoRAG's lesson: graph traversal replaces iterative retrieval calls).
**Evidence.** https://arxiv.org/abs/2502.12110 ; https://github.com/agiresearch/a-mem ; https://arxiv.org/abs/2405.14831
**Acceptance criteria.**
- R28 multi-hop category improves with a 7B LLM-1; link expansion adds no LLM calls at query time.

### R34. Bi-temporal fact invalidation (Graphiti-lite) — [medium impact / medium effort]
**Problem.** When circumstances change, old facts are either overwritten (history lost) or retained (stale-fact injection — the failure mode small models are most harmed by). Zep's evidence: temporally-resolved context gave the *bigger* relative lift to the smaller model (gpt-4o-mini +15.2%; temporal reasoning +48.2%; 1.6K context tokens vs 115K).
**Proposal.** Add `valid_from`/`invalid_at` (event time) alongside created/expired (ingestion time) to facts in `sherlock/storage/db.py`/`store.py`; R32's UPDATE/DELETE ops set `invalid_at` instead of deleting; slot assembly renders only currently-valid facts, while LLM-3 and memory tools may query history. No full knowledge graph.
**Evidence.** https://arxiv.org/abs/2501.13956
**Acceptance criteria.**
- "What was true before X changed?" probes answerable via memory tools; no invalidated fact ever appears in `[PINNED FACTS]`/highlights; R28 temporal category improves.

### R35. Span-grounded fact extraction + cheap faithfulness verification — [medium impact / medium effort]
**Problem.** 7–8B summarizers measurably hallucinate even over provided text (FaithBench), and a corrupted pinned fact or persona entry poisons every future turn through retrieval. Sherlock currently has no grounding requirement on LLM-2 output.
**Proposal.** Require each extracted fact/highlight to carry a source turn-id (the provenance ledger plumbing already exists) and, for facts persisted long-term, a short supporting quote; optionally verify persisted facts with a small local NLI model (Vectara HHEM-style — fits the local-first dependency pattern) before they enter long-term memory. Prevention over post-hoc judging: even SOTA judges catch only ~50% of hard cases.
**Evidence.** https://arxiv.org/html/2410.13210v1 ; https://arxiv.org/abs/2505.04847
**Acceptance criteria.**
- Every persisted fact has a turn-id; planted-hallucination fixtures (summarizer asked to summarize text it can't support) are rejected or flagged at write time.

---

## Suggested version mapping

| Version | Focus | Items |
|---|---|---|
| **v0.9 — "Honest small windows; stop paying twice"** | Make 8K–16K windows actually work; kill the two dominant waste streams; ship the free retrieval wins. All small-effort or measurement-critical. | R1, R2, R3, R4 (Theme 1 quick wins); R9, R10, R12, R13 (cache enablement, compaction frontier, companion overhead, slot dedup); R18 (fact-gain convergence); R26, R27, R28 (keywords, query rewrite + Korean, regression harness) |
| **v1.0 — "Structured reliability, cache-native layout, research trust"** | The medium-effort structural work: block-structured prompts with breakpoints, schema/grammar reliability, write-time memory reconciliation, research evidence trust. | R5, R6, R7, R8 (schemas, constrained decoding, tag hardening, curation levers); R11, R14, R15 (block layout + breakpoints, query-aware extraction, BYO blocks); R19, R20, R21, R22, R23 (RRF, counter-evidence, brief, contradiction merge, citation verification); R29, R30, R31, R32 (scoring/reflection, typed split, reading format, reconciliation) |
| **v1.1 — "Deep curation"** | The large-effort, highest-ceiling items, built on v1.0 plumbing. | R16, R17 (TOON serialization, LLMLingua-2); R24, R25 (evidence compression, outline-as-state + section-wise synthesis); R33, R34, R35 (A-Mem links, bi-temporal invalidation, faithfulness guardrails) |

Sequencing rationale: v0.9 is dominated by items where the fix is small and the measured waste or breakage is large (k_pool=0 at 8K, 128K default for BYO, dead `retrieval_keywords`, zero caching, double-paid history). v1.0 items mostly depend on v0.9 plumbing (R11 builds on R9/R10; R32 builds on R30; R23 builds on evidence retention introduced with R22) and are gated by the R28 harness landing first. v1.1 items either depend on v1.0 schema work (R33/R34 on R30/R32; R25 on R18/R21/R22) or are optional dependencies (R17). Every release must hold the line on the locked principles: no routing, no result caps, savings from waste only.