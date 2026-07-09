"""Pydantic config schema and YAML loader.

M1+M2+M3 subset of SPEC.md § 8.3. M2+ fields are declared with sane
defaults so downstream code can read them without breaking when omitted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Auto-load .env from CWD (or any parent) so users can put keys in a project-
# local .env file without exporting in their shell. No-op when no .env exists.
load_dotenv()


class MainPromptConfig(BaseModel):
    path: Path
    domain_hints: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    provider: str
    model: str
    api_key_env: str | None = None
    api_base: str | None = None  # for Ollama / LM Studio / proxies
    # v0.4.0: explicit context-window override. When unset, the slot
    # budget resolver looks this up from CONTEXT_WINDOW_REGISTRY in
    # sherlock/budget.py based on the model id.
    context_window: int | None = None
    # v1.0: the model's max output tokens; maps to the slot budget's
    # output_reserve when the user didn't override that explicitly.
    max_output_tokens: int | None = None

    def resolved_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    # Friendly aliases → litellm's exact provider slug. Open-source-model
    # aggregators are OpenAI-compatible and litellm-native; "together" is the
    # name users reach for, but litellm's slug is "together_ai".
    _PROVIDER_ALIASES = {"together": "together_ai", "togetherai": "together_ai"}

    def litellm_model_id(self) -> str:
        """Return the model id in litellm's expected format.

        litellm expects "anthropic/claude-...", "openai/gpt-...",
        "gemini/gemini-...", "ollama/...", "deepinfra/...", "together_ai/...",
        "openrouter/...", etc. Some providers (openai) don't need the prefix.
        We normalise here.
        """
        prov = self.provider.lower()
        prov = self._PROVIDER_ALIASES.get(prov, prov)
        if prov in {"openai"}:
            return self.model
        return f"{prov}/{self.model}"


class ModelsConfig(BaseModel):
    main: ModelConfig
    background_summary: ModelConfig | None = None
    background_inference: ModelConfig | None = None
    # v1.12 Stage B1: optional 4th role — LLM-4 VISUALIZER (turns an inline
    # <<sherlock-viz: ...>> marker into a self-contained HTML/SVG artifact).
    # Unset → the visualizer falls back to the MAIN provider, so `visualization`
    # can be enabled with no extra model key.
    viz: ModelConfig | None = None


class EmbeddingConfig(BaseModel):
    provider: str = "fake"  # default to fake so tests are hermetic
    # `local`/`fastembed` use a built-in default model when model is None.
    model: str | None = "fake-embedding"
    api_key_env: str | None = None


class StorageConfig(BaseModel):
    sqlite_path: Path = Path("./sherlock.db")
    vector_db: Literal["chroma", "lancedb"] = "chroma"
    vector_path: Path = Path("./sherlock_vectors")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class DecayPolicyConfig(BaseModel):
    warm_after_days: float = 7.0
    cold_after_days: float = 30.0
    forgotten_after_days: float = 90.0
    warm_after_turns: int = 1
    cold_after_turns: int = 12
    forgotten_after_turns: int = 30


class TopicClusterConfig(BaseModel):
    algorithm: str = "hdbscan"
    min_cluster_size: int = 3


class LongTermMemoryConfig(BaseModel):
    """v1.12 Stage A1: cross-conversation LONG-TERM memory.

    LLM-2 promotes a small, high-value subset of session facts (identity,
    health, explicit "remember this" directives, durable preferences/projects)
    into a reserved sentinel scope shared by every conversation. Off by default
    while the feature is staged; the code-level taxonomy gate — never the model
    alone — decides what is durable enough to keep forever.
    """

    # v1.12 release: long-term memory is ON by default — the whole point of the
    # feature is to be on. Set False to disable; the OFF path is byte-identical
    # to pre-v1.12 (no USER PROFILE block, no sentinel RAG search, no telemetry
    # keys), and with an EMPTY sentinel scope the ON path is result-identical too
    # (only a wasted empty search + telemetry differ).
    enabled: bool = True
    # Suppress long-term WRITES only (promotions). Reads are unaffected, so a
    # user can pause accumulating new durable facts without losing recall of
    # what's already stored. Mirrors a browser "incognito" session.
    incognito: bool = False
    # Hard cap on stored long-term rows. Past it, the lowest-confidence/oldest
    # promoted rows are hard-deleted (best-effort) so the store stays bounded.
    cap: int = 200
    # Budget for the (later-stage) injected long-term PROFILE block: at most
    # this many facts / characters reach the LLM-1 slot so it can't crowd out
    # the live conversation. Declared here so the schema is stable across stages.
    profile_max_facts: int = 12
    profile_max_chars: int = 1200
    # Also expose long-term facts to the RAG retrieval channel (semantic recall
    # across conversations), not just the always-on profile block.
    rag_channel: bool = True
    # On wipe_long_term() (and the memory wipe-confirm tool), export the sentinel
    # scope to a Markdown backup file BEFORE deleting, so a wipe is recoverable.
    # v1.12 Stage A4: the export hook has landed (sherlock.memory.portability),
    # so this now defaults True — the whole reason it was False (the promised
    # export didn't exist yet) is resolved. Set False to opt out of the backup.
    auto_export_on_wipe: bool = True
    # Category taxonomy (documentation — enforced in the summarizer's code gate):
    #   user_directive   — the user explicitly asked to remember it (ALWAYS)
    #   identity_health  — name/pronouns/allergies/medical (ALWAYS)
    #   stable_preference / relationship / long_term_project
    #                    — durable only; promoted with confidence≥0.7 + a quote
    #   none             — transient/one-off/speculation → NEVER promoted


class MemoryConfig(BaseModel):
    # v0.3.0 fields — still honoured for backcompat, but the K-turn slot
    # is now driven by SlotBudget when slot_budget_profile is set.
    k_turn_min: int = 3
    k_turn_max: int = 10
    k_turn_max_adaptive: bool = True
    decay: DecayPolicyConfig = Field(default_factory=DecayPolicyConfig)
    topic_cluster: TopicClusterConfig = Field(default_factory=TopicClusterConfig)
    summarize_every_n_turns: int = 3
    # v1.4: the AUTO compaction trigger is fill-based, not turn-based — LLM-2
    # auto-compacts only when the assembled LLM-1 prompt reaches this fraction of
    # the model context window (e.g. 0.80 = 80% full). Below it, the conversation
    # grows append-only and prompt caching keeps the cost low; at it, compaction
    # evicts summarized turns. LLM-1's explicit <<sherlock-companions: compact>>
    # tag still fires compaction anytime. 0.0 disables the auto trigger.
    compact_at_fill_ratio: float = 0.80
    topic_change_similarity_threshold: float = 0.4
    rag_top_k: int = 5

    # v0.6: selective auto-infer safety net. LLM-3 inference is primarily
    # tag-driven (LLM-1 decides via <<sherlock-companions: infer>>), but a
    # vanilla model under-emits the tag, leaving inference dormant. With
    # "smart" (default) the agent ALSO fires infer on a topic shift (the
    # already-computed cosine signal) and the first turn — never every turn,
    # so it costs little but is never silent. "off" = pure tag-driven;
    # "always" = every turn (token-heavy; for demos/debugging).
    auto_infer: Literal["smart", "off", "always"] = "smart"

    # v0.4.0: slot-budget knobs. When `slot_budget_profile == "auto"`
    # the agent picks DEFAULT_PROFILE / SMALL_MODEL_PROFILE based on
    # the main model's context-window size. "default" / "small" force
    # one explicitly. Set per-field overrides via slot_budget_overrides.
    slot_budget_profile: Literal["auto", "default", "small", "8k", "16k", "32k", "off"] = "auto"
    slot_budget_overrides: dict[str, int] = Field(default_factory=dict)

    # v1.0 B4: evict raw turns already covered by an LLM-2 summary from the
    # K-turn tail (they stay in SQLite + memory tools). Kill switch.
    compaction_frontier: bool = True

    # v1.5 Stage 3: LLM-2 memory-consistency check. "off" (default) → no check,
    # slot byte-identical. "code" → pure-code contradiction check of the new
    # message vs pinned facts (negation/number divergence), surfaced same-turn.
    # "code+llm2" → also confirm the rare code-flagged candidates with one LLM-2
    # call (ambiguous-case escalation only).
    memory_consistency_check: Literal["off", "code", "code+llm2"] = "off"

    # v0.5.0: redact secrets/PII before writing to long-term memory/RAG
    # (the raw transcript is never redacted — only the memory write path).
    redact_secrets: bool = False
    # v0.5.0: pin-bucket cap (was hardcoded 18). Safety-critical/system/user
    # pins are protected from demotion regardless of this cap.
    max_pinned: int = 18
    # v0.5.0: granularity of the slot's injected timestamp — coarser values
    # ("minute"/"hour"/"date") improve prompt-cache hits in the volatile zone.
    slot_time_granularity: Literal["second", "minute", "hour", "date"] = "minute"

    # v0.4.0: memory-tier weighting. Tier 1 (always-on pinned/persona)
    # and Tier 2 (entity-indexed) are always 1.0; Tier 4 (RAG fallback)
    # is downweighted so semantic-only matches don't drown precise
    # entity matches.
    memory_tier_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "tier1_always_on": 1.0,
            "tier2_entity": 1.0,
            "tier3_tool_only_when_called": 1.0,
            "tier4_rag_fallback": 0.5,
        }
    )

    # v1.12 Stage A1: cross-conversation long-term memory (off by default).
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)


class SearchConfig(BaseModel):
    """Web-search configuration (v0.3.0 — multi-provider).

    Default is DuckDuckGo (no key needed). Set ``provider`` to one of
    ``duckduckgo`` / ``tavily`` / ``brave`` / ``valyu`` / ``stub``.

    API keys: either pass directly via ``api_key`` (highest priority) or
    via ``api_key_env`` (env var name to read at runtime).

    Per-role override (optional): if a conversation needs the *main* LLM
    and the *inference* LLM to use different search engines (e.g. a
    cheap, fast provider for the main loop and a heavy one for
    inference's cross-verification), set the ``main_*`` / ``inference_*``
    fields. Any unset role-field inherits from the flat field above.
    """

    provider: str = "duckduckgo"
    api_key: str | None = None
    api_key_env: str | None = None
    always_on: bool = True
    inject_datetime: bool = True

    # Per-role overrides (optional). When set they take precedence over the
    # flat fields above for that role only.
    main_provider: str | None = None
    main_api_key: str | None = None
    main_api_key_env: str | None = None
    inference_provider: str | None = None
    inference_api_key: str | None = None
    inference_api_key_env: str | None = None

    # v0.7: search-depth knobs.
    # Upper bound on a model-chosen result count (`<<sherlock-tool: search "q" k=8>>`).
    max_results_cap: int = 10
    # `deep_research` tool — code-level deep loop (approval-gated).
    deep_research_require_approval: bool = True
    deep_research_max_rounds: int = 20
    deep_research_results_per_round: int = 6
    deep_research_fetch_top_m: int = 3
    # v0.8: multilingual wide-then-narrow search + fetch discipline.
    # `deep_research_languages`: None → LLM-3 planner picks search languages by
    # topic (≥2, never just the user's). Or pin e.g. ["ko", "en", "ja"].
    deep_research_languages: list[str] | None = None
    deep_research_keyword_queries: int = 6  # planned keyword queries (round-1 sweep)
    deep_research_round1_max_searches: int = 12  # hard cap on the round-1 fan-out
    deep_research_fetch_min_hits: int = 4  # fetch pages only when a round is this thin
    # v1.0 C0: LLM-1 drafts a short research STRATEGY (sub-topics, scope,
    # clarifying questions) before the run — a guideline, never a cage.
    deep_research_strategy: bool = True
    # v1.1 R17: compress fetched page text with LLMLingua-2 when the optional
    # `sherlock[compress]` extra is installed — more information in the same
    # prompt budget. Plain truncation fallback = zero behavior change.
    deep_research_compress: bool = False
    # v1.4: never discard a round's raw fragments (search snippets + fetched
    # excerpts). They are kept per sub-topic and RE-READ at synthesis, so the
    # final report can recover a concrete detail (an event name/date) that a
    # small model under-extracted into "facts" that round. Off → exact v1.3
    # facts-only synthesis. The raw bucket per section is capped (below), so this
    # adds recovery without unbounded prompts.
    # NOTE: the v1.10 verify layer (deep_research_verify) checks the report against
    # this RAW; turning reconstruct_from_raw OFF therefore ALSO disables faithfulness
    # verification (no raw to check against → the pass no-ops), even with verify set.
    deep_research_reconstruct_from_raw: bool = True
    deep_research_raw_char_budget: int = 8000  # per-section deduped raw cap
    # v1.4: let the strategy expand each sub-topic into a "what we need to know"
    # checklist that sharpens the round questions. Off → exact v1.3 strategy.
    deep_research_knowledge_checklist: bool = True
    # Deep-research final EDITOR pass (default ON; set False for the plain baseline
    # synthesis). After the round loop writes the report, one LLM-1 pass (a) grounds
    # every number/name to a gathered fact ([reconstructed] otherwise), (b) enforces
    # cross-section AND temporal consistency (no event both 'played' and 'upcoming';
    # no 'eliminated' vs 'can still advance'; re-derives computed figures like goal
    # difference), (c) DELETES hollow punt sections ("no data — consult the official
    # site"), and (d) leads with a direct verdict. Chosen over earlier facet-steer /
    # grounded-verify / reflexion experiments in a blind A/B eval. Editor-only: the
    # synthesis paths are untouched, so OFF is byte-identical to the baseline.
    deep_research_v3: bool = True
    # v1.10 — structured per-entity extraction (default ON; OFF = byte-identical
    # legacy {"fact","sources"} schema). Each fact MAY carry an `entity` (its single
    # subject — a city/person/team) + `attrs` so a bound attribute (a date, a score)
    # stays welded to ITS entity, stopping small-model entity-binding swaps. Additive
    # metadata: advisory only, never gates whether a fact is kept.
    deep_research_structured_extraction: bool = True
    # v1.10 — freshness (default ON; OFF = dates captured but never surfaced →
    # prompts byte-identical). Surfaces each source's reported date in round snippets
    # + the synthesis raw block so the model can prefer the freshest source for
    # time-sensitive claims and flag stale-as-current (e.g. a past event written as
    # upcoming). Dates are OPAQUE strings (never parsed/compared in code); nothing is
    # filtered or dropped on date — only annotated/weighted by the model.
    deep_research_freshness: bool = True
    # v1.10 — harvest a lead image even on RICH rounds (default ON; OFF =
    # byte-identical). og:image is only captured when a page is fetched, and fetches
    # otherwise fire only on thin rounds — so info-rich queries never got an image.
    # This fetches the top hit ONCE per round (only if nothing else was fetched) to
    # grab its og:image (+date); failures are swallowed.
    deep_research_fetch_image: bool = True
    # v1.10 — LLM-2 verify (the accuracy core). After the v3 editor, a SEPARATE
    # cross-model pass re-reads the report against the gathered RAW (per sub-topic) and
    # fixes mis-extractions (report says X, raw says Y) + contradictions the same-model
    # editor misses — verbatim-span fixes only, NON-DESTRUCTIVE (corrects, never deletes),
    # capped, 0.3 shrink guard. It then runs a FINAL whole-report consistency sweep that
    # reconciles any fact stated two ways across SECTIONS (a date, a name, a yes/no) to
    # one value (사실의 통일성). "off" = skip both (byte-identical). "faithfulness" = the
    # no-web check + consistency sweep (default). "faithfulness+web" = ALSO re-verify the
    # FEW remaining flagged claims via LLM-3 web search before the consistency sweep.
    deep_research_verify: Literal["off", "faithfulness", "faithfulness+web"] = "faithfulness"
    # v1.10 — cap on how many faithfulness-flagged claims the "faithfulness+web" tier
    # re-verifies via LLM-3 web search (the FEW, per design — bounds latency/cost).
    deep_research_web_recheck_max: int = 3
    # v1.10 — persist this run's raw fragments to SQLite for POST-HOC recall (ask the
    # agent later "what else did you find?"). NOT needed for the verify pass (raw is
    # in-memory then). Default OFF — it's storage growth, not an accuracy feature.
    deep_research_persist_raw: bool = False
    # v1.11 — run a round's searches CONCURRENTLY instead of one-at-a-time (default ON).
    # A round issues up to `deep_research_fetch_top_m`/round-1-cap independent queries;
    # they have no ordering dependency, so a bounded thread pool runs them in parallel
    # and results are collected in QUERY ORDER — the hit list, RRF (_q/_rank) tagging,
    # and cross-round URL dedup are byte-identical to the serial path. Only wall-clock
    # changes — a round's searches run up to 6-wide (a fixed internal cap that also
    # bounds the burst for rate-limited engines), so a 12-query round-1 sweep is ≈÷6,
    # not ÷12. A dedicated pool is used so a hung search can't starve the shared tool
    # pool. OFF = the exact serial loop (byte-identical).
    deep_research_parallel_search: bool = True


class InferenceConfig(BaseModel):
    evolution_enabled: bool = True
    evolution_interval_turns: int = 20
    confidence_threshold: float = 0.4
    cold_start_turns: int = 0  # M2 keeps this loose; M3 raises to 10
    # v0.7: LLM-3 background iterative inference-search loop (self-evaluating).
    max_search_rounds: int = 10
    search_results_per_round: int = 4
    # v1.5 Stage 2: evidence-grounded LLM-3. When on, the perception OBSERVED
    # block is fed to LLM-3, the prompt requires a VERBATIM quote per hypothesis,
    # and any hypothesis without a verifiable quote is capped to ≤0.35. Off →
    # LLM-3 prompt + output byte-identical (kill switch).
    evidence_grounding: bool = False
    evidence_grounding_cap: float = 0.35
    # v1.5 Stage 2: premise/knowledge-gap detection. When on, the prompt gains a
    # `premise_conflict` field — topics where a user premise conflicts with the
    # model's knowledge → routed to the existing inference-search loop for
    # external verification (gap detection, not silent "correction"). Off → the
    # DEFAULT_LLM3_PROMPT and schema stay byte-identical.
    premise_conflict: bool = False
    # v1.5 Stage 4: recursive inference notebook. When on, LLM-3 (background only)
    # deepens high-value open questions over a few grounded rounds and rides a
    # "half raw reasoning / half conclusions" notebook to the next turn's slot.
    # Off → never runs, slot byte-identical. Deep research is untouched (mirror).
    inference_notebook: bool = False
    notebook_max_rounds: int = 3


class CompanionsConfig(BaseModel):
    """v1.6 — dynamic companion gating ("Quiescence Gate").

    Decides per turn whether the BACKGROUND companions (LLM-2 compaction, LLM-3
    inference + notebook + proactive search) wake up. LLM-1 always answers
    immediately regardless — this only gates the background brain, so it never
    delays the user's reply.

    Modes:
      - ``"off"``        → byte-identical to the legacy default (smart auto_infer
                           + fill-ratio compaction gate). For migration safety.
      - ``"cold_start"`` → DEFAULT. Two leaky-bucket pressure accumulators (intent
                           ``p3`` / memory ``p2``) fed by the free perception
                           signals; Schmitt-trigger hysteresis; geometric decay =
                           emergent dwell (NO turn counter). A strong single
                           signal (e.g. a stock-price freshness cue) crosses the
                           escalate threshold the SAME turn — nothing is delayed.
      - ``"turbo"``      → the prior all-on: every turn fires {compact, infer} +
                           the deep tier (notebook + proactive search).
    """

    mode: Literal["off", "cold_start", "turbo"] = "cold_start"
    # Deployment-time model-strength profile (static config, NOT a runtime index).
    # A weak model lowers the intent escalate threshold so more turns get help.
    profile: Literal["strong", "weak"] = "strong"
    # Geometric decay per quiet turn: intent is message-local (fast), memory
    # integrates (slow). De-escalation IS this decay — never a turn count.
    decay3: float = 0.5
    decay2: float = 0.8
    # Schmitt thresholds: escalate at esc, stay loud until pressure < deesc.
    esc3: float = 0.6
    deesc3: float = 0.3
    esc2: float = 0.30
    deesc2: float = 0.15
    # Deep tier (notebook + proactive search) needs ≥2 strong signals THIS turn.
    esc3_deep_signals: int = 2
    # weak-profile override for esc3 (applied when profile == "weak").
    esc3_weak: float = 0.45


class PerceptionConfig(BaseModel):
    """v1.5 Stage 1: pure-stdlib per-turn perception layer.

    Deterministic OBSERVED facts (date arithmetic, script/locale, structural
    spans, exact arithmetic, freshness keywords) + probabilistic PRIOR cues
    injected into the LLM-1 slot (and, from Stage 2, LLM-3). ``enabled``
    defaults to ``False`` so the slot stays byte-identical for existing users;
    the playground turns it on. Per-primitive toggles let a noisy primitive be
    silenced without disabling the layer.
    """

    enabled: bool = False
    max_observations: int = 12
    dates: bool = True
    scripts: bool = True
    arithmetic: bool = True
    spans: bool = True
    code: bool = True
    discourse: bool = True
    freshness: bool = True


class VisualizationConfig(BaseModel):
    """v1.12 Stage B1: LLM-4 VISUALIZER — inline data visualizations.

    When LLM-1 answers a question where a diagram/chart would genuinely help, it
    drops an inline ``<<sherlock-viz: description | data hint>>`` marker at the
    spot the visual belongs. The agent replaces each marker with a stable
    placeholder token (``⟦viz:...⟧``) that survives markdown rendering, stashes a
    per-marker job, and (from Stage B2 onward) LLM-4 renders each job into a
    self-contained HTML/SVG artifact swapped in for its placeholder.

    ``enabled`` defaults to ``False`` so the marker protocol + system-prompt
    guidance are completely dormant — a stray marker in an LLM reply stays
    verbatim exactly as it does today (byte-identical off-state). The playground
    flips it on. Every knob below is a bound, not a behaviour change when off.
    """

    enabled: bool = False
    # LLM-4 self-critique rounds before a render is accepted (Stage B2+).
    self_review_rounds: int = 1
    # Max render-repair attempts when a produced artifact fails validation.
    max_repair_rounds: int = 2
    # Hard wall-clock budget for one marker's render (seconds); best-effort —
    # a timeout drops that visual, the placeholder degrades to plain text.
    timeout_s: float = 30.0
    # Cap on how many markers are honoured per CHAT reply / per deep-research
    # REPORT. Markers beyond the cap are stripped (no placeholder), never queued.
    max_markers_chat: int = 3
    max_markers_report: int = 4
    # Upper bound on a single rendered artifact's HTML payload (bytes).
    max_html_bytes: int = 64_000
    # Persist rendered artifacts to storage so a reopened session can re-hydrate
    # them (Stage B3+). Best-effort; off = render-and-forget.
    save_artifacts: bool = True


class ToolsConfig(BaseModel):
    builtin: list[str] = Field(
        default_factory=lambda: ["web_search", "current_time", "calculator", "url_fetch"]
    )
    mcp_servers: list[str] = Field(default_factory=list)


class BootstrapConfig(BaseModel):
    auto_run_on_init: bool = True
    regenerate_on_main_prompt_change: bool = True
    carry_over_user_patterns: bool = True
    require_user_confirmation: bool = False  # default False so dev/eval flow doesn't block


class ExecutionConfig(BaseModel):
    # NOTE (honesty): the following three are ADVISORY / NOT ENFORCED today. The
    # background companion worker is a single-thread executor (max_workers=1) for
    # deterministic replay, so `parallel_when_possible` /
    # `max_concurrent_background_tasks` do not change concurrency, and
    # `cost_cap_per_turn_usd` is not wired to any spend check (no per-turn USD
    # cap is enforced). They are kept for forward-compat config stability; treat
    # them as documentation of intent, not active controls.
    parallel_when_possible: bool = True  # advisory — not enforced (single bg worker)
    max_concurrent_background_tasks: int = 3  # advisory — not enforced (single bg worker)
    cost_cap_per_turn_usd: float = 0.50  # advisory — NOT enforced (no spend gate)
    fallback_to_sequential_on_local: bool = True  # advisory — not enforced
    # v0.5.0: run companions (LLM-2/LLM-3) + decay in a background worker so
    # chat() returns the main reply immediately. Default True (v1.8): the
    # user-facing reply must never wait on companion work. Set False for inline
    # execution (deterministic — used by tests/eval/replay, or when a caller
    # wants to inspect companion output synchronously right after chat()). The
    # bg worker uses non-daemon threads, so concurrent.futures' atexit hook
    # drains pending work on normal exit — no memory loss for a script that
    # exits right after chat().
    background: bool = True
    # How long chat() waits at turn start for the PRIOR turn's background to
    # land its pending context before proceeding without it (seconds).
    background_pending_wait_s: float = 2.0
    # v0.5.0: per-conversation cumulative tool-call cap (on top of the
    # per-turn round cap). 0 = unlimited.
    max_tool_calls_per_conversation: int = 100
    # v0.7: tool-tag rounds per turn (was the _MAX_TOOL_ROUNDS_PER_TURN const)
    # + hard timeout for a single search/fetch call.
    max_tool_rounds: int = 3
    tool_timeout_s: float = 20.0


class Config(BaseModel):
    """Root config. M1-relevant fields are required; M2+ fields default to spec values."""

    project: str = "sherlock_default"
    main_system_prompt: MainPromptConfig
    models: ModelsConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    companions: CompanionsConfig = Field(default_factory=CompanionsConfig)
    # v1.12 Stage B1: LLM-4 visualizer (off by default → dormant marker protocol).
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("main_system_prompt", mode="after")
    @classmethod
    def _check_prompt_exists(cls, v: MainPromptConfig) -> MainPromptConfig:
        if not v.path.exists():
            raise ValueError(f"main_system_prompt.path does not exist: {v.path}")
        return v

    def read_main_system_prompt(self) -> str:
        return self.main_system_prompt.path.read_text(encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp) or {}
        # Resolve relative paths against the YAML file's directory so configs
        # stay portable.
        base = path.parent
        if isinstance(raw.get("main_system_prompt"), dict):
            p = raw["main_system_prompt"].get("path")
            if p and not Path(p).is_absolute():
                raw["main_system_prompt"]["path"] = str((base / p).resolve())
        if isinstance(raw.get("storage"), dict):
            for key in ("sqlite_path", "vector_path"):
                p = raw["storage"].get(key)
                if p and not Path(p).is_absolute():
                    raw["storage"][key] = str((base / p).resolve())
        return cls.model_validate(raw)
