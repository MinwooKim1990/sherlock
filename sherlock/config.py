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

    def litellm_model_id(self) -> str:
        """Return the model id in litellm's expected format.

        litellm expects "anthropic/claude-...", "openai/gpt-...",
        "gemini/gemini-...", "ollama/...", "openrouter/...", etc.
        Some providers (openai) don't need the prefix. We normalise here.
        """
        prov = self.provider.lower()
        if prov in {"openai"}:
            return self.model
        return f"{prov}/{self.model}"


class ModelsConfig(BaseModel):
    main: ModelConfig
    background_summary: ModelConfig | None = None
    background_inference: ModelConfig | None = None


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
    deep_research_reconstruct_from_raw: bool = True
    deep_research_raw_char_budget: int = 8000  # per-section deduped raw cap
    # v1.4: let the strategy expand each sub-topic into a "what we need to know"
    # checklist that sharpens the round questions. Off → exact v1.3 strategy.
    deep_research_knowledge_checklist: bool = True


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
    # chat() returns the main reply immediately. False = inline (deterministic
    # for tests/eval/replay).
    background: bool = False
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
