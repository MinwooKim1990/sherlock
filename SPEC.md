# SPEC — Project Sherlock System Specification

> Version: v0.4 · 2026-06-19 (adds M11 — perception v1.5 + Quiescence Gate v1.6)
> Authoritative source for what the system does and how it is structured.
> If this document conflicts with anything you remember from training data, this document wins.

---

## 0. Purpose of this document

This is the system specification — what the system does and how it is structured. Implementation choices not explicitly specified here are left to judgement; when in doubt, prefer the option that better serves the one-line definition below.

---

## 1. One-line definition

> **A domain-agnostic context-curation library where the user provides only the main LLM's system prompt, and Sherlock autonomously designs how its background reasoning models (summarizer and intent-inferrer) should operate. It saves tokens by curating memory, retains meaning by inferring before the user has to explain, and improves over time by adapting its inference style to this specific user.**

The system is named after Sherlock Holmes. It treats every user message as evidence of a larger situation and reasons about what was not said.

---

## 2. Philosophy

These principles override implementation details when they conflict:

1. **The user writes one thing only**: the main system prompt. Everything else — companion prompts, retrieval policies, decay tuning — is derived.
2. **Main LLM responds fast; everything else is async background work.**
3. **The system prompt of the main LLM is immutable per session.** Dynamic context lives in slots that are swapped between turns. We do **not** rewrite the main system prompt at runtime — that path leads to silent corruption.
4. **Inferred context is marked as inferred.** The main LLM must always be able to distinguish what the user said from what the system guessed.
5. **The system fades information naturally.** Older memories that are not used and are not semantically connected to active topics fade to RAG storage and eventually disappear, unless pinned.
6. **The system explains itself.** Every inferred memory carries a confidence score, an evidence trail, and a source.
7. **Evolution is gentle.** Companion prompts evolve incrementally, with all versions saved and reversible.
8. **Domain-agnostic.** Whether the main LLM is a coding assistant, a medical triage agent, a customer-support bot, or an in-game NPC, Sherlock should adapt without code changes.

---

## 3. How Sherlock differs from existing systems

| System | Mode | Limitation |
|--------|------|------------|
| mem0 / Letta / MemGPT | Stores explicit user-stated facts | Reactive; cannot prefetch what the user has not said |
| Anthropic Skills | Static pattern-matching for prompt fragments | No inference, no decay |
| LangGraph checkpointer | Persists explicit state | No interpretation of user intent |
| Cursor / Claude Code rules | Path/language-based context loading | No conversation-aware reasoning |
| **Sherlock** | **Inference-driven prefetch + semantic-cluster decay + user-adaptive evolution + autonomous companion-prompt bootstrap** | (intentionally novel; risk profile in §10) |

Individual parts exist elsewhere; the integration does not.

---

## 4. Architecture

### 4.1 Components

| Component | Role | Sync/Async |
|-----------|------|------------|
| `LLM 1` | Main response model. Talks to the user. Also drives Bootstrap. | Sync |
| `LLM 2` | Background summarizer + classifier + retrieval-keyword extractor. | Async |
| `LLM 3` | Background intent-inferrer. Generates ≥3 hypotheses with probabilities. | Async, on-demand |
| `Bootstrap Engine` | At init (and on prompt change), uses LLM 1 to author LLM 2 and LLM 3 system prompts. | One-shot + on-demand |
| `Storage Layer` | SQLite (structured) + Vector DB (embeddings). | — |
| `RAG Engine` | Hybrid (Vector + BM25 + reranker) in v1; Graph layer in v2. | — |
| `Web Search` | Always on. Injects current date/time and fetches fresh info on inferred topics. | Auto |
| `Decay Engine` | Time + semantic-cluster based memory fading (Fresh → Warm → Cold → Forgotten). | Background batch |
| `Tool Layer` | Built-in tools, MCP discovery, custom Python decorators. | On-demand |
| `Evolution Engine` | Updates LLM 2 / LLM 3 system prompts based on user-feedback signals. | Background batch |

### 4.2 Data flows

#### Bootstrap (run once at init; re-run on main-prompt change or user request)

```
[user-authored main_system_prompt.md]
    +
[built-in Sherlock meta-context: how this system works,
 what role LLM 2 and LLM 3 play, what JSON each must emit]
    +
[reasoning reference (Appendix A)]
    +
[domain hints from config (optional)]
    ↓
[LLM 1 performs meta-reasoning]
    ↓
{ llm2_system_prompt, llm3_system_prompt, rationale }
    ↓
[validation: JSON schema, length sanity, required sections]
    ↓
[saved to SQLite as version 1; user may inspect / edit / regenerate]
```

#### Synchronous turn (user-facing)

```
[user input arrives]
    ↓
assemble:
  [immutable system prompt]
+ [current date/time]
+ [pinned user profile]
+ [active intent slot — latest LLM 3 hypothesis]
+ [retrieved memories — RAG top-K]
+ [cached web search results, if any]
+ [last K turns, uncompressed]
+ [current user input]
    ↓
[LLM 1]
    ↓
[response → user]
```

#### Asynchronous turn preparation (after main response sent)

```
[response sent] → background trigger
    ↓
[trigger check: n turns elapsed OR topic change OR context saturated?]
    ↓
[LLM 2: summarize + classify + extract retrieval keywords + decide if expansion needed]
    ↓ (if expansion)
[LLM 3: ≥3 intent hypotheses, tool recommendations, freshness-required topics]
    ↓
[parallel where possible: web search + tool calls + memory consolidation]
    ↓
[memories committed to SQLite + Vector DB]
    ↓
[next-turn slot prepared]
    ↓
[Decay Engine pass: state transitions across all memories]
    ↓ (every N turns)
[Evolution Engine: if user-feedback signals warrant, bump LLM 2/3 prompts to next version]
```

### 4.3 Parallelism policy

```python
if config.has_remote_provider() and not config.local_only:
    # Parallel: companions run concurrently
    await asyncio.gather(summarize_task, infer_task, search_task)
else:
    # Sequential: local hardware can only run one model at a time
    await summarize_task
    if needs_inference:
        await infer_task
    await search_task
```

---

## 5. Core policies

### 5.1 K-turn original retention

The most recent K turns are passed to LLM 1 **uncompressed**. Compression loss never affects the freshest part of the conversation.

- `K_min` (default 3) — fixed lower bound.
- `K_max` adaptive — shrinks when topic changes or when context utilization exceeds 70%, expands during single-topic stretches.

### 5.2 Decay policy (4-state lifecycle)

| State | Location | Used when |
|-------|----------|-----------|
| `fresh` | Directly in LLM 1 context window | Last K turns + active slot |
| `warm` | In slot, as a semantic triple (compressed) | Active topic, recently referenced |
| `cold` | In RAG store only (vector + BM25) | Retrieved by semantic search when needed |
| `forgotten` | Soft-deleted (recoverable for a window) | Unused + topic cluster inactive |

Transitions:
- `fresh → warm` if not directly used next turn
- `warm → cold` after `decay.warm_after_days` (default 7) without reference
- `cold → forgotten` after `decay.cold_after_days` (default 30) **and** semantically isolated from active clusters
- `forgotten → hard delete` after `decay.forgotten_after_days` (default 90)
- Pinned items skip all transitions.

### 5.3 Bootstrap and Evolution

API models cannot be fine-tuned, so adaptation happens at the prompt level.

**Bootstrap (initial autonomous authoring):**
- Inputs: main system prompt + Sherlock meta-context + reasoning reference + domain hints
- Output: companion system prompts (validated JSON) and rationale
- Persisted as version 1 in SQLite
- User may inspect, hand-edit, regenerate, roll back

**Evolution (gradual refinement on top of v1):**
- Every `evolution_interval_turns` (default 20), LLM 2 analyzes recent user reactions ("that's right" / "no that's wrong" / silence / correction)
- Good inferences become positive few-shot examples appended to LLM 3's prompt
- Bad inferences become negative examples
- Frequent user patterns update a `## User-specific patterns` section
- Same mechanism applies to LLM 2
- Each update creates a new version; rollback always available

**Carry-over rule:** when the user changes the main prompt and Bootstrap re-runs, accumulated user-patterns may carry over to the new v1 (configurable, default `true`).

### 5.4 Provider and model selection

Configured per role in YAML. Library queries each provider's model-list endpoint when available so the user can pick from current options. Sane fallback chains are defined for each provider when a model is omitted.

Local providers (Ollama, LM Studio) are first-class. They force sequential execution.

### 5.5 Web search

`search.always_on` defaults to `true`. Every turn, current date/time is injected. When LLM 3 produces `freshness_required` topics, those are queried automatically; results land in the next-turn slot.

Provider is configurable (Tavily, Brave, Serper, Google CSE, Bing).

### 5.6 Tools

**Built-in (always available; required for inference accuracy):**
- `web_search`
- `current_time` (timezone-aware)
- `file_read`
- `calculator`
- `url_fetch`
- `location` (optional)

**MCP discovery:** servers declared in `tools.mcp_servers` are introspected and their tools become available.

**User-registered tools:** `@sherlock.tool` decorator on Python functions; JSON schema is auto-generated.

---

## 6. Data model

### 6.1 Memory entry

```python
{
  "id": "uuid",
  "type": "fact" | "inference" | "search_result" | "tool_output" | "user_utterance",
  "content": "raw text or dict",
  "semantic_triple": ["subject", "relation", "object"] | None,  # for compression
  "source": "user" | "llm_inference" | "search" | "tool",
  "confidence": 0.0~1.0,  # < 1.0 for inferences; 1.0 for direct user utterance
  "topic_cluster_id": "uuid" | None,
  "created_at": "datetime",
  "last_used_at": "datetime",
  "use_count": int,
  "pinned": bool,
  "state": "fresh" | "warm" | "cold" | "forgotten",
  "embedding": [...],        # in vector DB
  "tags": ["..."],
  "turn_id": "uuid",         # which turn produced this
  "rl_signal": "good" | "bad" | "neutral" | None,  # for Evolution
  "evidence": ["..."]        # for inferences: clue trail used to derive it
}
```

### 6.2 Context slot (current-turn injection structure)

```
[1. SYSTEM PROMPT]            ← immutable
[2. CURRENT TIME / DATE]      ← refreshed each turn
[3. USER PROFILE — pinned]    ← user-pinned facts
[4. ACTIVE INTENT]            ← latest LLM 3 hypothesis
[5. RELEVANT MEMORIES]        ← RAG top-K
[6. WEB SEARCH RESULTS]       ← if triggered this turn
[7. LAST K TURNS]             ← uncompressed
[8. CURRENT USER INPUT]
```

### 6.3 Token-efficient injection (3-tier)

To avoid the "lost in the middle" effect (where models lose 10–25% accuracy on info in the middle of long contexts):

| Tier | Position | Content | Compression |
|------|----------|---------|-------------|
| 1 (front) | system + slot | semantic triples + pinned facts | strong |
| 2 (back) | last K turns + current input | natural language | none |
| 3 (external) | RAG | per-chunk | retrieved on demand |

Critical info goes into Tier 1 or Tier 2. Tier 3 carries everything else.

---

## 7. RAG architecture

### 7.1 v1 (through M4) — Hybrid Vector + BM25

- **Vector DB:** Chroma (default), LanceDB (option)
- **Hybrid search:** vector + BM25, fused
- **Reranker:** Cohere Rerank API or `BAAI/bge-reranker-v2` (local)
- **Chunking:** turn-pair + sentence-level (sentence-transformers)
- **Embedding:** `text-embedding-3-small` (default), Cohere, Voyage, BAAI/bge-m3 (options)

Naive vector-only RAG fails ~40% of the time on production workloads; reranker is mandatory.

### 7.2 v2 (post-M9) — Graph layer

- Graph DB: NetworkX (local) or Neo4j (managed)
- Multi-hop reasoning over entity relationships
- Activated only after sufficient memory accumulates (cold-start avoidance)

### 7.3 Filtering inferred memories

Inferred memories are filtered by `inference.confidence_threshold` (default 0.4) before injection into LLM 1. Below threshold, they remain in storage but are not injected.

---

## 8. Interfaces

### 8.1 Python API (core)

```python
from sherlock import Sherlock, Config

config = Config.from_yaml("sherlock.yaml")
agent = Sherlock(config)

# Synchronous response
response = agent.chat("user input")

# Background work runs automatically; completion guaranteed before next turn

# Direct memory operations
agent.memory.pin(memory_id="...")
agent.memory.delete(memory_id="...")
agent.memory.search("query", top_k=5)

# Inspect last turn for debugging
state = agent.inspect_last_turn()
print(state.hypotheses)
print(state.retrieved_memories)
print(state.tokens_used)

# Compare against a baseline (raw chat with same provider, no Sherlock)
result = agent.compare("user input", baseline="raw")

# Evolution control
print(agent.evolution.current_version())
agent.evolution.rollback(version=3)
```

### 8.2 Custom tool registration

```python
@sherlock.tool(name="get_stock_price", description="Returns current stock price")
def get_stock_price(ticker: str) -> dict:
    return fetch_stock(ticker)
```

### 8.3 Config schema (YAML)

```yaml
project: sherlock_default

# The single user-authored prompt asset
main_system_prompt:
  path: "./prompts/main_system_prompt.md"
  domain_hints:                            # optional, improves Bootstrap
    - "used as a coding assistant"
    - "long-running project context"
    - "user mixes Korean and English"

models:
  main:
    provider: anthropic
    model: claude-opus-4-7
    api_key_env: ANTHROPIC_API_KEY
  background_summary:
    provider: anthropic
    model: claude-haiku-4-5-20251001
  background_inference:
    provider: openai
    model: gpt-5

bootstrap:
  auto_run_on_init: true
  regenerate_on_main_prompt_change: true
  carry_over_user_patterns: true
  require_user_confirmation: true     # if true, library does not start
                                      # serving until user signals approval

storage:
  sqlite_path: "./sherlock.db"
  vector_db: chroma
  vector_path: "./sherlock_vectors"
  embedding:
    provider: openai
    model: text-embedding-3-small

search:
  provider: tavily
  api_key_env: TAVILY_API_KEY
  always_on: true
  inject_datetime: true

memory:
  k_turn_min: 3
  k_turn_max_adaptive: true
  decay:
    warm_after_days: 7
    cold_after_days: 30
    forgotten_after_days: 90
  topic_cluster:
    algorithm: hdbscan
    min_cluster_size: 3

tools:
  builtin: [web_search, current_time, file_read, calculator, url_fetch]
  mcp_servers: []

inference:
  evolution_enabled: true
  evolution_interval_turns: 20
  confidence_threshold: 0.4

execution:
  # advisory / NOT enforced today (single background worker; no spend gate):
  parallel_when_possible: true
  max_concurrent_background_tasks: 3
  cost_cap_per_turn_usd: 0.50
  fallback_to_sequential_on_local: true
```

### 8.4 CLI

```bash
sherlock chat                          # interactive mode
sherlock chat --compare                # side-by-side with raw baseline

sherlock bootstrap run                 # force Bootstrap (regenerate companion prompts)
sherlock bootstrap inspect             # show current LLM 2/3 prompts
sherlock bootstrap edit llm2|llm3      # hand-edit
sherlock bootstrap regenerate          # discard current and rebuild
sherlock bootstrap diff <v1> <v2>

sherlock memory list [--state] [--source]
sherlock memory delete <id>
sherlock memory pin <id>
sherlock memory clear --forgotten
sherlock memory export <path> | import <path>

sherlock inspect <turn_id> | last

sherlock evolve status | diff <v1> <v2> | rollback <version>

sherlock config edit | validate | models list

sherlock ui                            # launch Streamlit UI
```

### 8.5 UI

**v1 — Streamlit (Python, fast iteration):**
- Left: chat
- Right (live-updating panel): current slot, LLM 3 hypotheses, memories accumulated this turn, token usage
- Tabs: memory browser (search/filter/delete/pin), compare mode, evolution timeline

**v2 — React + TypeScript (post-v1.0):** same features plus richer visualization.

---

## 9. Milestones

Each milestone has explicit Exit criteria; a milestone is only considered done when its criteria pass.

### M1 — Core skeleton (v0.1)
- Provider abstraction (`providers/base.py` ABC)
- Provider implementations: Anthropic, OpenAI, Gemini, xAI, Ollama, LM Studio (consider `litellm` for the unified path)
- Config schema + loader (YAML + pydantic)
- Bare LLM 1 chat (no memory, no inference)
- SQLite baseline (`sqlmodel`)
- CLI primitives (`chat`, `config`)

**Exit:** `sherlock chat` produces conversation; provider can be switched via config without code change.

### M2 — Memory layer (v0.2)
- Vector DB integration (Chroma)
- Embedding provider abstraction
- LLM 2 summarization cycle (n-turn + topic-change triggers)
- K-turn original retention
- Time-based decay (4-state lifecycle, scheduled batch)
- Memory CRUD via CLI

**Exit:** after a 30-turn conversation, memory accumulates and decays as specified; K-turn retention verified.

### M3 — Bootstrap + Inference engine (v0.3)
- Bootstrap Engine (`bootstrap/engine.py`)
  - Sherlock meta-context document (built-in)
  - Prompt-assembly logic (main + meta + Appendix A + JSON output schema)
  - Output validation (parse, required sections, length sanity)
  - Persistence as v1
- LLM 3 module using the Bootstrap-generated prompt
- LLM 2 → LLM 3 trigger logic ("expansion needed")
- Web search integration (Tavily first)
- JSON output schemas for both LLM 2 and LLM 3
- CLI: `sherlock bootstrap run | inspect | edit | regenerate`
- **Multi-domain test:** run Bootstrap with 2-3 distinctly different main prompts (coding assistant / general chat / customer support) and verify the generated companion prompts diverge meaningfully

**Exit:** Bootstrap produces valid companion prompts; LLM 3 emits hypotheses with confidence and evidence; multi-domain test passes; web search results show in next-turn slot.

### M4 — RAG (v0.4)
- Hybrid search (vector + BM25)
- Reranker integration
- 3-tier injection
- Semantic-triple compression for warm-state memories

**Exit:** retrieval recall on a synthetic benchmark exceeds vector-only baseline; "lost in the middle" mitigation verified by ablation.

### M5 — Async pipeline (v0.5)
- Parallel execution with `asyncio.gather`
- Sequential fallback for local mode
- Background task scheduler
- Per-turn cost cap enforcement

**Exit:** in API mode, perceived latency equals LLM 1 alone (background non-blocking); cost cap actually halts background work when exceeded.

### M6 — Evolution (v0.6)
- User-reaction analyzer (LLM 2 sub-task)
- Automatic prompt versioning + rollback
- Few-shot positive/negative example accumulation
- User-pattern carry-over on main-prompt change

**Exit:** after 100 turns with feedback signals, companion prompts have evolved measurably; rollback works; carry-over preserves patterns across a main-prompt change.

### M7 — Tool affordance (v0.7)
- Built-in tools (6)
- MCP discovery (Claude Desktop-compatible format)
- Custom-function decorator
- Tool outputs persist as memories

**Exit:** an MCP server connects, a custom tool is registered, both are callable via inferred suggestion, results reused across turns.

### M8 — Evaluation platform (v0.8)
- CLI compare mode
- Streamlit UI v1
- Memory browser / delete / pin
- Evolution timeline visualization

**Exit:** user can inspect everything (memories, inferences, response differences) from a single screen.

### M9 — Polish (v1.0)
- HDBSCAN-based semantic-cluster decay
- Token-saving optimizations (consider TOON-format injection)
- Documentation (README, API docs, tutorials)
- OSS release prep (PyPI, license, CI)

**Exit:** PyPI-publishable; an external user can onboard from README alone.

### M10+ (post-v1.0)
- Graph RAG layer
- React/TypeScript UI
- Multi-user (server mode)
- Distributed memory (Redis / Postgres)

### M11 — Perception + dynamic gating (v1.5 / v1.6)
- **Perception layer (v1.5)** — a pure-stdlib, deterministic per-turn sensor:
  OBSERVED facts (date arithmetic, script/locale, structural spans, exact
  arithmetic, freshness keywords) and probabilistic PRIOR cues (anaphora,
  hedging, topic shift) injected into the LLM-1 slot. Off by default →
  slot byte-identical for existing users.
- **Evidence-grounded LLM-3 (v1.5)** — span-grounded evidence cap +
  `premise_conflict` detection (a false user premise routes to web verification).
- **LLM-2 memory-consistency check (v1.5)** — code-first detection of a new
  message contradicting a pinned fact (negation / number divergence, gated by
  topical overlap); optional one-call LLM-2 confirmation.
- **Quiescence Gate (v1.6)** — dynamic companion gating. Two leaky-bucket
  pressure accumulators (intent `p3` / memory `p2`) fed by the free perception
  signals, Schmitt-trigger hysteresis, and geometric decay as emergent dwell
  (NO turn counter). Modes: `off` (byte-identical v1.4), `cold_start` (default —
  single-model until a real signal escalates), `turbo` (all companions every
  turn). LLM-1 always answers regardless, so gating never delays the reply.

**Exit:** off-mode is SHA-identical to v1.4; cold_start stays single-model on
calm turns and escalates the same turn a strong signal appears, de-escalating
via decay with no turn counter; every new feature is behind a default-off
kill-switch and adversarially audited per stage.

---

## 10. Risks and mitigations

### 10.1 Inference silent failure
- **Risk:** LLM 3 produces a wrong inference; LLM 1 treats it as fact; user gets a hallucination.
- **Mitigation:** confidence threshold gates injection; inferences carry `source: "llm_inference"` so LLM 1 can distinguish; Evolution adapts to user corrections; `silent_failure_rate` is monitored.

### 10.2 Bootstrap generation failure
- **Risk:** LLM 1 produces broken or misaligned companion prompts.
- **Mitigation:** JSON schema validation; user inspection before serving (`require_user_confirmation`); all versions persisted; rollback always available; dry-run with sample inputs before activation.

### 10.3 Decay miscalibration
- **Risk:** memory fades too fast (loses useful context) or too slow (accumulates noise).
- **Mitigation:** user pinning + metric-driven auto-tuning by M9.

### 10.4 Cold start
- **Risk:** early in a session, no history → inference is uninformed → low utility.
- **Mitigation:** conservative mode for first N turns (default 10); LLM 3 dormant until threshold reached.

### 10.5 Cost runaway
- **Risk:** background loops infinitely call expensive models.
- **Mitigation:** `cost_cap_per_turn_usd`, daily caps, rate limiter, token caps.

### 10.6 Privacy
- **Risk:** user data sent to remote LLM / search APIs without consent.
- **Mitigation:** local-only mode, PII masking, explicit opt-in flags, `--local-only` CLI.

### 10.7 Provider API drift
- **Risk:** model-name or API changes break the library.
- **Mitigation:** consider `litellm` for the unified path; declare fallback chains; query model lists dynamically.

---

## 11. Open questions (TBD)

These are explicitly left open for later decision.

- Exact wording of the Sherlock meta-context document used in Bootstrap (build it during M3; revise as the agent learns what LLM 1 needs to see).
- Concrete clustering algorithm for semantic-cluster decay (HDBSCAN vs adaptive thresholds).
- Whether companion-prompt versioning should resemble Git (branching, merging) or simple linear version numbers (recommended start: linear).
- Default cost cap value (`$0.50/turn` is provisional).
- v2 graph DB choice (NetworkX local vs Neo4j managed).
- Whether to adopt a token-compression format like TOON for slot injection (M9 evaluation).
- `litellm` adoption vs hand-rolled provider abstraction (decide in M1).
- Streamlit vs Gradio for v1 UI (compare quickly in M8).

---

## Appendix A — Reasoning reference (provided to LLM 1 during Bootstrap)

This is **not** the LLM 3 system prompt. It is reference material that LLM 1 reads while authoring the LLM 3 prompt. LLM 1 selects what is relevant for the specific main role and writes it into LLM 3's prompt in its own way.

### A.1 Five reasoning tools

**Deduction:** explicit facts → necessary conclusion. Example: user is in Korea + current time → "today" maps to a specific local date.

**Abduction (Peirce):** clues → most natural explanation. "If these clues are true, what most plausibly explains them?" Always generate at least 3 hypotheses.

**Bayesian thinking:** each hypothesis has a prior; new evidence updates it to a posterior. Beliefs are degrees, not 0/1.

**Pragmatics (Grice):** extract implied meanings using cooperative principles. "Should I buy?" is rarely a permission request; usually it means "tell me whether I'll regret this."

**RSA (Rational Speech Act):** "Why this exact phrasing?" The choice of words is itself evidence. "Search for it" can mean "verify against fresh sources, not your memory."

### A.2 Eight clue categories

| Clue | What to look for | Examples |
|------|------------------|----------|
| Time | today / tomorrow / deadline / hours | bookings, tickets |
| Place | current location, mobility | "I'm in Sakae now" |
| Prior turn | what just failed | repeated question, anger, correction |
| Long-term tendency | what the user repeatedly wants | "conclusions first", "verified sources" |
| Emotion | irritation / anxiety / urgency | terse messages, profanity |
| Constraints | what's off-limits | "no phone calls", "can't visit in person" |
| Cost / risk | money / time / energy | "is it worth it?" |
| Next action | what they'll do with the answer | buy / move / cancel / argue |

### A.3 Three-hypothesis rule

Always produce at least three hypotheses. Adjust their probabilities by clues. Never collapse to a single guess prematurely.

### A.4 LLM 3 output JSON schema

```json
{
  "hypotheses": [
    {
      "intent": "the user is asking X but actually wants to know Y",
      "probability": 0.6,
      "evidence": ["clue 1", "clue 2"],
      "search_keywords": ["keyword1", "keyword2"],
      "reasoning_type": "abduction|deduction|bayesian|pragmatic|rsa"
    },
    { /* second hypothesis */ },
    { /* third hypothesis */ }
  ],
  "tools_recommended": ["web_search", "calculator"],
  "context_to_expand": ["topic to fetch ahead"],
  "context_to_exclude": ["topic to drop from slot"],
  "freshness_required": ["topic that must be re-searched"],
  "confidence_overall": 0.7,
  "evolution_signals": {
    "user_pattern_observed": "...",
    "good_inference_candidate": true
  }
}
```

### A.5 Use of the output

- `hypotheses` → memories (`type: "inference"`, with confidence, evidence)
- `search_keywords` → web search queue
- `tools_recommended` → tool call queue
- `context_to_expand` → next-turn slot candidates
- `context_to_exclude` → slot exclusion filter
- `freshness_required` → periodic re-search list
- `evolution_signals` → Evolution Engine queue

---

## Appendix B — Tech stack summary

| Area | Primary | Alternates |
|------|---------|------------|
| Language (core) | Python 3.11+ | — |
| Language (UI) | Streamlit (Python) v1, React+TS v2 | Gradio |
| Async | `asyncio` | `anyio`, `trio` |
| LLM SDKs | `anthropic`, `openai`, `google-generativeai` | `litellm` (unified) |
| Vector DB | Chroma | LanceDB, Qdrant |
| BM25 | `rank-bm25` | Elasticsearch |
| Reranker | Cohere Rerank API | `BAAI/bge-reranker-v2` |
| Embedding | `text-embedding-3-small` | Cohere, Voyage, BAAI/bge-m3 |
| Storage | SQLite + `sqlmodel` | Postgres |
| Topic clustering | HDBSCAN | Agglomerative |
| Graph (v2) | NetworkX | Neo4j |
| Web search | Tavily | Brave, Serper, Google CSE |
| MCP | `mcp` Python SDK | — |
| Test | `pytest`, `pytest-asyncio` | — |
| Lint/format | `ruff`, `black` | — |
| CLI | `typer` | `click` |
| Config | `pydantic` + YAML | — |
| Logging | `structlog` | — |
| Cost tracking | `tiktoken` + provider pricing | — |

---

## Final note

Every component exists to answer one question at a time, in two contexts:

> **Runtime:** *"Why is the user saying this, right now?"*
>
> **Bootstrap:** *"Given this main role, how should the companion models reason in service of it?"*

When implementation choices are ambiguous, pick the option that better answers these.
