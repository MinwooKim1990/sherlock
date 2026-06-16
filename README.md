# Sherlock

**Sherlock is a context layer for any LLM.** Wrap one chat function —
keep your model, keep your stack — and get persistent memory, compacted
context, implied-intent inference, multilingual deep research, and a
live inspector that shows **exactly what your model saw and why**.

```python
agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="...")
agent.chat("hi")   # that's the whole integration
```

Sherlock never picks models for you and never trims results to save
money. **You own the model choice; Sherlock owns the context.** Token
savings come exclusively from eliminating *waste* (re-sent material,
duplicated context, lost calls) — never from capping what you get back.

## When to use Sherlock

Use Sherlock if you are building an assistant that needs to:

- **remember users** across long conversations — and keep the per-turn
  prompt bounded while it does (compaction frontier: old turns leave the
  prompt, never the database);
- **make small / local models behave smarter** (Ollama, LM Studio, vLLM,
  llama.cpp) without switching providers — plain-text tag protocol, honest
  8K/16K/32K context budgets, JSON-repair retries;
- **debug exactly what context the model saw** on any turn — live
  inspector + one-click session export, plus a built-in A/B mode that runs
  the same prompt against the same model *without* Sherlock;
- **do real research**: planned multilingual search, source triangulation,
  citation verification, approval-gated deep research.

## Where Sherlock fits

| You need | Best fit |
|---|---|
| A full stateful agent runtime / platform | Letta |
| A drop-in managed memory API | Mem0 / Supermemory |
| Enterprise temporal knowledge-graph memory | Zep / Graphiti |
| LangGraph-native memory | LangMem |
| **BYO-LLM context assembly + a live context inspector, no framework lock-in** | **Sherlock** |

Sherlock deliberately does not compete on hosted-memory-API convenience or
agent-runtime breadth — it wins when you want to keep your own callable and
*see* (and debug) every token of context your model receives.

## How it works

Three LLM roles, all wired by you (they can be one function or three):

```
            ┌────────────────────────────────────────────────┐
 user ──────►  LLM-1 · main chat                              │
            │  answers using the assembled context slot:      │
            │  [TIER 1] your system prompt + Sherlock protocol│
            │  [TIER 2] pinned facts · persona · highlights   │
            │  [TIER 3] hypotheses · fresh search results     │
            │  [TIER 4] last N raw turns (dynamic budget)     │
            └───────┬────────────────────────────┬───────────┘
                    │ background                  │ background
            ┌───────▼──────────┐         ┌───────▼──────────┐
            │ LLM-2 · compactor │         │ LLM-3 · inferrer │
            │ summaries, facts, │         │ ≥3 hypotheses on │
            │ persona profile   │         │ the real ask;    │
            │ (pin/active/drop) │         │ search planning  │
            └───────┬──────────┘         └───────┬──────────┘
                    └────────► memory ◄──────────┘
                      SQLite + vector store · decay
                      (fresh → warm → cold → forgotten)
```

- **Memory with provenance** — user-stated facts are never confused with
  system inferences; each prompt block carries the source and the turn it
  was learned (`(user t12)`), so newer facts win on conflict.
- **Slot budgets** — context is assembled against real token ceilings
  per tier; the raw-turn tail takes whole turns only (no mid-thought
  truncation).
- **Tag protocol** — your LLM drives everything with plain-text tags
  (`<<sherlock-companions: …>>`, `<<sherlock-tool: …>>`); no native
  function-calling required, which is exactly what small models handle
  best. Native tool-calling adapters exist too.
- **Deep research** — an approval-gated, multilingual, multi-round web
  research loop with a token-frugal shared-state protocol (details
  below).

## Install

From a checkout:

```bash
cd project_sherlock_spec
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .                          # base — incl. free DuckDuckGo search + page fetch
pip install -e ".[embeddings]"            # + real local semantic memory (recommended)
pip install -e ".[embeddings,search]"     # + Tavily provider (Brave/Valyu need only a key)
pip install -e ".[playground]"            # + the Live Inspector web app
```

Or build and share a wheel:

```bash
pip install build && python -m build      # → dist/sherlock_context-1.2.0-py3-none-any.whl
pip install "sherlock-context[embeddings,playground] @ file:./dist/sherlock_context-1.2.0-py3-none-any.whl"
```

> Distribution name is **`sherlock-context`** (PyPI `sherlock` is an
> unrelated locks library); the import stays `import sherlock`.

Targets Python 3.12 (3.11 / 3.13 also work). `litellm` is imported
lazily so `import sherlock` stays fast. The embedding default is
`"auto"`: real local embeddings (fastembed, multilingual, no API key)
when the `[embeddings]` extra is installed, graceful fallback to a
deterministic hash embedder (with a warning) otherwise. DuckDuckGo
search + page fetch work from the base install.

## Quick start (30 seconds)

```python
from sherlock import Sherlock

def my_llm(messages):
    """Receive list of {"role": ..., "content": ...}; return text."""
    import anthropic
    client = anthropic.Anthropic()
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    r = client.messages.create(
        model="claude-haiku-4-5", max_tokens=2048, system=sys, messages=chat,
    )
    return r.content[0].text

agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="You are a candid, casual assistant.",
)

print(agent.chat("hi"))
print(agent.chat("what did i just say?"))   # Sherlock will have the history
```

That's it. Sherlock handles the per-turn message store (SQLite),
background compaction (LLM-2), Sherlock-style inference (LLM-3 — ≥3
hypotheses about the user's underlying ask whenever surface meaning ≠
actual ask), provenance tracking, and memory decay.

### Use different models per role

```python
def chat_via_main(messages): ...        # e.g. a strong model for user-facing replies
def chat_via_companion(messages): ...   # e.g. a small/cheap model for compaction + inference

agent = Sherlock.with_callable(
    main_chat=chat_via_main,
    summary_chat=chat_via_companion,
    inference_chat=chat_via_companion,
    system_prompt="You are a helpful assistant.",
)
```

### OpenAI

```python
from openai import OpenAI
client = OpenAI()

def my_llm(messages):
    r = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return r.choices[0].message.content

agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="You are concise.")
```

### Ollama / any local model

```python
import requests

def my_llm(messages):
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3", "messages": messages, "stream": False},
        timeout=120,
    ).json()
    return r["message"]["content"]

agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="…")
```

### Async

```python
import anthropic
aio = anthropic.AsyncAnthropic()

async def my_llm(messages):
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    r = await aio.messages.create(
        model="claude-haiku-4-5", max_tokens=2048, system=sys, messages=chat,
    )
    return r.content[0].text

agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="…")
agent.chat("hi")          # sync entry point runs the async fn under the hood
# await agent.achat("…")  # native async entry point
```

LLM-1 is awaited synchronously (it gates the reply); LLM-2/LLM-3 + decay
run after the reply in the background.

## 🔍 The playground — Sherlock Live Inspector

A single-page web app for **watching Sherlock think in real time** — the
fastest way to verify the system end to end with real models.

```bash
.venv/bin/python -m uvicorn playground.server:app --reload
# → open http://localhost:8000
```

**Bring any provider — and mix them per role.** Connect one or more:

| Provider | Credential | Notes |
|---|---|---|
| **Gemini** | AI Studio key (`AIza…`) | live model list from your key |
| **OpenAI** | API key (`sk-…`) | chat-capable models, newest first |
| **Anthropic** | API key (`sk-ant-…`) | official models list |
| **Local** | base URL (e.g. `http://localhost:11434`) | any OpenAI-compatible server: Ollama, LM Studio, vLLM, llama.cpp |

Then pick a model for each role — e.g. a local Qwen for LLM-1 with
GPT-4o-mini companions, or Gemini Flash everywhere. Selections can be
changed mid-session from the top bar (takes effect next turn). API keys
stay in the server-side session and are never echoed back to the
browser.

What the inspector shows (one tab per concern):

- **⚡ Flow** — every event in order: turn start, retrieval, slot
  assembly, LLM calls with latency/tokens, tool runs, background work.
- **🧱 Slot** — the exact assembled context LLM-1 received this turn:
  TIER-highlighted system prompt, per-block token budget, K-turn tail.
- **💬 LLM I/O** — the verbatim prompts + responses for all three roles,
  including every internal call of a multi-call turn.
- **🧠 Inference / 🗜 Compaction** — LLM-3 hypotheses with confidence
  bars; LLM-2 summary, facts table (with pin recommendations), persona.
- **🗃 Memory** — the live memory table with decay-state chips,
  provenance, confidence, and use counts.
- **🔬 Research** — deep-research progress: the multilingual search
  plan, per-round cards (queries, new sources/fragments, LLM-3-generated
  questions, facts so far), live **🪙 token usage by stage**, and the
  final cited synthesis + session documents.
- The top bar shows cumulative session tokens per role (`L1 · L2 · L3 · Σ`).

The **"Always run reasoning"** toggle force-fires LLM-2 + LLM-3 every
turn so the panels always fill, even with models that under-emit the
companion tag. Web search defaults to free DuckDuckGo; Brave/Tavily/
Valyu keys can be entered for better results.

## Security & privacy

Sherlock handles transcripts and long-term memory, so the guardrails are
explicit:

- **Tags execute only from LLM output.** Tool/companion tags pasted by a
  user are never parsed — a user cannot trigger searches, fetches, or
  memory reads by typing a tag.
- **Deep research never auto-runs.** It requires an explicit user "yes",
  a UI approve click, or your own `deep_research_approver` callback.
- **Secret/PII redaction** (`redact_secrets=True`) scrubs content before
  it enters long-term memory and RAG; the raw transcript is never altered.
- **Local-first by default**: SQLite + on-device embeddings (fastembed) —
  nothing leaves your machine except the LLM/search calls you wired.
  In the playground, provider API keys live in the server-side session and
  are never echoed to the browser or the event stream.
- **Real deletion**: `delete_session()` cascade-deletes messages, memory
  entries, and vectors. Memory corrections are non-destructive and
  auditable (superseded facts keep their history, marked with the turn
  they were invalidated).
- Untrusted web content is fenced in the prompt ("data, not instructions")
  and every cited URL is verified against the actually-gathered sources.
- *Not yet built* (on the roadmap): encryption at rest, audit logs.

## What your LLM can do via tags

At the end of any reply, your LLM may emit (each on its own line):

```
<<sherlock-companions: compact, infer>>

<<sherlock-tool: search "Seoul weather today">>
<<sherlock-tool: search "nvidia earnings" k=8>>             # set result count (1–10)
<<sherlock-tool: fetch https://example.com/article>>
<<sherlock-tool: fetch raw https://example.com/article>>    # raw HTML

<<sherlock-tool: memory lookup "Yujin 알레르기">>            # semantic + entity recall
<<sherlock-tool: memory entity "Yujin">>                    # deterministic entity match
<<sherlock-tool: memory timeline last 10>>                  # raw recent turns
<<sherlock-tool: memory pinned>>                            # all pinned facts

<<sherlock-tool: deep_research "compare EU vs US AI regulation">>   # approval-gated
```

Sherlock parses tags **only from LLM output, never from user input** (a
user pasting a tag cannot trigger anything), runs the tool, feeds
results back as a synthetic message, and re-calls your LLM — up to
`execution.max_tool_rounds` (default 3) per turn. Tags are always
stripped from the user-visible reply.

If your LLM under-emits the companion tag, a safety net keeps the
companions alive: `compact` auto-fires every N turns
(`memory.summarize_every_n_turns`) and `infer` auto-fires on topic
shifts (`memory.auto_infer`: `"smart"` default | `"off"` | `"always"`).

### Web search engines

```python
agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="...",
    main_search_engine="duckduckgo",        # default; free, no key
    inference_search_engine="brave",        # LLM-3 freshness searches
    inference_search_api_key_env="BRAVE_API_KEY",
)
```

Engines: `duckduckgo` (no key; non-commercial terms, weak for news),
`tavily` (needs `pip install sherlock[search]`), `brave`, `valyu`,
`stub` (tests). Pass `None` to disable search for a role. Native
tool-calling adapters (`make_openai_tools()`, `make_anthropic_tools()`,
`make_openai_memory_tool()`, `dispatch_tool_call`, `dispatch_memory`)
exist for integrations that prefer real function calls.

LLM-3's prompt enforces cross-verification discipline: ≥2 sources per
claim, lowered confidence on disagreement, single-source web facts are
never pinned.

## 🔬 Deep research

When a question needs real depth, LLM-1 *proposes*
`<<sherlock-tool: deep_research "topic">>`. It is **never auto-run**:

1. Sherlock asks the user (playground button, or just reply "yes"/"해줘"
   in a library/CLI session) — or a programmatic
   `deep_research_approver(topic, plan)` callback decides
   (`True`/`False`/`None`=ask). Explicit refusals ("no, don't…",
   "하지마") always cancel, even if they contain trigger-ish words.
   **v1.0:** the approval ask carries a drafted *research strategy*
   (objective + sub-topics) and up to 2 clarifying questions for genuinely
   ambiguous points — answer them alongside your "yes" (or reply with just
   the answer; Sherlock folds it in and re-asks once). The strategy then
   guides the run as a guideline, never a cage.
2. Once approved, the loop runs (in the background when an event sink or
   `background=True` is active):
   - **Multilingual keyword plan** — LLM-3 picks the languages whose web
     most likely holds the answer (a Japan-travel question sweeps
     Japanese + Korean + English) and emits short, particle-stripped
     keyword queries. The *query language* is the i18n lever — global
     search, no locale parameters. Your answer still comes back in
     *your* language.
   - **Wide round 1, narrow after** — round 1 is a broad snippet-only
     sweep; later rounds deepen promising threads. Pages are fetched
     sparingly, only when the round is thin, and never twice. Fragments
     that don't fit a round wait in a backlog — **no result is ever
     dropped**.
   - **Compact shared state (the token saver)** — each round LLM-1 reads
     only the NEW fragments + a compact digest of confirmed facts and
     open gaps, and answers in terse JSON. LLM-3 (from round 3) reads
     ONLY the digest — never raw pages — to generate the next round's
     meta-questions. Nothing is ever re-paid for.
   - **Triangulation** — the same fact found via different
     domains/source types (community / news / official / blog)
     accumulates corroboration; `[corroborated ×N]` facts rank first and
     are stated with higher confidence in the synthesis.
   - **Honest stopping** — stops on `model_sufficient`,
     `converged_no_new_sources` (2 rounds with nothing new),
     `no_next_queries`, `search_engine_error` (2 consecutive all-failed
     rounds — reported as a failure, never disguised as "convergence"),
     or the round cap (≤20).
3. Every round is saved as a `DEEP_RESEARCH` session document (read on
   demand — research never floods the context window or the pinned-fact
   cap), and a single comprehensive cited synthesis closes the run.
   Messages sent mid-research are queued, acknowledged, and folded in at
   the next checkpoint.
4. A `deep_research.tokens` event reports per-stage input/output tokens
   (plan / round answers / meta-questions / synthesis) — visible live in
   the playground — so the cost is measured, not guessed. Background
   failures surface as a real reply (`deep_research.failed`), never
   silence.

```python
agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="...",
    main_search_engine="brave",
    main_search_api_key_env="BRAVE_API_KEY",
    deep_research_approver=lambda topic, plan: None,   # None = ask the user
)
```

## Memory

### Slot budget & dynamic K-turn

The context slot is laid out across four TIERs with explicit token
budgets; whatever remains goes to the raw-turn tail, which accumulates
*whole turns* walking backward (a turn either fits whole or stays out):

```
[TIER 1 — GROUND TRUTH]      sherlock_system + tool_prompt + user_system
[TIER 2 — SYSTEM-TRACKED]    pinned + persona_summary + compacted highlights
[TIER 3 — SPECULATIVE]       inference hypotheses + web search results
[TIER 4 — ACTIVE CONTEXT]    last N raw turns (dynamic, walk-backward)
```

Profiles auto-select by model context window
(`MemoryConfig.slot_budget_profile`: `auto`/`default`/`small`/`off`,
plus `slot_budget_overrides`). Inspect what was used:

```python
state = agent.inspect_last_turn()
print(state.slot_budget)
print(state.k_turn_turns_used, state.k_turn_tokens_used)
print("hypotheses:", state.hypotheses)
```

### Prompt layering

Your `system_prompt` stays the primary system message; Sherlock's
protocol (tag conventions, cross-verify rules) rides alongside it.
Adjust with `extension_position="before"`, replace it via
`sherlock_extension="…"`, or opt out with `sherlock_extension=""`.

### Sessions

```python
for s in agent.list_sessions():
    print(s.id, s.created_at, s.turn_count, "—", s.persona_summary)

agent.new_session()                    # start fresh, keep history
agent.switch_session("abc-123")        # resume an earlier session
agent.delete_session("xyz-789")        # cascade-delete raw + memory + vectors
```

### Inspecting state & storage

```python
for m in agent.messages():
    print(m.role, m.content[:80])
for m in agent.memory.list():
    print(m.type, m.source, m.state, "—", m.content[:80])
```

`with_callable()` defaults to an ephemeral temp directory; pass
`storage_dir="~/.local/share/my_app/sherlock"` to keep state across
runs. Secrets/PII can be redacted before anything enters long-term
memory (`redact_secrets=True`); the raw transcript is never altered.

## YAML + CLI

Configure everything (providers, embeddings, decay, search, budgets) in
one file:

```python
agent = Sherlock.from_yaml("sherlock.yaml")
```

See `sherlock.example.yaml` for the full schema. The package installs a
`sherlock` command:

```bash
sherlock chat --config sherlock.yaml
sherlock config validate | show
sherlock models
sherlock evaluate --config sherlock.yaml --conversation evaluation/dummy_conversation.md
```

## Validation & benchmarks

- End-to-end against an 80-turn synthetic benchmark
  (`evaluation/dummy_conversation.md` + gold standard): **82/100** with
  Claude Opus workers, passing the spec's 80% gate (`logs/REPORT.html`).
- **Ralph v2 behavior probes** — 25 single-capability probes
  (`evaluation/probes/`):

  ```bash
  python -m evaluation.ralph_v2 --probes evaluation/probes/ \
      --config sherlock.live.yaml --report logs/probe.json
  ```

- The full pytest suite (356 tests) runs hermetically — scripted
  callables + fake engines, no network or keys needed: `pytest -q`.
- **Measure it yourself**: the playground's A/B mode runs every prompt
  against the same model with and without Sherlock (the baseline gets the
  same search engine and today's date — a fair control), side by side with
  latency and token counts. We'd rather you compare than take our word.
- Public memory benchmarks (LongMemEval/LoCoMo-style) are on the roadmap
  (`docs/ROADMAP.md`, R28) — we don't publish numbers we haven't run.

## Limits

- Works best on conversations with a coherent persona and multiple topic
  threads; random short exchanges give the companions little to do.
- The provenance ledger distinguishes user-stated vs system-inferred
  facts but does not verify external claims.
- Memory decay is time/turn-based; semantic-cluster decay is specced but
  not wired.
- The Evolution Engine versions companion prompts but doesn't yet learn
  from user feedback automatically.

The prioritized upgrade plan — small-model intelligence, provider prompt
caching, deep-research trust, memory reconciliation — lives in
**[docs/ROADMAP.md](docs/ROADMAP.md)** (R1–R35, evidence-linked).

## Changelog highlights

### v1.1 — the whole roadmap, shipped
- **Small-model reliability**: one-shot examples anchor LLM-2/LLM-3 JSON
  output; constrained JSON decoding is requested automatically where the
  provider supports it (memoized fallback elsewhere); near-miss tool tags
  (`sherlock_tool`, single brackets) are repaired instead of leaking.
- **Deep research trust**: every cited URL is checked against the gathered
  sources — invented citations get an inline `(unverified)` flag; search
  plans and meta-questions actively seek counter-evidence; big runs (>18
  facts + a strategy outline) synthesize **section by section**, each call
  reading only its own facts; evidence is trimmed at sentence boundaries.
- **Memory quality**: retrieval adds recency/importance boosts and 1-hop
  expansion over a new memory-links table (A-Mem style); per-type result
  caps; superseded facts carry `invalid_at_turn` ("superseded at t7") for
  temporal questions; LLM-2 facts can carry a supporting **quote** that is
  verified against the transcript — ungrounded facts get confidence-capped
  and can never be pinned.
- **Token efficiency**: the provenance ledger is skipped entirely when no
  persona facts exist; RAG never re-surfaces pinned facts (TIER-2 already
  carries them); carried-forward search results are relevance-gated; the
  system prompt now marks TWO cache zones (protocol / TIER-2) so pinned-fact
  churn no longer invalidates the protocol cache; optional LLMLingua-2
  compression packs ~2.5× more relevant page text into the same research
  budget (`pip install "sherlock[compress]"`).

### v1.0 — research strategy, fragment reassembly, infinite memory, cache-native prompts
- **Research strategy step**: before a deep research run, LLM-1 drafts a short
  strategy (objective, sub-topics, scope) and asks up to 2 clarifying
  questions alongside the approval — answers fold into the run; the strategy
  is a *guideline, not a cage*. Sub-topics seed the open-gap tracking, so
  coverage is measured by the existing convergence machinery.
- **Fragment reassembly**: fetched pages are excerpted by query relevance
  (comment-buried fragments surface instead of page heads); rephrased facts
  merge their sources (corroboration accumulates across phrasings and
  languages); near-miss contradictions are tagged `[disputed]` and reported
  two-sided; rounds that add no NEW facts converge (`converged_no_new_facts`);
  shown fragments lead with source-type diversity (RRF + round-robin).
- **LLM-2 memory reconsolidation**: the compactor can now emit
  `corrections` that supersede stale pinned facts non-destructively
  (old rows stay queryable, marked `(superseded)`, excluded from prompts and
  dedup); its `retrieval_keywords` now expand the RAG query; Hangul bigram
  BM25 makes Korean agglutinated forms retrievable.
- **Infinite memory (compaction frontier)**: raw turns already covered by an
  LLM-2 summary leave the K-turn tail (the last 4 turns always stay verbatim;
  everything remains in SQLite + memory tools). Measured: −684 tokens/turn at
  turn 14 of a session compacted at turn 8 — savings grow with session length.
- **Honest small windows**: `with_callable(context_window=8192,
  max_output_tokens=…, slot_budget_profile=…)` + new 8K/16K/32K budget
  profiles; the most recent turns bypass the budget so **history is never
  zero**; one-time warning when no window is declared.
- **Cache-native prompts**: the system message marks its byte-stable TIER 1+2
  prefix; LiteLLM converts it to `cache_control` blocks (Anthropic) and
  reports `cache_read/creation_tokens`; LLM-2/LLM-3 prompts are whole-message
  hinted; BYO callables can opt in via a `cache_hints` kwarg — a plain
  `f(messages)` payload stays byte-identical. The playground token bar shows
  `⚡cached`.
- **Waste removal (RTK-style)**: dead LLM-3 output fields removed; provenance
  ledger / message wrappers / tool-result banner compressed (109→52 tokens,
  guardrails intact); protocol docs are now conditional (no search engine →
  1,308→745 tokens/turn); a failed JSON parse retries ONCE with the error fed
  back instead of wasting the whole companion call.

### v0.9 — hardening + universal playground
- Playground is **multi-provider**: Gemini, OpenAI, Anthropic, and any
  local OpenAI-compatible server, mixable per role; cumulative token
  bar; WS auto-reconnect; per-turn LLM call history; IME-safe input.
- Deep research correctness from an adversarial multi-agent audit (30
  confirmed findings fixed): malformed small-model JSON can no longer
  abort a run; refusals never approve; round-1 overflow goes to a
  backlog instead of being dropped; engine outages stop the loop
  honestly; background failures surface as replies; mid-research
  messages are always acknowledged, persisted, and folded or accounted
  for; research docs no longer evict pinned facts.
- Memory integrity: restated facts resurrect from FORGOTTEN; corrections
  past the dedup prefix now *update* the stored fact; LLM-2 output can
  no longer launder into "user-verified" pins; prompt blocks carry the
  turn each fact was learned (`t12`) so newer wins on conflict.
- Korean search queries: only multi-char particles are stripped —
  하와이/제주도/고양이 survive; quoted phrases and version numbers
  (`"exact phrase"`, `3.12`, `C++`) pass through cleaning intact.

### v0.8 — multilingual search + token hygiene
- LLM-3 plans clean keyword queries across the languages most relevant
  to the topic; global search, query language as the lever.
- Compact shared research state: per-round deltas, LLM-3 never sees raw
  pages, terse JSON rounds, synthesis from de-duplicated facts;
  `deep_research.tokens` per-stage measurement.
- Fragment triangulation with `[corroborated ×N]` ranking.

### v0.7 — three search modes
- LLM-1 sets search result count (`k=N`); LLM-3 runs a self-evaluating
  background search loop; `deep_research` ships as an approval-gated
  ≤20-round loop with session documents and meta-cognition Q&A.

### v0.5 — core loop
- True background companions, local embeddings by default, redaction,
  session management, slot budgets, Ralph v2 probes.
