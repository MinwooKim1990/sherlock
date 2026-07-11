# Sherlock event stream

Sherlock emits a structured event for every meaningful step of a turn — memory
retrieval, slot assembly, each LLM call, the companion passes, and the whole
deep-research pipeline. The playground uses this to render its live inspector;
you can use it for logging, metrics, tracing, or a UI of your own.

## Consuming events

```python
agent = Sherlock.with_callable(main_chat=..., ...)

def sink(ev: dict) -> None:
    # ev = {"type": str, "actor": str, "turn": int, "data": dict}
    print(ev["type"], ev["actor"], ev["data"])

agent.set_event_sink(sink)          # pass None to disable
```

The sink is **best-effort**: it is called from both the main thread and the
background companion thread, so it must be thread-safe, and any exception it
raises is swallowed (an event sink can never crash a turn). When no sink is
attached, `_emit` is a strict no-op — so instrumenting the codebase with events
is always byte-identical to running without a sink.

### Envelope

| field | meaning |
|-------|---------|
| `type`  | dotted event name, e.g. `deep_research.faithfulness` |
| `actor` | who produced it — `system`, `llm1`, `llm2`, `llm3`, `llm4`, `memory`, `decay`, `perception` |
| `turn`  | the turn index the event belongs to |
| `data`  | event-specific payload (documented below) |

## The four LLM roles

Sherlock is a **bring-your-own-LLM** curator that coordinates up to four models.
Each has a fixed job and its own `actor` tag in the stream:

| role | actor | wired via | job |
|------|-------|-----------|-----|
| **LLM-1** | `llm1` | `main_chat` / `_provider` | answers the user; drives search/fetch/deep-research; the only model that ever replies |
| **LLM-2** | `llm2` | `summary_chat` / `_summary_provider` | the librarian — background compaction, long-term promotion, memory-consistency confirmation, and the deep-research **faithfulness + consistency** verify passes |
| **LLM-3** | `llm3` | `inference_chat` / `_inference_provider` | the scout — background inference/hypotheses, freshness search, deep-research keyword planning, and the opt-in web re-check |
| **LLM-4** | `llm4` | `viz_chat` / `_viz_provider` | the visualizer (v1.12, opt-in) — turns an inline `<<sherlock-viz: …>>` marker into a self-contained HTML/SVG artifact via generate → lint → self-review → repair; unset → falls back to the MAIN provider |

LLM-2, LLM-3, and LLM-4 are optional; when unset they degrade gracefully (the
relevant passes no-op and say so — see `deep_research.verify_skipped`; a viz job
falls back to LLM-1's provider). LLM-1 always answers, so the companions never
delay the user's reply when running async.

> Some management events carry the `memory` actor (a plumbing tag for the
> long-term store), distinct from the LLM roles above.

## Event catalog

### Turn lifecycle
| event | actor | payload |
|-------|-------|---------|
| `turn.start` | system | `user_text` |
| `slot.assembled` | system | `system_tokens`, `k_turn_turns`, `retrieved_count`, … |
| `memory.retrieved` | system | `hits[]` (content + score) |
| `turn.completed` | llm1 | `tokens_used`, `companions_requested[]`, `error?` |

### Companions (background brains)
| event | actor | payload |
|-------|-------|---------|
| `compact.done` | llm2 | `facts[]`, `predicted_directions[]`, `worth_digging?` |
| `compact.error` | llm2 | `turn`, `error` — LLM-2 compaction failed (best-effort; turn unaffected) |
| `infer.done` | llm3 | `hypotheses[]`, `really_asking?`, `anticipated_next[]` |
| `infer.error` | llm3 | `turn`, `error` — LLM-3 inference failed |
| `infer.search.round` | llm3 | one freshness/inference search round |
| `freshness.done` | llm3 | `searches[]` (topic + hit count) |
| `notebook.done` | llm3 | recursive inference-notebook result |
| `decay.done` | decay | `fresh_to_warm`, `warm_to_cold`, `cold_to_forgotten` |
| `memory.consistency` | system | code-flagged contradiction candidates |
| `memory.consistency_confirm_error` | llm2 | `error`, `candidates` — LLM-2 confirm pass failed (fail-open → kept the code candidates) |
| `memory.redaction_failed` | system | `error`, `chars` — a redactor crash; content was **withheld** (fail-closed), not stored raw |
| `perception.observed` | perception | deterministic OBSERVED/PRIOR cues |

### Long-term memory (v1.12)
Cross-session durable facts live in a reserved sentinel scope. LLM-2 *promotes* a
small high-value subset (identity/health, explicit "remember this" directives,
durable preferences/projects) through a code-level taxonomy gate; LLM-1 recalls
them via an always-on USER PROFILE block + a sentinel RAG channel. Natural-language
management (save / forget / wipe) is code-gated with single-use confirm tokens.
On by default (`config.memory.long_term.enabled`); every event below is silent
when the feature is off, incognito, or nothing durable happened.

| event | actor | payload |
|-------|-------|---------|
| `memory.promoted` | llm2 | `count`, `items[]` (each `category` + `content`, ≤120 chars) — durable facts promoted this compaction |
| `memory.remember_cue` | memory | `turn` — a deterministic "remember this" directive was detected in the user turn |
| `memory.saved` | memory | `id`, `category` — a fact saved via the `memory save` management verb |
| `memory.updated` | memory | `old_id`, `new_id` — a stored fact was superseded by a corrected version |
| `memory.delete_pending` | memory | `count` — a `memory forget` matched rows and is awaiting the confirm token |
| `memory.deleted` | memory | `count` — the confirmed forget hard-deleted this many rows (+ their vectors) |
| `memory.wiped` | memory | `count`, `backup_path` — the whole sentinel scope was wiped (auto-export backup path, or `null`) |
| `memory.exported` | memory | `format` (md/json/sql), `count`, `path` (or `null` for a string return) |
| `memory.imported` | memory | `imported`, `skipped` — facts added from an MD/JSON import |

### Visualizations (v1.12, LLM-4)
When enabled (`config.visualization.enabled`), LLM-1 drops an inline
`<<sherlock-viz: description | data hint>>` marker where a visual belongs; the
agent swaps it for a placeholder token and queues a job that LLM-4 renders into a
sandboxed-iframe HTML/SVG artifact (generate → lint → self-review → runtime
repair). Off (default) → the marker protocol is dormant and a stray marker stays
verbatim (byte-identical).

| event | actor | payload |
|-------|-------|---------|
| `viz.pending` | llm4 | `turn`, `viz_id`, `anchor`, `description` — a marker was parsed and queued |
| `viz.repairing` | llm4 | `viz_id`, `round`, `errors[]` — a produced artifact failed the lint; feeding the errors back for a repair round |
| `viz.rendered` | llm4 | `viz_id`, `anchor`, `html`, `validated` (`"static"`), `bytes`, and `turn`/`research_id`/`path` when applicable — a validated artifact (HTML included, capped at `max_html_bytes`) |
| `viz.failed` | llm4 | `viz_id`, `anchor`, `reason` (+ `turn`/`research_id` when applicable) — the job was dropped (timeout / repair budget exhausted); its placeholder degrades to plain text |

### Deep research
The pipeline: `strategy → plan → rounds (search → extract → merge) → synthesis
→ v3 editor → faithfulness (LLM-2) → [web re-check (LLM-3, opt-in)] → consistency
(LLM-2)`.

| event | actor | payload |
|-------|-------|---------|
| `deep_research.approval_needed` / `approved` / `cancelled` | system | gated-start handshake |
| `deep_research.strategy` | llm1 | `objective`, `sub_topics[]`, `scope`, `clarifying_questions[]` |
| `deep_research.strategy_failed` | llm1 | `topic`, `error` — planning failed → empty strategy |
| `deep_research.plan` | llm3 | `languages[]`, `queries[]` (keyword sweep) |
| `deep_research.round` | system | `round`, `hits`, `new_sources`, `key_finding` |
| `deep_research.coverage_steer` | system | `covered`, `total`, `uncovered[]` — a gap the next round is steered at |
| `deep_research.tokens` | system | running token accounting; `final: true` on the post-verify total. `by_stage` covers `strategy/plan/meta_*/editor/faithfulness/consistency/web_recheck` |
| `deep_research.synthesizing` | llm1 | `rounds`, `stop_reason` |
| `deep_research.verified` | llm1 | the v3 same-model editor pass ran (`changed`, `chars`) |
| **`deep_research.faithfulness`** | llm2 | `groups_checked`, `fixes_applied`, `flagged_for_web` — report re-checked against the RAW |
| **`deep_research.web_recheck`** | llm3 | `checked`, `corrected`, `unverifiable` — flagged claims re-verified via web (opt-in tier) |
| **`deep_research.consistency`** | llm2 | `reconciled` — cross-section contradictions unified to one value |
| **`deep_research.verify_skipped`** | system | `stage` (faithfulness/consistency/web_recheck), `reason` (no_raw/no_provider/no_engine) — the accuracy pass was skipped and WHY |
| `deep_research.documents` / `done` | system | final docs + answer |
| `deep_research.input_folded` / `queued` / `inbox_discarded` | system | mid-research message handling |
| `deep_research.failed` | system | `error` |

The **bold** rows are the v1.10/v1.11 accuracy layer. `deep_research_verify`
(config, or the playground **Verify** toggle / `POST /api/verify`) selects the
tier: `off` | `faithfulness` (default) | `faithfulness+web`.
