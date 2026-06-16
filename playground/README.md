# Sherlock Live Inspector

A browser test platform that drives the Sherlock memory system with **any
LLM provider — Gemini, OpenAI, Anthropic, or a local OpenAI-compatible server
(Ollama / LM Studio / vLLM)** — and visualizes the 4-LLM internals **in real
time** — slot assembly, LLM-3 inference, LLM-2 compaction, memory
add/retrieve/decay, deep research, and the carry-forward loop — as you chat.

![flow](https://img.shields.io/badge/zero--build-vanilla%20JS%20%2B%20Tailwind%20CDN-blue)

## What you see

- **Chat** (left): talk to LLM-1; the reply streams back fast.
- **⚡ Flow** (right): a live, color-coded timeline of every internal event in
  arrival order — watch the main reply land, then the background companions
  (LLM-3 → LLM-2 → decay → carry-forward) fire one by one. Click any node for raw JSON.
- **🧱 Slot**: the exact context assembled for LLM-1 (TIER 1–4) with token budget.
- **💬 LLM I/O**: the real prompt / response / tokens / latency for LLM-1/2/3.
- **🧠 Inference**: LLM-3 hypotheses (intent · probability · evidence · reasoning).
- **🗜 Compaction**: LLM-2 summary, extracted facts, persona, predictions.
- **🗃 Memory**: the live memory table with provenance + decay-state chips
  (FRESH / WARM / COLD / FORGOTTEN), pinned, confidence, use-count.
- **↪ Carry**: the pending hypotheses + freshness results seeding the **next**
  turn's slot — this is how the loop closes.

- **🔬 Research**: deep-research runs live — the multilingual search plan,
  per-round cards, token usage by stage, and the final cited synthesis.

You can pick a **different model per role** (Main / Summarizer / Inferencer) —
even **mixing providers** (e.g. local Qwen as LLM-1 with GPT-4o-mini
companions) — and change them live mid-session. The top bar shows cumulative
tokens per role.

## Run

```bash
# from the repo root, in your venv
pip install -e ".[playground,embeddings]"        # fastapi+uvicorn (+ local embedder)
python -m uvicorn playground.server:app --port 8000
# open http://localhost:8000
```

Then in the browser:
1. **Connect at least one provider** — paste a Gemini / OpenAI / Anthropic key,
   or a local base URL (e.g. `http://localhost:11434`), and click **Connect**.
2. Pick a model for each role (providers can be mixed); edit the system prompt.
3. **Start session** → chat. Watch the right-hand panels animate.

> First run downloads the local embedding model (~once) — embeddings run
> on-device (fastembed). API keys stay server-side in the session and are never
> echoed to the event stream; secrets in chat are redacted in the memory panel
> while the raw transcript stays faithful.

## How it works

- The three role callables are wrapped (`playground/providers.py`) to call
  `litellm.completion(...)` against whichever provider/model the session selected
  for that role, and emit an `llm.call` event with the exact I/O — so when
  LLM-2/LLM-3 fire in the background, you see them live.
- The Sherlock core exposes an opt-in `agent.set_event_sink(fn)` probe (added in
  `sherlock/agent.py`, no-op when unused) that streams structured lifecycle events
  (`slot.assembled`, `infer.done`, `compact.done`, `decay.done`, `carry.stored`, …).
- The FastAPI backend (`playground/server.py`) forwards every event from any thread
  to a per-session WebSocket via `loop.call_soon_threadsafe`, and runs each turn in
  a worker thread so events stream while the turn is in flight.

Nothing here changes Sherlock's behavior: the probe is inert unless a sink is set.
