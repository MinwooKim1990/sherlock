# Sherlock — Python package usage

> Bring-your-own-LLM agentic memory. You hand Sherlock a chat function;
> it gives you back an agent that remembers past turns, compacts older
> context, and runs Sherlock-style implicit-ask inference in the
> background. Works with any LLM (Anthropic, OpenAI, Ollama, local
> models, your own gateway — anything callable).

## Install

From a checkout:

```bash
cd project_sherlock_spec
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

The package targets Python 3.12 (3.11 / 3.13 also work). Heavy
dependencies (chromadb, sentence-transformers, sqlmodel, litellm) are
pulled in; one-time install takes ~1-2 minutes.

## 30-second example

```python
from sherlock import Sherlock

def my_llm(messages):
    """Receive list of {"role": ..., "content": ...}; return text."""
    # Example: call Anthropic
    import anthropic
    client = anthropic.Anthropic()
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    r = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=sys,
        messages=chat,
    )
    return r.content[0].text

agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="You are a candid, casual assistant.",
)

print(agent.chat("hi"))
print(agent.chat("what did i just say?"))   # Sherlock will have the history
```

That's it. Sherlock handles:

- Per-turn message store (SQLite)
- Background compaction (LLM-2 — called by the same `main_chat` by default)
- Sherlock-style inference (LLM-3) — generates ≥3 hypotheses about the
  user's underlying ask whenever surface meaning ≠ actual ask
- Provenance tracking — distinguishes facts the **user stated** from
  facts the **system inferred** or read from a **persona note**
- Memory decay — old, unreferenced turns fade through fresh → warm →
  cold → forgotten lifecycle
- A `<<sherlock-companions: compact, infer>>` tag your LLM can emit at
  the end of its reply to request background work (the tag is stripped
  from the user-visible response)

## Use different models for the companions

```python
def chat_via_main(messages): ...    # claude-opus for user-facing replies
def chat_via_companion(messages): ...   # gpt-mini for compaction / inference

agent = Sherlock.with_callable(
    main_chat=chat_via_main,
    summary_chat=chat_via_companion,
    inference_chat=chat_via_companion,
    system_prompt="You are a helpful assistant.",
)
```

## OpenAI example

```python
from openai import OpenAI
client = OpenAI()

def my_llm(messages):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return r.choices[0].message.content

agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="You are concise and direct.",
)
```

## Ollama (local) example

```python
import requests

def my_llm(messages):
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3", "messages": messages, "stream": False},
        timeout=120,
    ).json()
    return r["message"]["content"]

agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="You are a helpful local-model assistant.",
)
```

## Async callable

```python
import anthropic

aio = anthropic.AsyncAnthropic()

async def my_llm(messages):
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    r = await aio.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=sys,
        messages=chat,
    )
    return r.content[0].text

agent = Sherlock.with_callable(main_chat=my_llm, system_prompt="…")
# Sync chat() will run the async fn under-the-hood:
agent.chat("hi")
```

For true async pipelining (LLM-1 + LLM-2 + LLM-3 in parallel), use
`agent.achat("...")` instead of `agent.chat("...")`.

## What your LLM can do via the tag

At the end of any reply, your LLM may emit (on its own line):

```
<<sherlock-companions: compact, infer>>
```

- `compact` — triggers LLM-2 to summarise recent turns into structured
  facts (gets persisted with PIN/ACTIVE/BACKGROUND/DROP classification).
- `infer` — triggers LLM-3 to generate ≥3 hypotheses about the user's
  underlying ask, with confidence + evidence trail.

Either, both, or neither. Sherlock strips the tag before passing the
reply back to the user. Teach your LLM about this via your system
prompt — see `prompts/main_system_prompt.md` for a template that
documents the convention.

If your LLM never emits the tag, Sherlock auto-fires both companions
on the very last turn (safety net so memory is never empty).

## Inspecting state

```python
# Conversation history
for m in agent.messages():
    print(m.role, m.content[:80])

# Memory entries (facts, inferences, summaries)
for m in agent.memory.list():
    print(m.type, m.source, m.state, "—", m.content[:80])

# Last turn snapshot
state = agent.inspect_last_turn()
print("hypotheses:", state.hypotheses)
print("retrieved:", [e.content for e, _ in state.retrieved_memories])
```

## Storage location

By default `Sherlock.with_callable(...)` creates a fresh temp
directory per process — state is ephemeral. Pin it to a permanent
directory to keep history across runs:

```python
agent = Sherlock.with_callable(
    main_chat=my_llm,
    system_prompt="…",
    storage_dir="/Users/me/.local/share/my_app/sherlock",
)
```

## YAML-driven path (advanced)

If you'd rather configure everything in a single file (model providers,
embedding model, decay policy, web search, etc.), use `sherlock.yaml`:

```python
from sherlock import Sherlock
agent = Sherlock.from_yaml("sherlock.yaml")
```

See `sherlock.example.yaml` in the repo for the full schema.

## CLI

The package also installs a `sherlock` command:

```bash
sherlock chat --config sherlock.yaml
sherlock config validate
sherlock config show
sherlock models
sherlock evaluate --config sherlock.yaml --conversation evaluation/dummy_conversation.md
```

## Validation

Sherlock has been validated end-to-end against a 80-turn synthetic
benchmark (`evaluation/dummy_conversation.md` + `evaluation/gold_standard.md`).
With Claude Opus 4.5 workers the system scores 82/100 against the
gold standard, passing the spec's 80% gate. See `logs/REPORT.md` and
`logs/REPORT.html` for the full 22-loop trajectory and architectural
evolution.

## Limits

- The architecture works best on conversations with a coherent persona
  and multiple topic threads. Random short exchanges produce little for
  the companions to compact / infer over.
- The provenance ledger distinguishes user-stated vs system-source
  facts but does not verify external claims (it can't detect that
  "Yujin is 4 years old" is wrong if the user says it).
- Memory decay is time- and turn-count based; semantic-cluster decay
  (HDBSCAN) is in the SPEC but not yet wired.
- Evolution Engine versions companion prompts but does not yet learn
  from user feedback signals automatically.
