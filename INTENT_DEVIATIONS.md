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
