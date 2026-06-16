# Sherlock v0.4.0 Tier A Behavior Probes

Small, fast, single-capability YAML probes that replace the v0.2.x deterministic-output comparison (which broke when LLM-1 became autonomous and trajectories went non-deterministic).

Each probe = one capability under test. The runner (`evaluation/ralph_v2.py`, separate work) replays `setup`, fires `trigger`, then evaluates `assertions` against the response + memory state.

## Probe schema

```yaml
name: snake_case_unique_id
category: provenance | pin_discipline | tool_discipline | memory_recall |
          cross_verification | tier1_always_on | implicit_ask | topic_shift
description: One sentence describing what's being tested.

setup:
  - role: persona_hint           # seeds domain_hints / persona at startup
    content: "..."
  - role: user                   # played as a user turn before the trigger
    content: "..."
  # ...up to 8 turns

trigger:
  role: user
  content: "the actual test message"

assertions:
  - kind: <one of the 8 kinds below>
    <kind-specific fields>
  # ...up to 5 assertions
```

`persona_hint` entries must precede `user` entries — they seed startup config (`main_system_prompt.domain_hints`), not conversational history.

## Assertion kinds

### 1. `response_contains`
Case-insensitive regex on the trigger turn's response text.
```yaml
- kind: response_contains
  pattern: "(soba|buckwheat|메밀)"
```

### 2. `response_does_not_contain`
Negation of the above. Used to catch failure modes (hallucinated facts, wrong pronouns, banned phrases).
```yaml
- kind: response_does_not_contain
  pattern: "\\b(he|his|him)\\b"
```

### 3. `response_attributes_to_source`
Probes whether the model claims the user said something the system actually said (or vice versa). `expected: system` means the asserted fact must be marked as coming from a system/persona note, not as a user utterance. `either` accepts both.
```yaml
- kind: response_attributes_to_source
  expected: system   # or "user", or "either"
```
Runner implementation: lightweight check that the response uses source-language ("persona note", "context", "wasn't in our conversation") when `expected: system`, and avoids fabricated user-stated attributions ("you told me", "you said").

### 4. `pinned_facts_count_between`
After the trigger turn settles, count entries in the pinned-facts slot.
```yaml
- kind: pinned_facts_count_between
  min: 3
  max: 8
```

### 5. `pinned_facts_include_keyword`
Substring match (case-insensitive) against the pinned-facts slot contents.
```yaml
- kind: pinned_facts_include_keyword
  keyword: "yujin"
```

### 6. `memory_tool_called_with_kind`
Asserts LLM-1 invoked the memory tool during the trigger turn with the given retrieval kind. `lookup` = generic recall; `entity` = person/thing lookup; `timeline` = open-list reconstruction; `pinned` = pinned-slot inspection.
```yaml
- kind: memory_tool_called_with_kind
  tool_kind: lookup    # NB: use `tool_kind` (not `kind`) to avoid YAML collision
```

### 7. `companion_tag_emitted`
Asserts the response emitted a companion-call tag for the named background companion. `compact` = the summarizer; `infer` = the inference/hypothesis companion.
```yaml
- kind: companion_tag_emitted
  name: infer
```

### 8. `confidence_below`
For cross-verification scenarios — the inference engine should produce hedged hypotheses for ambiguous / time-sensitive / disagreement-prone facts. Asserts the max-confidence hypothesis emitted on this turn falls below the threshold.
```yaml
- kind: confidence_below
  threshold: 0.7
```

### 9. `llm_judge` (v0.5.0)
Semantic judgement for qualities regex can't capture (e.g. "genuinely acknowledges the permission-seeking subtext rather than a flat yes/no"). A judge model reads the criterion + the response and returns PASS/FAIL. Requires `--judge-model provider:model`; **skips (counts as pass)** when no judge is configured, so rule assertions still gate cheaply.
```yaml
- kind: llm_judge
  criterion: >
    The reply reframes the buy/no-buy question around fit and regret
    (the user's real workflow) instead of giving a bare yes/no.
```
Run with a judge:
```
python -m evaluation.ralph_v2 --probes evaluation/probes/ \
  --config sherlock.live.yaml --judge-model anthropic:claude-haiku-4-5
```

## Canonical score (v0.5.0)

The **canonical gate** is the `ralph_v2` pass-rate over this probe set
**with `--judge-model` enabled** (rule assertions + semantic judge
together). That single number — printed as `Pass rate: N/M` and written
to the `--report` JSON — is the score of record. The older
`score.txt` (75) and one-off subagent judgements (82) from the v0.2–v0.4
loops are historical and are not the canonical gate.

## Adding a new probe

1. Pick the smallest single capability you want to test.
2. Pick a `category` from the eight above. Add a new one only if the existing eight genuinely don't cover it.
3. Name the file `CATEGORY_short_slug.yaml`. Keep setup ≤8 turns, assertions ≤5.
4. Write assertions that would catch a real failure. If the probe passes trivially without exercising the capability, it's worthless. A useful gut check: a regressed model (or one with the targeted feature disabled) should be able to fail at least one assertion.
5. Prefer Korean naturally in user-turn content — the dummy persona is bilingual and the runner handles it.

## Running

The v0.4.0 driver lives at `evaluation/ralph_v2.py` (separate change). Typical invocation:

```bash
# all probes
python evaluation/ralph_v2.py --probes evaluation/probes/

# one category
python evaluation/ralph_v2.py --probes evaluation/probes/ --category provenance

# one probe by name
python evaluation/ralph_v2.py --probe provenance_t76_name_probe
```

The driver instantiates Sherlock fresh per probe (clean DB, fresh vector store), replays `setup` turns with internal tools/memory enabled, then asks the model the `trigger` turn and grades each assertion against the response and the post-turn memory state. Per-probe pass/fail and an aggregate score are written to `evaluation/runs/<timestamp>/`.

## Salvage map (where the v0.2.x assets ended up)

- **T76 name probe** → `provenance_t76_name_probe.yaml`
- **T67 confabulation trap** → `provenance_unstated_fact_admission.yaml`
- **T55 EpiPen safety catch** → `cross_verification_epipen_correction.yaml` + `pin_discipline_safety_critical.yaml`
- **17 anchor facts** → distributed across `pin_discipline_anchor_facts`, `tier1_always_on_pinned_in_slot`, `memory_recall_open_list`
- **Yujin allergy + EpiPen + gender + framework corrections** → `memory_recall_*` family
- **Persona hints (Korean/English, casual register)** → `tier1_always_on_register`, `tier1_always_on_persona_summary`
- **T14, T18, T38 decay candidates** → `topic_shift_*` family + `pin_discipline_no_overpin`
