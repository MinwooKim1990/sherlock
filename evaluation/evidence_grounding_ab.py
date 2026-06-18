"""Stage 2 live A/B — evidence-grounded LLM-3, OFF vs ON, same model.

Validates the three Stage-2 behaviors end-to-end with a real small model
(gemini-flash-lite) + a real free search engine (DuckDuckGo):

  1. premise_conflict — on "show me spaceX stock price" the ON run should emit a
     premise_conflict topic (the model believes SpaceX is private → flags the
     discrepancy for a web check) and route it into freshness_required, firing
     the inference-search loop. The OFF run must emit none.
  2. span-grounding — ON hypotheses should carry verbatim quotes; any uncited
     hypothesis is capped to ≤0.35 ("grounding": "uncited — capped").
  3. observations — the perception OBSERVED block reaches LLM-3's input.

We force infer() every turn (auto_infer="always", background off) and capture
the infer.done / infer.search.round events via the event sink.

Run:  .venv/bin/python evaluation/evidence_grounding_ab.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock  # noqa: E402
from sherlock.providers.base import ChatResponse, TokenUsage  # noqa: E402

MODEL = "gemini-2.5-flash-lite"
OUT = Path("evaluation/evidence_grounding_ab.json")
_CLIENT = None
_LOCK = threading.Lock()


def _raw_chat(prompt: str):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from unified_cli import create

            _CLIENT = create("gemini", model=MODEL, timeout=90.0)
        return _CLIENT.chat(prompt)


def _flatten(messages):
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _cb(counter):
    def fn(messages):
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        u = getattr(resp, "usage", None)
        pin = int(getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0) if u else 0
        out = (
            int(getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0)
            if u
            else 0
        )
        counter["in"] += pin
        counter["calls"] += 1
        return ChatResponse(
            text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out)
        )

    return fn


def _run(stage2_on: bool, label: str, prompt: str) -> dict:
    import tempfile

    counter = {"in": 0, "calls": 0}
    events: list[tuple] = []
    agent = Sherlock.with_callable(
        main_chat=_cb(counter),
        inference_chat=_cb(counter),
        summary_chat=_cb(counter),
        system_prompt="You are a precise assistant. Reply concisely.",
        storage_dir=tempfile.mkdtemp(prefix=f"eg_{label}_"),
        embedding="fake",
        background=False,
        context_window=128_000,
        main_search_engine="duckduckgo",
        inference_search_engine="duckduckgo",
        perception=stage2_on,
        evidence_grounding=stage2_on,
        premise_conflict=stage2_on,
    )
    agent.config.memory.auto_infer = "always"  # force LLM-3 every turn
    agent.config.inference.max_search_rounds = 2  # keep the live loop short
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    reply = agent.chat(prompt)
    agent.drain()

    infer = next((d for (t, d) in events if t == "infer.done"), {})
    hyps = infer.get("hypotheses", []) or []
    search_rounds = [d for (t, d) in events if t == "infer.search.round"]
    capped = [h for h in hyps if isinstance(h, dict) and h.get("grounding") == "uncited — capped"]
    quoted = [
        h
        for h in hyps
        if isinstance(h, dict)
        and any('"' in str(e) or "'" in str(e) for e in (h.get("evidence") or []))
    ]
    return {
        "label": label,
        "reply": reply[:400],
        "premise_conflict": infer.get("premise_conflict", []),
        "freshness_required": infer.get("freshness_required", []),
        "n_hypotheses": len(hyps),
        "n_quoted_evidence": len(quoted),
        "n_capped": len(capped),
        "capped_probs": [h.get("probability") for h in capped],
        "search_rounds_fired": len(search_rounds),
        "calls": counter["calls"],
        "in_tokens": counter["in"],
    }


def main():
    prompt = "show me spaceX stock price"
    print(f"=== evidence-grounding A/B — prompt: {prompt!r} ===", flush=True)
    off = _run(False, "off", prompt)
    on = _run(True, "on", prompt)
    out = {"model": MODEL, "prompt": prompt, "off": off, "on": on}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    print("\n" + "=" * 72)
    for r in (off, on):
        print(f"\n### {r['label'].upper()}")
        print(f"  premise_conflict: {r['premise_conflict']}")
        print(f"  freshness_required: {r['freshness_required']}")
        print(
            f"  hypotheses={r['n_hypotheses']} quoted-evidence={r['n_quoted_evidence']} "
            f"capped={r['n_capped']} {r['capped_probs']}"
        )
        print(f"  inference-search rounds fired: {r['search_rounds_fired']}")
        print(f"  reply: {r['reply'][:200]}")
    print("\n" + "=" * 72)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
