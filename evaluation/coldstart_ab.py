"""v1.6 deterministic A/B — Quiescence Gate (cold_start) vs turbo.

A short, mostly-quiet conversation with ONE genuinely ambiguous turn. cold_start
should stay single-model on the quiet turns (no LLM-3) and escalate on the
ambiguous one. We count REAL prompt tokens (tiktoken) across EVERY companion
call — the cost driver — with scripted replies, so the result is deterministic
and faithful to token cost without needing a live model (gemini auth tier was
revoked mid-session). turbo fires LLM-3 every turn; cold_start only on signal.

Run:  .venv/bin/python evaluation/coldstart_ab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sherlock import Sherlock  # noqa: E402
from sherlock.budget import count_tokens  # noqa: E402

OUT = Path("evaluation/coldstart_ab.json")

# One coherent topic (a quiet weekend) so real embeddings see no topic shift on
# the calm turns; turn 4 is a genuine anaphora+hedge ambiguity that should escalate.
TURNS = [
    "이번 주말에 그냥 집에서 푹 쉬려고 해요.",
    "넷플릭스 보면서 좀 늘어지고 싶네요.",
    "저녁엔 맛있는 거 시켜 먹을까 해요.",
    "음... 근데 이거 진짜 해야 할지 말아야 할지 모르겠어.",  # anaphora + hedge → escalate
    "아무튼 주말엔 푹 쉴래요.",
]

_INFER_JSON = json.dumps(
    {
        "hypotheses": [
            {
                "intent": "needs reassurance",
                "probability": 0.5,
                "evidence": ["e"],
                "search_keywords": [],
                "reasoning_type": "pragmatic",
            }
        ],
        "tools_recommended": [],
        "freshness_required": [],
        "confidence_overall": 0.5,
        "evolution_signals": {},
    }
)


def _flatten(messages):
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _cb(counter, role, reply):
    def fn(messages):
        # count REAL prompt tokens for this call (the cost we're measuring).
        counter["in"] += count_tokens(_flatten(messages))
        counter["calls"] += 1
        counter[role] = counter.get(role, 0) + 1
        return reply

    return fn


def _run(mode):
    import tempfile

    counter = {"in": 0, "calls": 0}
    events = []
    agent = Sherlock.with_callable(
        main_chat=_cb(counter, "main", "그렇군요, 잘 들었어요."),
        inference_chat=_cb(counter, "infer", _INFER_JSON),
        summary_chat=_cb(counter, "summary", "{}"),
        system_prompt="You are a warm, brief assistant. Reply in Korean.",
        storage_dir=tempfile.mkdtemp(prefix=f"cs_{mode}_"),
        embedding="local",
        background=False,
        context_window=128_000,
        main_search_engine=None,
        inference_search_engine=None,
        companions_mode=mode,
    )
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    per_turn = []
    for t in TURNS:
        before = dict(counter)
        agent.chat(t)
        agent.drain()
        infer_fired = counter.get("infer", 0) - before.get("infer", 0)
        gate = [d for (ty, d) in events if ty == "companion.gate"]
        last_gate = gate[-1] if gate else {}
        per_turn.append(
            {
                "turn": t[:18],
                "infer_calls": infer_fired,
                "fire3": last_gate.get("fire3"),
                "deep": last_gate.get("deep"),
                "in_tokens_delta": counter["in"] - before["in"],
            }
        )
    return {"mode": mode, "totals": counter, "per_turn": per_turn}


def main():
    print("=== cold_start vs turbo A/B ===", flush=True)
    turbo = _run("turbo")
    cold = _run("cold_start")
    OUT.write_text(json.dumps({"turbo": turbo, "cold_start": cold}, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    for r in (turbo, cold):
        t = r["totals"]
        print(
            f"\n### {r['mode'].upper()}  total: prompt-tokens={t['in']} "
            f"calls={t['calls']} (infer={t.get('infer',0)} summary={t.get('summary',0)})"
        )
        for pt in r["per_turn"]:
            print(
                f"   {pt['turn']:<20} infer={pt['infer_calls']} fire3={pt['fire3']} "
                f"deep={pt['deep']} (+{pt['in_tokens_delta']} in-tok)"
            )
    ti, ci = turbo["totals"]["in"], cold["totals"]["in"]
    print("\n" + "=" * 70)
    print(
        f"cold_start input tokens = {ci} vs turbo {ti}  "
        f"→ {(1 - ci / max(1, ti)) * 100:.0f}% fewer; "
        f"infer calls {cold['totals'].get('infer',0)} vs {turbo['totals'].get('infer',0)}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
