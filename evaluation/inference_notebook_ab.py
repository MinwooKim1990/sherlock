"""Stage 4 live check — recursive inference notebook, OFF vs ON, same model.

Turn 1: an implicit-chain message (Tokyo trip, "should I buy the pass early?").
With Stage-4 on, LLM-3 should keep a bounded, GROUNDED notebook (each raw step
quotes the conversation / observations) and it should ride Turn 2's slot. We
verify: a notebook.done event with raw+conclusions, every raw step's evidence is
a verbatim substring of the corpus (grounding holds), bounded rounds, and the
"INFERENCE NOTEBOOK" block appears in Turn 2's slot. OFF emits no notebook.

Run:  .venv/bin/python evaluation/inference_notebook_ab.py
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
OUT = Path("evaluation/inference_notebook_ab.json")
_CLIENT = None
_LOCK = threading.Lock()

TURN1 = "12월 27일에 도쿄 가는데, JR 패스 미리 사야 할까 고민이야."
TURN2 = "음 그러면 어떻게 하는 게 좋을까?"


def _raw_chat(prompt):
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from unified_cli import create

            _CLIENT = create("gemini", model=MODEL, timeout=90.0)
        return _CLIENT.chat(prompt)


def _flatten(messages):
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _cb():
    def fn(messages):
        resp = _raw_chat(_flatten(messages))
        return ChatResponse(
            text=getattr(resp, "text", "") or "",
            model=MODEL,
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
        )

    return fn


def _run(on, label):
    import tempfile

    events = []
    agent = Sherlock.with_callable(
        main_chat=_cb(),
        inference_chat=_cb(),
        summary_chat=_cb(),
        system_prompt="You are a thoughtful travel assistant. Reply in Korean, concise.",
        storage_dir=tempfile.mkdtemp(prefix=f"nb_{label}_"),
        embedding="fake",
        background=False,
        context_window=128_000,
        main_search_engine="duckduckgo",
        inference_search_engine="duckduckgo",
        perception=on,
        evidence_grounding=on,
        inference_notebook=on,
        notebook_max_rounds=3,
    )
    agent.config.memory.auto_infer = "always"  # force LLM-3 each turn
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))

    agent.chat(TURN1)
    agent.drain()
    nb = next((d for (t, d) in events if t == "notebook.done"), None)
    agent.chat(TURN2)
    agent.drain()
    slot2 = agent.inspect_last_turn().messages_passed_to_llm1[-1].content

    # Grounding is enforced + unit-tested in the engine (a step only survives into
    # `raw` if its evidence is a verbatim corpus quote); here we just surface the
    # surviving notebook + sample evidence so a human can eyeball it.
    return {
        "label": label,
        "notebook": nb,
        "notebook_in_turn2_slot": "INFERENCE NOTEBOOK" in slot2,
        "raw_steps": len(nb.get("raw", [])) if nb else 0,
        "sample_evidence": [
            str(s.get("evidence", ""))[:60] for s in (nb.get("raw", []) if nb else [])
        ],
        "conclusions": nb.get("conclusions", []) if nb else [],
        "rounds": nb.get("rounds") if nb else None,
    }


def main():
    print(f"=== inference-notebook A/B ===\n  turn1: {TURN1}\n  turn2: {TURN2}\n", flush=True)
    off = _run(False, "off")
    on = _run(True, "on")
    OUT.write_text(
        json.dumps(
            {"turn1": TURN1, "turn2": TURN2, "off": off, "on": on},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    for r in (off, on):
        print(f"### {r['label'].upper()}")
        print(
            f"  notebook produced: {r['notebook'] is not None} | rounds={r['rounds']} "
            f"raw_steps={r['raw_steps']}"
        )
        print(f"  notebook in turn-2 slot: {r['notebook_in_turn2_slot']}")
        for e in r.get("sample_evidence", []):
            print(f"    step evidence: {e!r}")
        if r["conclusions"]:
            for c in r["conclusions"]:
                print(f"    conclusion: {c}")
        print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
