"""Deep-research A/B on garbage free search (DuckDuckGo): Sherlock vs a strong
one-shot RAG baseline, same model + same junky engine.

Prompt (the user's real one): winter events across Sapporo (Dec 21–26),
Aomori (27), Akita (28), Morioka (29), Tokyo (Dec 30 – Jan 6). Today is
2026-06-16, so the honest answer is "2026–27 dates not yet officially
announced; here's the recurring pattern" — NOT confident last-season tables.

Sherlock  : _run_deep_research with DDG (multi-round plan→search→fetch→triangulate
            →honest-convergence→synthesis), capped rounds.
Baseline  : one-shot RAG — 5 city searches + fetch the top hit of each + dump
            excerpts + today's date → ONE synthesis call. (A *strong* fair
            baseline, not a 1-query strawman.)

Both get the SAME gemini-2.5-flash-lite and the SAME DuckDuckGo engine. The
only variable is Sherlock's curation/discipline.

Run:  .venv/bin/python evaluation/deep_research_ab.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re

from sherlock import Sherlock
from sherlock.providers.base import ChatResponse, TokenUsage
from sherlock.tools.web_search import BraveSearch, DuckDuckGoSearch

MODEL = "gemini-2.5-flash-lite"
# Engine selectable: SHERLOCK_AB_ENGINE=brave|duckduckgo (default duckduckgo).
ENGINE = (os.environ.get("SHERLOCK_AB_ENGINE") or "duckduckgo").lower()
ROUNDS = int(os.environ.get("SHERLOCK_AB_ROUNDS") or (8 if ENGINE == "brave" else 6))
OUT = Path(os.environ.get("SHERLOCK_AB_OUT") or f"evaluation/deep_research_ab_{ENGINE}.json")
_CLIENT = None


def _brave_key() -> str:
    raw = Path("brave_key.txt").read_text().strip()
    m = re.search(r"BRAVE_SEARCH_API_KEY\s*=\s*([A-Za-z0-9_\-]+)", raw)
    return m.group(1) if m else raw.strip().strip("[]")


def _make_engine():
    if ENGINE == "brave":
        return BraveSearch(api_key=_brave_key())
    return DuckDuckGoSearch()


PROMPT = os.environ.get("SHERLOCK_AB_PROMPT") or (
    "올해 12월 21일에서 26일까지 삿포로에서 하는 행사, 27일에 아오모리에서 하는 행사, "
    "28일 아키타에서 하는 행사, 29일 모리오카에서 하는 행사, 30일부터 1월 6일까지 "
    "도쿄에서 하는 행사 있는거 딥리서치 해서 알려줘."
)

# Baseline one-shot RAG queries — override with SHERLOCK_AB_QUERIES (||-separated).
_q_env = os.environ.get("SHERLOCK_AB_QUERIES")
BASELINE_QUERIES = (
    [q.strip() for q in _q_env.split("||") if q.strip()]
    if _q_env
    else [
        "삿포로 12월 겨울 축제 이벤트 2026 일루미네이션",
        "아오모리 12월 겨울 행사 2026",
        "아키타 12월 겨울 행사 이벤트 2026",
        "모리오카 이와테 12월 겨울 행사 2026",
        "도쿄 12월 연말 1월 신년 초일출 행사 2026 2027",
    ]
)


def _raw_chat(prompt: str):
    global _CLIENT
    if _CLIENT is None:
        from unified_cli import create

        _CLIENT = create("gemini", model=MODEL, timeout=90.0)
    return _CLIENT.chat(prompt)


def _flatten(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m.get('role','user').upper()}]\n{m.get('content','')}" for m in messages)


def _usage(resp) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return (
        int(getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0),
        int(getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0),
    )


def gemini_callable(counter: dict):
    def fn(messages: list[dict]):
        resp = _raw_chat(_flatten(messages))
        text = getattr(resp, "text", "") or ""
        pin, out = _usage(resp)
        counter["in"] += pin
        counter["out"] += out
        counter["calls"] += 1
        return ChatResponse(
            text=text, model=MODEL, usage=TokenUsage(prompt_tokens=pin, completion_tokens=out)
        )

    return fn


def run_sherlock() -> dict:
    import tempfile

    print(f"=== SHERLOCK deep research ({ENGINE}) ===", flush=True)
    counter = {"in": 0, "out": 0, "calls": 0}
    engine_kwargs = {}
    if ENGINE == "brave":
        engine_kwargs = {
            "main_search_engine": "brave",
            "inference_search_engine": "brave",
            "search_api_key": _brave_key(),
        }
    else:
        engine_kwargs = {
            "main_search_engine": "duckduckgo",
            "inference_search_engine": "duckduckgo",
        }
    agent = Sherlock.with_callable(
        main_chat=gemini_callable(counter),
        inference_chat=gemini_callable(counter),
        summary_chat=gemini_callable(counter),
        system_prompt="You are a meticulous research assistant. Reply in Korean.",
        storage_dir=tempfile.mkdtemp(prefix="shdr_"),
        embedding="local",
        background=False,
        context_window=1_000_000,
        **engine_kwargs,
    )
    agent.config.search.deep_research_max_rounds = ROUNDS
    events: list[tuple] = []
    agent.set_event_sink(lambda ev: events.append((ev.get("type"), ev.get("data", {}))))
    conv = agent._ensure_conversation().id
    t0 = time.time()
    doc = agent._run_deep_research(conv, PROMPT, 1, "drJP", user_text=PROMPT)
    secs = int(time.time() - t0)
    docs_ev = next((d for (t, d) in events if t == "deep_research.documents"), {})
    rounds = [d for (t, d) in events if t == "deep_research.round"]
    strat = next((d for (t, d) in events if t == "deep_research.strategy"), {})
    return {
        "answer": doc,
        "seconds": secs,
        "tokens": counter,
        "dr_tok": getattr(agent, "_dr_tok", {}),
        "stop_reason": docs_ev.get("stop_reason"),
        "rounds": len(rounds),
        "facts_total": (rounds[-1].get("facts_total") if rounds else None),
        "unverified_citations": docs_ev.get("unverified_citations"),
        "strategy_sub_topics": strat.get("sub_topics"),
        "coverage_steers": [
            {
                "round": d.get("round"),
                "covered": d.get("covered"),
                "total": d.get("total"),
                "uncovered": d.get("uncovered"),
            }
            for (t, d) in events
            if t == "deep_research.coverage_steer"
        ],
        "rounds_detail": [
            {
                "round": d.get("round"),
                "queries": d.get("queries"),
                "new_fragments": d.get("new_fragments"),
                "fetched": d.get("fetched"),
                "facts_total": d.get("facts_total"),
                "raw_fragments_stored": d.get("raw_fragments_stored"),
                "sufficient": d.get("sufficient"),
            }
            for d in rounds
        ],
    }


def run_baseline() -> dict:
    print(f"=== BASELINE one-shot RAG ({ENGINE}) ===", flush=True)
    eng = _make_engine()
    counter = {"in": 0, "out": 0, "calls": 0}
    blocks = []
    hit_log = []
    for q in BASELINE_QUERIES:
        hits = eng.search(q, max_results=4) or []
        hit_log.append({"q": q, "hits": [h.get("title") or h.get("error") for h in hits]})
        lines = [f"## 검색: {q}"]
        for h in hits[:4]:
            if "error" in h:
                lines.append(f"- (검색실패: {h['error'][:60]})")
                continue
            title = str(h.get("title", ""))[:90]
            url = str(h.get("url", ""))
            snip = str(h.get("content", "") or h.get("snippet", ""))[:200]
            lines.append(f"- {title} | {url}\n  {snip}")
            # fetch the top hit's page text to give the baseline a fair shot
            if h is hits[0] and url.startswith("http"):
                try:
                    page = eng.fetch(url, timeout=12.0)
                    txt = str((page or {}).get("text", ""))[:800]
                    if txt:
                        lines.append(f"  [본문발췌] {txt}")
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"  [본문 fetch 실패: {type(exc).__name__}]")
        blocks.append("\n".join(lines))

    sys_prompt = (
        "You are a meticulous research assistant. TODAY is 2026-06-16. "
        "The user asks about events in December 2026 – January 2027. "
        "Use the web snippets below. If a date is from a PAST season, say so; "
        "if 2026–27 info is not found, say it is not yet announced rather than "
        "guessing. Reply in Korean, organized by city."
    )
    user_prompt = PROMPT + "\n\n=== 웹 검색 결과 ===\n" + "\n\n".join(blocks)
    t0 = time.time()
    resp = _raw_chat(f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{user_prompt}")
    secs = int(time.time() - t0)
    pin, out = _usage(resp)
    counter["in"] += pin
    counter["out"] += out
    counter["calls"] += 1
    return {
        "answer": getattr(resp, "text", "") or "",
        "seconds": secs,
        "tokens": counter,
        "search_hits": hit_log,
    }


def main():
    result = {"model": MODEL, "prompt": PROMPT, "today": "2026-06-16"}
    try:
        result["sherlock"] = run_sherlock()
    except Exception as exc:  # noqa: BLE001
        import traceback

        result["sherlock"] = {"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    try:
        result["baseline"] = run_baseline()
    except Exception as exc:  # noqa: BLE001
        import traceback

        result["baseline"] = {"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    s = result.get("sherlock", {})
    b = result.get("baseline", {})
    print(
        f"SHERLOCK: {s.get('rounds')} rounds, stop={s.get('stop_reason')}, "
        f"facts={s.get('facts_total')}, calls={s.get('tokens',{}).get('calls')}, "
        f"tok in/out={s.get('tokens',{}).get('in')}/{s.get('tokens',{}).get('out')}, {s.get('seconds')}s"
    )
    print(
        f"BASELINE: calls={b.get('tokens',{}).get('calls')}, "
        f"tok in/out={b.get('tokens',{}).get('in')}/{b.get('tokens',{}).get('out')}, {b.get('seconds')}s"
    )
    print("=" * 70)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
