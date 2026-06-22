"""Phase 1 stop (cooperative cancel) + Phase 1.5 agentic continuation (nudge).

- Stop: request_stop() halts the turn at the next boundary — it skips the
  post-response companions — and the flag clears at the start of the next turn.
- Nudge: a capable model that PROMISES to search/fetch ("I'll fetch…",
  "가져오겠습니다") but emits no tool tag is nudged ONCE to actually emit it (or
  answer), gated on a configured main search engine, within max_tool_rounds.
"""

from __future__ import annotations

from sherlock import Sherlock
from sherlock.agent import _is_unfulfilled_promise


def _agent(tmp_path, name, main, *, mode, search="stub", infer=None, summary=None):
    return Sherlock.with_callable(
        main_chat=main,
        inference_chat=infer or (lambda m: "{}"),
        summary_chat=summary or (lambda m: "{}"),
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        main_search_engine=search,
        inference_search_engine=None,
        companions_mode=mode,
        background=False,
    )


# ---------- stop ------------------------------------------------------------
def test_request_stop_skips_companions_then_clears(tmp_path):
    counts = {"infer": 0}
    holder: dict = {}

    def main(m):
        if holder.get("stop"):
            holder["a"].request_stop()  # user pressed Stop mid-generation
        return "ok."

    def infer(m):
        counts["infer"] += 1
        return "{}"

    a = _agent(tmp_path, "stop", main, mode="turbo", search=None, infer=infer)
    holder["a"] = a

    holder["stop"] = True
    a.chat("hello")
    assert counts["infer"] == 0  # stopped → companions skipped this turn

    holder["stop"] = False
    a.chat("again")
    assert counts["infer"] >= 1  # _stop_event cleared next turn → companions fire


def test_request_stop_sets_and_turn_clears_event(tmp_path):
    a = _agent(tmp_path, "ev", lambda m: "ok.", mode="off")
    a.request_stop()
    assert a._stop_event.is_set()
    a.chat("hi")  # a fresh turn clears it
    assert not a._stop_event.is_set()


# ---------- nudge detection -------------------------------------------------
def test_unfulfilled_promise_detection():
    for s in [
        "최신 결과 페이지를 직접 가져오겠습니다.",
        "확인해보겠습니다",
        "Let me fetch the page for you.",
        "I'll search for that now.",
        "그건 제가 찾아보겠습니다.",
    ]:
        assert _is_unfulfilled_promise(s), s
    for s in ["오늘은 화요일입니다.", "The answer is 42.", "멕시코가 2승으로 1위입니다.", ""]:
        assert not _is_unfulfilled_promise(s), s


# ---------- nudge end-to-end ------------------------------------------------
def _counting_main(seq):
    st = {"n": 0}

    def f(m):
        st["n"] += 1
        return seq(st["n"])

    return f, st


def test_nudge_fires_once_on_promise(tmp_path):
    main, st = _counting_main(
        lambda n: "최신 페이지를 직접 가져오겠습니다." if n == 1 else "최종 결과입니다: 멕시코 2승."
    )
    a = _agent(tmp_path, "nudge", main, mode="off")
    out = a.chat("최신 월드컵 결과 정리해줘")
    assert st["n"] == 2  # initial promise + one nudge round
    assert "최종" in out  # the nudge produced the real answer


def test_no_nudge_on_normal_answer(tmp_path):
    main, st = _counting_main(lambda n: "오늘은 화요일입니다.")
    a = _agent(tmp_path, "nonudge", main, mode="off")
    a.chat("오늘 무슨 요일이야?")
    assert st["n"] == 1  # a plain answer is never nudged


def test_no_nudge_without_search_engine(tmp_path):
    # The nudge is pointless with no search engine to fetch with → don't fire it.
    main, st = _counting_main(lambda n: "가져오겠습니다.")
    a = _agent(tmp_path, "noeng", main, mode="off", search=None)
    a.chat("최신 결과 알려줘")
    assert st["n"] == 1  # promise, but no engine → no nudge


def test_nudge_fires_at_most_once(tmp_path):
    # A model that keeps promising must NOT loop — nudged once, then the turn ends.
    main, st = _counting_main(lambda n: "가져오겠습니다.")  # always a promise
    a = _agent(tmp_path, "once", main, mode="off")
    a.chat("최신 결과")
    assert st["n"] == 2  # initial + exactly one nudge, then break (no infinite loop)
