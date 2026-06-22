"""v1.8: companions run in the background by DEFAULT.

The user-facing LLM-1 reply must never wait on companion work (LLM-2/LLM-3 +
decay). ``background=True`` is the default for the public BYO entry point; the
inline path stays available (``background=False``) for deterministic inspection.
"""

from __future__ import annotations

from sherlock import Sherlock


def _build(tmp_path, name, *, main, infer=None, background=None):
    kw = dict(
        main_chat=main,
        inference_chat=infer or (lambda m: "{}"),
        summary_chat=lambda m: "{}",
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        companions_mode="turbo",
    )
    if background is not None:
        kw["background"] = background
    return Sherlock.with_callable(**kw)


def test_with_callable_defaults_to_background_async(tmp_path):
    a = _build(tmp_path, "default", main=lambda m: "ok.")
    assert a._background_enabled is True  # async by default


def test_background_false_is_a_working_opt_out(tmp_path):
    calls = {"infer": 0}

    def infer(m):
        calls["infer"] += 1
        return "{}"

    a = _build(
        tmp_path,
        "inline",
        main=lambda m: "ok.\n<<sherlock-companions: infer>>",
        infer=infer,
        background=False,
    )
    a._turn_index = 10  # bypass cold-start so the infer companion is eligible
    a.chat("hi")
    # Inline → the companion already ran by the time chat() returned.
    assert calls["infer"] >= 1


def test_background_true_defers_companions_until_drain(tmp_path):
    calls = {"infer": 0}

    def infer(m):
        calls["infer"] += 1
        return "{}"

    a = _build(
        tmp_path,
        "async",
        main=lambda m: "ok.\n<<sherlock-companions: infer>>",
        infer=infer,
    )  # default background=True
    a._turn_index = 10
    a.chat("hi")  # returns immediately; companion runs in the bg worker
    a.drain()  # wait for background companions to land
    assert calls["infer"] >= 1


# --- the lock-release primitive: a slow bg deep-tier must NOT block the next turn ---


def test_slow_deep_tier_does_not_block_next_turn(tmp_path):
    """_lock_released_for_slow_work releases _mem_lock during the deep tier so a
    waiting next turn acquires it promptly instead of stalling for the whole
    (minutes-long, on a tiny model) freshness+notebook run."""
    import threading
    import time

    a = _build(tmp_path, "lockrel", main=lambda m: "ok.")

    # Simulate the bg worker holding the lock (as _bg_wrapper does).
    a._mem_lock.acquire()
    a._bg_lock_held = True
    a._bg_lock_thread = threading.get_ident()

    got = {"t": None}

    def other():
        t0 = time.time()
        if a._mem_lock.acquire(timeout=2.0):
            got["t"] = time.time() - t0
            a._mem_lock.release()

    th = threading.Thread(target=other)
    with a._lock_released_for_slow_work():
        th.start()
        th.join(timeout=2.0)  # other thread grabs the lock while we're "released"
    a._mem_lock.release()  # drop the lock the CM re-acquired on exit
    a._bg_lock_held = False
    a._bg_lock_thread = None

    assert got["t"] is not None, "next turn never acquired the lock during slow work"
    assert got["t"] < 1.0  # it got in fast, not after a long stall


def test_lock_blocks_without_release(tmp_path):
    """Control: while _mem_lock is held and NOT released, another thread cannot
    acquire it — proving the release in the test above is what frees the turn."""
    import threading

    a = _build(tmp_path, "lockheld", main=lambda m: "ok.")
    a._mem_lock.acquire()
    got = {"acquired": False}

    def other():
        if a._mem_lock.acquire(timeout=0.3):
            got["acquired"] = True
            a._mem_lock.release()

    th = threading.Thread(target=other)
    th.start()
    th.join(timeout=1.0)
    a._mem_lock.release()
    assert got["acquired"] is False
