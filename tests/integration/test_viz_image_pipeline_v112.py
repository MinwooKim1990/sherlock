"""v1.12 Stage V3 — text→image pipeline e2e on the agent: ``image:`` markers
through chat() → the deterministic wrapper → viz.rendered (or graceful
degradation to the normal LLM-4 path when the modality is unconfigured)."""

from __future__ import annotations

import base64
import threading
from pathlib import Path

from sherlock import Sherlock

B64 = base64.b64encode(b"fakepng").decode()
URL = "https://img.example.com/gen/abc.png"

IMG_REPLY = "Here you go:\n<<sherlock-viz: image: a lighthouse in a storm, watercolor>>\ndone"

VALID = (
    "<!DOCTYPE html><html><head>\n"
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">\n"
    "</head><body><div><span>ok</span></div>\n"
    "<script>window.onerror=(e)=>parent.postMessage({sherlockViz:'error',message:String(e)},'*');"
    "parent.postMessage({sherlockViz:'ready'}, '*');</script></body></html>"
)


class _ScriptViz:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, messages):
        prompt = "\n".join((m["content"] if isinstance(m, dict) else m.content) for m in messages)
        with self._lock:
            self.prompts.append(prompt)
            return self._responses.pop(0) if self._responses else "NO MORE"


class _RecImageGen:
    def __init__(self, ret):
        self.ret = ret
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.ret, Exception):
            raise self.ret
        return self.ret


def _agent(tmp_path, name, *, main, viz_chat=None, image_gen=None):
    return Sherlock.with_callable(
        main_chat=main,
        summary_chat=lambda m: "{}",
        inference_chat=lambda m: "{}",
        viz_chat=viz_chat,
        viz_image_gen=image_gen,
        system_prompt="You are terse.",
        storage_dir=tmp_path / name,
        context_window=128_000,
        embedding="fake",
        background=False,
        companions_mode="off",
        visualization=True,
    )


def _events_of(events, type_):
    return [e for e in events if e["type"] == type_]


def test_b64_image_renders_data_uri_artifact(tmp_path):
    gen = _RecImageGen({"b64": B64})
    viz = _ScriptViz()  # must never be called
    events: list[dict] = []
    agent = _agent(tmp_path, "b64", main=lambda m: IMG_REPLY, viz_chat=viz, image_gen=gen)
    agent.set_event_sink(events.append)

    agent.chat("draw it")
    assert agent.wait_for_viz(timeout=5) is True

    rendered = _events_of(events, "viz.rendered")
    assert len(rendered) == 1
    html = rendered[0]["data"]["html"]
    assert "data:image/png;base64," + B64 in html
    assert "a lighthouse in a storm" in html  # title = description
    assert viz.prompts == []  # no LLM round for an image job
    assert gen.prompts and "lighthouse" in gen.prompts[0]
    # persisted, no .allow sidecar for a data: URI
    path = Path(rendered[0]["data"]["path"])
    assert path.exists()
    assert not path.with_suffix(".allow").exists()


def test_url_image_pins_allowlist_and_sidecar(tmp_path):
    gen = _RecImageGen({"url": URL})
    events: list[dict] = []
    agent = _agent(tmp_path, "url", main=lambda m: IMG_REPLY, image_gen=gen)
    agent.set_event_sink(events.append)

    agent.chat("draw it")
    assert agent.wait_for_viz(timeout=5) is True

    rendered = _events_of(events, "viz.rendered")
    assert len(rendered) == 1
    html = rendered[0]["data"]["html"]
    assert f'src="{URL}"' in html
    assert f"img-src data: {URL}" in html  # CSP pin (V1 mechanism)
    sidecar = Path(rendered[0]["data"]["path"]).with_suffix(".allow")
    assert sidecar.exists()
    assert sidecar.read_text(encoding="utf-8").strip() == URL
    # audit: the STASH copy of the job still has image_urls=() (the pool works on
    # dict(job)) — allowlist recovery must fall through to the sidecar, not
    # shadow it with the stale empty entry.
    assert agent._viz_allowlist_for(rendered[0]["data"]["viz_id"]) == (URL,)


def test_unsafe_url_fails_closed(tmp_path):
    gen = _RecImageGen({"url": 'https://evil.example.com/a".png'})  # forbidden quote
    events: list[dict] = []
    agent = _agent(tmp_path, "bad", main=lambda m: IMG_REPLY, image_gen=gen)
    agent.set_event_sink(events.append)

    agent.chat("draw it")
    assert agent.wait_for_viz(timeout=5) is True
    failed = _events_of(events, "viz.failed")
    assert len(failed) == 1
    assert "sanitisation" in failed[0]["data"]["reason"]
    assert _events_of(events, "viz.rendered") == []


def test_provider_error_falls_back_to_llm_render(tmp_path):
    # omni fix: a model that can't produce an image (text-only viz model, api
    # error) must NOT kill the slot — the job falls back to the HTML/SVG path.
    gen = _RecImageGen(RuntimeError("image api down"))
    viz = _ScriptViz(VALID)
    events: list[dict] = []
    agent = _agent(tmp_path, "err", main=lambda m: IMG_REPLY, viz_chat=viz, image_gen=gen)
    agent.set_event_sink(events.append)

    agent.chat("draw it")
    assert agent.wait_for_viz(timeout=5) is True
    assert _events_of(events, "viz.failed") == []
    assert len(_events_of(events, "viz.rendered")) == 1
    assert gen.prompts  # the image path WAS attempted first
    assert len(viz.prompts) == 1  # then the LLM drew it


def test_unconfigured_image_job_degrades_to_llm_render(tmp_path):
    # kind=image but NO image gen configured → normal LLM-4 render of the bare
    # description (a stray prefix never breaks anything).
    viz = _ScriptViz(VALID)
    events: list[dict] = []
    agent = _agent(tmp_path, "deg", main=lambda m: IMG_REPLY, viz_chat=viz)
    agent.set_event_sink(events.append)

    agent.chat("draw it")
    assert agent.wait_for_viz(timeout=5) is True
    assert len(_events_of(events, "viz.rendered")) == 1
    assert len(viz.prompts) == 1
    assert "a lighthouse in a storm" in viz.prompts[0]
    assert "image:" not in viz.prompts[0].split("SURROUNDING MATERIAL")[0]


def test_tier1_guidance_gated_on_image_modality(tmp_path):
    class _CapMain:
        def __init__(self):
            self.system_prompts: list[str] = []

        def __call__(self, messages):
            for m in messages:
                role = m["role"] if isinstance(m, dict) else m.role
                if role == "system":
                    self.system_prompts.append(m["content"] if isinstance(m, dict) else m.content)
                    break
            return "ok"

    with_img = _CapMain()
    agent = _agent(tmp_path, "g1", main=with_img, image_gen=lambda p: {"b64": B64})
    agent.chat("hello")
    assert any("image:" in sp for sp in with_img.system_prompts)

    without = _CapMain()
    agent2 = _agent(tmp_path, "g0", main=without)
    agent2.chat("hello")
    assert any("sherlock-viz" in sp for sp in without.system_prompts)  # viz on
    assert not any("image:" in sp for sp in without.system_prompts)  # image off


def test_inflight_render_persists_under_source_conversation(tmp_path):
    # audit: a render finishing AFTER new_session/switch_session must file its
    # artifact under the conversation that produced it, not the new one.
    import threading

    gate = threading.Event()

    def slow_viz(messages):
        gate.wait(timeout=5)
        return VALID

    events: list[dict] = []
    agent = _agent(
        tmp_path, "xconv", main=lambda m: "x:\n<<sherlock-viz: a chart | A 1>>", viz_chat=slow_viz
    )
    agent.set_event_sink(events.append)

    agent.chat("draw")  # submits t1-1, render blocked on the gate
    source_conv = agent.conversation_id
    agent.new_session()  # user moves on mid-render
    gate.set()
    assert agent.wait_for_viz(timeout=5) is True

    rendered = [e for e in events if e["type"] == "viz.rendered"]
    assert len(rendered) == 1
    assert rendered[0]["data"]["conv"] == source_conv  # pinned at job creation
    assert Path(rendered[0]["data"]["path"]).parent.name == source_conv
