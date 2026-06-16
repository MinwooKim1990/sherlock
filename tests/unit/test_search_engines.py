"""Unit tests for v0.3.0 multi-provider search engines + fetch.

Network is mocked everywhere via ``httpx.MockTransport`` so the suite is
hermetic. Live DuckDuckGo / Tavily / Brave / Valyu calls are exercised
only via integration tests gated behind env vars (not run by default).
"""

from __future__ import annotations

import json

import httpx
import pytest

from sherlock.tools.web_search import (
    BraveSearch,
    DuckDuckGoSearch,
    SearchEngine,
    StubSearch,
    TavilySearch,
    ValyuSearch,
    create_search,
    _default_fetch,
    _extract_text,
)


# Offline resolver: pretend any hostname resolves to a public IP, so the
# SSRF guard passes WITHOUT a real DNS lookup. Keeps the fetch tests hermetic
# (the guard itself is tested separately in test_security_v050).
def _public_resolver(host, port):
    return ["93.184.216.34"]  # example.com's documented public address


# ---------- factory --------------------------------------------------------


def test_create_search_duckduckgo_no_key():
    eng = create_search("duckduckgo")
    assert isinstance(eng, DuckDuckGoSearch)


def test_create_search_alias_ddg():
    assert isinstance(create_search("ddg"), DuckDuckGoSearch)


def test_create_search_tavily_with_direct_key():
    eng = create_search("tavily", api_key="dummy-key")
    assert isinstance(eng, TavilySearch)
    assert eng.api_key == "dummy-key"


def test_create_search_brave_via_env(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "bsa-xxx")
    eng = create_search("brave", api_key_env="BRAVE_API_KEY")
    assert isinstance(eng, BraveSearch)
    assert eng.api_key == "bsa-xxx"


def test_create_search_unknown_engine_raises():
    with pytest.raises(ValueError):
        create_search("not-a-real-provider")


def test_create_search_missing_key_raises():
    # Tavily *requires* a key — no api_key and no env should error.
    with pytest.raises(ValueError):
        create_search("tavily")


def test_create_search_stub_no_key_needed():
    assert isinstance(create_search("stub"), StubSearch)


# ---------- engine HTTP shapes (mocked) -----------------------------------


def _install_transport(engine, handler):
    """Swap the engine's outbound HTTP client by patching httpx.Client."""
    transport = httpx.MockTransport(handler)
    # The engines instantiate `httpx.Client(...)` inline; we patch the class
    # globally for the duration of the test.
    return transport


def test_brave_search_request_shape(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Sample",
                            "url": "https://example.com",
                            "description": "A snippet.",
                        }
                    ]
                }
            },
        )

    transport = httpx.MockTransport(handler)
    # Patch httpx.Client to use the mock transport for any kwargs.
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("sherlock.tools.web_search.httpx.Client", fake_client)
    eng = BraveSearch(api_key="bsa-xxx")
    out = eng.search("python", max_results=1)
    assert "x-subscription-token" in {k.lower() for k in captured["headers"]}
    assert "q=python" in captured["url"]
    assert out[0]["title"] == "Sample"
    assert out[0]["source"] == "brave"


def test_valyu_search_request_shape(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Valyu Result",
                        "url": "https://valyu.example",
                        "content": "Knowledge fragment.",
                    }
                ]
            },
        )

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("sherlock.tools.web_search.httpx.Client", fake_client)
    eng = ValyuSearch(api_key="valyu-xxx")
    out = eng.search("rust async", max_results=1)
    assert captured["body"]["query"] == "rust async"
    assert captured["body"]["max_num_results"] == 1
    assert "x-api-key" in {k.lower() for k in captured["headers"]}
    assert out[0]["source"] == "valyu"


def test_brave_search_error_returns_error_entry(monkeypatch):
    def handler(request):
        return httpx.Response(500, text="boom")

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("sherlock.tools.web_search.httpx.Client", fake_client)
    eng = BraveSearch(api_key="bsa-xxx")
    out = eng.search("anything")
    assert len(out) == 1
    assert "error" in out[0]


# ---------- fetch & extract -----------------------------------------------

_SAMPLE_HTML = """
<html><head><title>Hello</title></head>
<body>
  <script>var x = 1;</script>
  <h1>Headline</h1>
  <p>This is a paragraph with actual content worth keeping.</p>
  <p>Second paragraph for good measure.</p>
</body></html>
"""


def test_extract_text_strips_scripts():
    text = _extract_text(_SAMPLE_HTML)
    assert "var x = 1" not in text
    assert "Headline" in text or "paragraph" in text


def test_default_fetch_returns_text(monkeypatch):
    def handler(request):
        return httpx.Response(200, text=_SAMPLE_HTML)

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("sherlock.tools.web_search.httpx.Client", fake_client)
    out = _default_fetch("https://example.com", resolver=_public_resolver)
    assert out["status"] == 200
    assert "text" in out and "html" not in out
    assert "Headline" in out["text"] or "paragraph" in out["text"]


def test_default_fetch_raw_returns_html(monkeypatch):
    def handler(request):
        return httpx.Response(200, text=_SAMPLE_HTML)

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("sherlock.tools.web_search.httpx.Client", fake_client)
    out = _default_fetch("https://example.com", raw=True, resolver=_public_resolver)
    assert out["status"] == 200
    assert "html" in out and "text" not in out
    assert "<script>" in out["html"]


def test_default_fetch_error_does_not_raise(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("nope")

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("sherlock.tools.web_search.httpx.Client", fake_client)
    out = _default_fetch("https://example.com", resolver=_public_resolver)
    assert "error" in out


# ---------- factory + helpers --------------------------------------------


def test_subclass_inherits_default_fetch():
    class Toy(SearchEngine):
        def search(self, query, *, max_results=5):
            return []

    eng = Toy()
    assert hasattr(eng, "fetch")
    assert callable(eng.fetch)


def test_stub_engine_search_and_fetch():
    eng = StubSearch()
    res = eng.search("anything")
    assert res and "[stub]" in res[0]["title"]
    out = eng.fetch("https://example.com")
    assert out["text"].startswith("[stub]")
