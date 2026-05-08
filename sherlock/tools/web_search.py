"""Web search tool. Tavily-backed primary, with a stub provider for tests."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sherlock.config import SearchConfig


class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: str, *, max_results: int = 5) -> list[dict]: ...


@dataclass
class TavilySearch(SearchEngine):
    api_key: str
    _client: object = None

    def __post_init__(self) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        try:
            res = self._client.search(query=query, max_results=max_results)
        except Exception as exc:
            return [{"error": f"{type(exc).__name__}: {exc}", "query": query}]
        out: list[dict] = []
        for r in res.get("results", []):
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
            )
        return out


@dataclass
class StubSearch(SearchEngine):
    """Returns a deterministic placeholder. Useful for hermetic tests."""

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        return [
            {
                "title": f"[stub] {query}",
                "url": "https://example.com/",
                "content": (
                    "Web search is not configured (no Tavily API key). "
                    "Sherlock proceeded without fresh web context."
                ),
            }
        ]


def build_search_engine(cfg: "SearchConfig | None") -> SearchEngine | None:
    if cfg is None or not cfg.always_on:
        return None
    if cfg.provider.lower() == "stub":
        return StubSearch()
    if cfg.provider.lower() == "tavily":
        api_key = os.environ.get(cfg.api_key_env or "TAVILY_API_KEY")
        if not api_key:
            return StubSearch()
        return TavilySearch(api_key=api_key)
    return StubSearch()
