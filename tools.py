"""SerpAPI web search, exposed to the LLM as the ``web_search`` tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from config import Settings, load_settings

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for current, time-sensitive information: news, "
            "current events, prices, weather, sports scores, and facts that may "
            "have changed recently. Always pass a 'query' string. Do not use it "
            "for stable knowledge, coding, math, or general advice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query, preferably in English.",
                }
            },
        },
    },
}
WEB_SEARCH_TOOLS: list[dict[str, Any]] = [WEB_SEARCH_TOOL]


class SearchError(Exception):
    """Raised when search cannot be performed."""


@dataclass(frozen=True)
class SearchResult:
    """A compact SerpAPI organic result."""

    title: str
    snippet: str
    link: str

    def as_context(self, rank: int) -> str:
        """Render one result for the model's context."""
        snippet = " ".join(self.snippet.split())[:300]
        lines = [f"{rank}. {self.title}"]
        if snippet:
            lines.append(f"   {snippet}")
        lines.append(f"   {self.link}")
        return "\n".join(lines)


def web_search(
    query: str,
    settings: Settings | None = None,
    *,
    session: requests.Session | None = None,
) -> list[SearchResult]:
    """Return the top organic results for ``query`` (empty list if none found)."""
    query = query.strip()
    if not query:
        raise SearchError("The search query was empty.")

    settings = settings or load_settings(require_groq=False)
    if not settings.web_search_enabled:
        raise SearchError(
            "SERPAPI_API_KEY is missing. Add it to .env (see .env.example) to "
            "enable live web search."
        )

    params = {
        "engine": "google",
        "q": query,
        "api_key": settings.serpapi_api_key,
        "num": settings.search_results,
    }
    requester = session or requests
    try:
        response = requester.get(
            SERPAPI_ENDPOINT, params=params, timeout=settings.search_timeout_seconds
        )
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        raise SearchError(f"SerpAPI request failed with HTTP {status}.") from None
    except requests.RequestException as exc:
        raise SearchError(
            f"Web search request failed ({type(exc).__name__})."
        ) from None
    except ValueError as exc:
        raise SearchError("Web search returned invalid JSON.") from exc

    if payload.get("error"):
        raise SearchError(f"SerpAPI error: {payload['error']}")

    raw_results = payload.get("organic_results") or []
    results: list[SearchResult] = []
    for item in raw_results[: settings.search_results]:
        link = str(item.get("link") or "").strip()
        if not link:
            continue
        results.append(
            SearchResult(
                title=str(item.get("title") or link).strip(),
                snippet=str(item.get("snippet") or "").strip(),
                link=link,
            )
        )
    return results


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results compactly for the tool message."""
    if not results:
        return "No results found for this query."
    return "\n".join(
        result.as_context(rank) for rank, result in enumerate(results, start=1)
    )


def format_sources(results: list[SearchResult], *, limit: int = 5) -> str:
    """Build a Markdown sources section from unique search results."""
    lines = ["", "", "**Sources:**"]
    seen: set[str] = set()
    for result in results:
        if result.link in seen:
            continue
        seen.add(result.link)
        lines.append(f"- [{result.title}]({result.link})")
        if len(seen) >= limit:
            break
    return "\n".join(lines) + "\n" if len(seen) else ""
