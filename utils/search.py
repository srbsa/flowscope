"""
utils/search.py
Shared Tavily web search utility used by research, alignment, and synthesis agents.
Centralises the Tavily client lifecycle and result formatting so individual agents
don't each maintain their own client singleton.
"""

import logging
import os

from tavily import TavilyClient

logger = logging.getLogger(__name__)

_tavily: TavilyClient | None = None


def get_tavily() -> TavilyClient:
    """Return a TavilyClient singleton, raising clearly if the API key is absent."""
    global _tavily
    if _tavily is None:
        key = os.environ.get("TAVILY_API_KEY", "")
        if not key:
            raise ValueError(
                "TAVILY_API_KEY is not set. Set it in your .env file. "
                "Get a free key at https://tavily.com"
            )
        _tavily = TavilyClient(api_key=key)
    return _tavily


def run_search(query: str, max_results: int = 5) -> tuple[str, list[str]]:
    """
    Execute a Tavily search and return (formatted_results_text, source_urls).

    Args:
        query:       Search query string (5–10 words, specific and targeted).
        max_results: Number of search results to request from Tavily.

    Returns:
        (formatted_text, source_url_list)
    """
    logger.info("Searching: %s", query)
    results = get_tavily().search(
        query=query, max_results=max_results, search_depth="advanced"
    )
    formatted: list[str] = []
    sources: list[str] = []
    for r in results.get("results", []):
        formatted.append(
            f"**{r.get('title', 'Untitled')}**\n"
            f"URL: {r.get('url', '')}\n"
            f"{r.get('content', '')[:400]}\n"
        )
        if r.get("url"):
            sources.append(r["url"])
    return "\n---\n".join(formatted) or "No results found.", sources
