from __future__ import annotations

from loguru import logger

from config.settings import RAG_CHUNK_MAX


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo text search. Returns list of {title, url, snippet}."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": (r.get("body") or "")[:400],
            }
            for r in results
        ]
    except Exception as exc:
        logger.error(f"search_web error: {exc}")
        return []


def fetch_page(url: str) -> str:
    """Fetch and extract clean text from a URL using trafilatura.
    Returns extracted markdown-ish text, truncated to RAG_CHUNK_MAX * 3 chars."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"[fetch_page] Could not download: {url}"
        text = trafilatura.extract(
            downloaded,
            include_links=False,
            include_tables=True,
            no_fallback=False,
        )
        if not text:
            return f"[fetch_page] No extractable content at: {url}"
        return text[: RAG_CHUNK_MAX * 3]
    except Exception as exc:
        logger.error(f"fetch_page error for {url}: {exc}")
        return f"[fetch_page] Error: {exc}"
