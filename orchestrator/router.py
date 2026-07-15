import json
import re
from functools import lru_cache
from time import perf_counter

from loguru import logger

from serving.llm_client import LLMClient
from orchestrator.prompts import ROUTER_PROMPT
from config.settings import MODEL_ROUTER, ROUTER_CACHE_SIZE

_llm = LLMClient()
_VALID_TAGS = {"use_rag", "use_browser", "use_memory", "direct"}

# ── Keyword heuristic sets ────────────────────────────────────────────────────

_BROWSER_KW = re.compile(
    r"\b("
    r"latest|current|today|tonight|yesterday|this week|this month|this year|"
    r"news|weather|forecast|price|stock|crypto|bitcoin|market|"
    r"search|google|bing|find online|look up|"
    r"website|url|http|www\.|\.com|\.org|\.io|"
    r"recent|right now|live|real.?time|"
    r"who is|what is happening|what happened|"
    r"2024|2025|2026"
    r")\b",
    re.IGNORECASE,
)

_RAG_KW = re.compile(
    r"\b("
    r"document|file|pdf|contract|report|invoice|"
    r"uploaded|attachment|my notes|the notes|"
    r"what does it say|according to|from the|in the doc|"
    r"summarize|summarise|extract|highlight"
    r")\b",
    re.IGNORECASE,
)

_MEMORY_KW = re.compile(
    r"\b("
    r"remember(?: that| me| this)?|"
    r"don'?t forget|save this|store this|note that|"
    r"my name|my age|my job|my preference|my email|"
    r"i told you|you know (me|my)|what do you know about me|"
    r"forget (that|me|this)"
    r")\b",
    re.IGNORECASE,
)

# Queries that look like personal-context questions but need memory retrieval
_MEMORY_RETRIEVE_KW = re.compile(
    r"\b("
    r"what('?s| is) my |do you (know|remember) (my|what)|"
    r"what did i (tell|say|mention)|"
    r"who am i|what are my"
    r")\b",
    re.IGNORECASE,
)


def _heuristic(query: str) -> tuple[str, ...] | None:
    """Fast keyword-based classification. Returns None if ambiguous → LLM fallback."""
    q = query.strip()
    tags: list[str] = []

    if _BROWSER_KW.search(q):
        tags.append("use_browser")
    if _RAG_KW.search(q):
        tags.append("use_rag")
    if _MEMORY_KW.search(q) or _MEMORY_RETRIEVE_KW.search(q):
        tags.append("use_memory")

    if not tags:
        # No signals → direct (greetings, math, general knowledge, etc.)
        return ("direct",)

    if len(tags) == 1:
        return tuple(tags)

    # Multiple signals → ambiguous, let LLM decide
    return None


# ── LLM fallback (cached) ─────────────────────────────────────────────────────

@lru_cache(maxsize=ROUTER_CACHE_SIZE)
def _classify_llm(query: str) -> tuple[str, ...]:
    """LLM-based fallback for ambiguous queries only. Cached."""
    prompt = ROUTER_PROMPT.format(query=query)
    resp = _llm.chat(
        [{"role": "user", "content": prompt}],
        model=MODEL_ROUTER["fast"],
    )
    raw = (resp.content or "").strip()
    try:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            tags = json.loads(match.group())
            valid = [t for t in tags if t in _VALID_TAGS]
            if valid:
                return tuple(valid)
    except (json.JSONDecodeError, TypeError):
        pass
    logger.warning(f"router LLM parse failed: {raw!r} — falling back to direct")
    return ("direct",)


# ── Public interface ──────────────────────────────────────────────────────────

def classify(query: str) -> tuple[str, ...]:
    """Classify a user query into routing tags.

    Fast path: keyword heuristic (<1ms).
    Slow path: LLM call only for ambiguous multi-signal queries (cached).
    Returns a tuple — callers iterate it like a list.
    """
    t0 = perf_counter()

    result = _heuristic(query)
    elapsed_ms = (perf_counter() - t0) * 1000

    if result is not None:
        logger.debug(f"router heuristic: {elapsed_ms:.1f}ms → {list(result)}")
        return result

    # Ambiguous — check LLM cache first, then call if needed
    before = _classify_llm.cache_info().hits
    result = _classify_llm(query)
    elapsed_ms = (perf_counter() - t0) * 1000
    if _classify_llm.cache_info().hits > before:
        logger.debug(f"router LLM cache hit: {elapsed_ms:.1f}ms → {list(result)}")
    else:
        logger.info(f"router LLM fallback: {elapsed_ms:.0f}ms → {list(result)}")

    return result
