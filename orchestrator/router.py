import json
import re
from functools import lru_cache

from loguru import logger

from serving.llm_client import LLMClient
from orchestrator.prompts import ROUTER_PROMPT
from config.settings import MODEL_ROUTER, ROUTER_CACHE_SIZE

_llm = LLMClient()
_VALID_TAGS = {"use_rag", "use_browser", "use_memory", "direct"}


@lru_cache(maxsize=ROUTER_CACHE_SIZE)
def _classify_cached(query: str) -> tuple[str, ...]:
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

    logger.warning(f"router parse failed on: {raw!r} — falling back to direct")
    return ("direct",)


def classify(query: str) -> tuple[str, ...]:
    """Classify a user query into routing tags using the fast model.

    Returns a tuple (hashable). Results are LRU-cached — repeated queries skip
    the LLM call entirely. Callers iterate it like a list.
    """
    before = _classify_cached.cache_info().hits
    result = _classify_cached(query)
    if _classify_cached.cache_info().hits > before:
        logger.debug(f"router cache hit: {query[:40]!r} → {list(result)}")
    return result
