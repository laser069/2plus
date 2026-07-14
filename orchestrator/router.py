import json
import re

from loguru import logger

from serving.llm_client import LLMClient
from orchestrator.prompts import ROUTER_PROMPT
from config.settings import MODEL_ROUTER

_llm = LLMClient()
_VALID_TAGS = {"use_rag", "use_browser", "use_memory", "direct"}


def classify(query: str) -> list[str]:
    """Classify a user query into routing tags using the fast model."""
    prompt = ROUTER_PROMPT.format(query=query)
    resp = _llm.chat(
        [{"role": "user", "content": prompt}],
        model=MODEL_ROUTER["fast"],
    )
    raw = (resp.content or "").strip()

    # Try to parse JSON array from response
    try:
        # Find the first [...] block in the response
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            tags = json.loads(match.group())
            valid = [t for t in tags if t in _VALID_TAGS]
            if valid:
                return valid
    except (json.JSONDecodeError, TypeError):
        pass

    logger.warning(f"router parse failed on: {raw!r} — falling back to direct")
    return ["direct"]
