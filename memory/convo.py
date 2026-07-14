from __future__ import annotations
from typing import TYPE_CHECKING

from config.settings import (
    CONVO_WINDOW,
    SUMMARY_MAX_CHARS,
    FACTS_MAX_CHARS,
)
from memory import user_facts as uf

if TYPE_CHECKING:
    from serving.llm_client import LLMClient

_SUMMARY_KEY = "convo_summary"


class ConvoMemory:
    """Manages in-process turn window + persistent rolling summary."""

    def __init__(self) -> None:
        self._turns: list[dict] = []   # verbatim recent turns

    def add_turn(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})

    def maybe_summarise(self, llm: LLMClient) -> None:
        """Compress oldest half of turns into summary when window overflows."""
        if len(self._turns) <= CONVO_WINDOW:
            return

        half = len(self._turns) // 2
        to_compress = self._turns[:half]
        self._turns = self._turns[half:]

        text = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in to_compress)
        existing = uf.get(_SUMMARY_KEY) or ""
        prompt = (
            f"Summarise this conversation excerpt in ≤{SUMMARY_MAX_CHARS} chars. "
            f"Existing summary to extend: {existing}\n\nExcerpt:\n{text}"
        )
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            model=None,   # uses default
        )
        summary = (resp.content or "").strip()[:SUMMARY_MAX_CHARS]
        uf.upsert(_SUMMARY_KEY, summary)

    def get_context(self, query: str, system_prompt: str) -> tuple[list[dict], int]:
        """Build message list for the agent. Returns (messages, chars_used)."""
        chars_used = 0

        # 1. System message
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        chars_used += len(system_prompt)

        # 2. Rolling summary (if exists)
        summary = uf.get(_SUMMARY_KEY)
        if summary:
            block = f"[Conversation so far]\n{summary}"
            messages.append({"role": "system", "content": block})
            chars_used += len(block)

        # 3. Relevant user facts
        facts_block = uf.get_relevant(query, budget=FACTS_MAX_CHARS)
        if facts_block:
            block = f"[Known user facts]\n{facts_block}"
            messages.append({"role": "system", "content": block})
            chars_used += len(block)

        # 4. Recent verbatim turns
        for turn in self._turns[-CONVO_WINDOW:]:
            messages.append(turn)
            chars_used += len(turn.get("content") or "")

        return messages, chars_used

    def clear(self) -> None:
        self._turns = []
        uf.delete(_SUMMARY_KEY)
