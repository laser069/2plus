from __future__ import annotations
import threading
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
    """Manages in-process turn window + persistent rolling summary.

    Thread-safe: add_turn/get_context/maybe_summarise may be called from
    background daemon threads spawned by the agent.
    """

    def __init__(self) -> None:
        self._turns: list[dict] = []
        self._lock = threading.Lock()

    def add_turn(self, role: str, content: str) -> None:
        with self._lock:
            self._turns.append({"role": role, "content": content})

    def maybe_summarise(self, llm: LLMClient) -> None:
        """Compress oldest half of turns into summary when window overflows."""
        with self._lock:
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
            model=None,
        )
        summary = (resp.content or "").strip()[:SUMMARY_MAX_CHARS]
        uf.upsert(_SUMMARY_KEY, summary)

    def get_context(self, query: str, system_prompt: str) -> tuple[list[dict], int]:
        """Build message list for the agent. Returns (messages, chars_used)."""
        chars_used = 0

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        chars_used += len(system_prompt)

        summary = uf.get(_SUMMARY_KEY)
        if summary:
            block = f"[Conversation so far]\n{summary}"
            messages.append({"role": "system", "content": block})
            chars_used += len(block)

        facts_block = uf.get_relevant(query, budget=FACTS_MAX_CHARS)
        if facts_block:
            block = f"[Known user facts]\n{facts_block}"
            messages.append({"role": "system", "content": block})
            chars_used += len(block)

        with self._lock:
            recent = list(self._turns[-CONVO_WINDOW:])

        for turn in recent:
            messages.append(turn)
            chars_used += len(turn.get("content") or "")

        return messages, chars_used

    def clear(self) -> None:
        with self._lock:
            self._turns = []
        uf.delete(_SUMMARY_KEY)
