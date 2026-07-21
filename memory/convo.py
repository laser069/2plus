from __future__ import annotations
import threading
from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

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
        self._turns: list[BaseMessage] = []   # verbatim recent turns
        self._lock = threading.Lock()

    def add_turn(self, role: str, content: str) -> None:
        msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        with self._lock:
            self._turns.append(msg)

    def maybe_summarise(self, llm: LLMClient, model: str | None = None) -> None:
        """Compress the messages that fall outside the trimmed window into the
        persisted rolling summary. Uses LangChain's trim_messages with
        token_counter=len so each message counts as one unit, i.e. a
        message-count window rather than a token-budget one.

        Routes with the active chat model so a cloud session doesn't trigger a
        local model load in the background (and a local session reuses the
        resident model)."""
        with self._lock:
            if len(self._turns) <= CONVO_WINDOW:
                return
            kept = trim_messages(
                self._turns,
                max_tokens=CONVO_WINDOW,
                token_counter=len,
                strategy="last",
            )
            n_dropped = len(self._turns) - len(kept)
            if n_dropped <= 0:
                return
            to_compress = self._turns[:n_dropped]
            self._turns = self._turns[n_dropped:]

        text = "\n".join(f"{m.type.upper()}: {m.content}" for m in to_compress)
        existing = uf.get(_SUMMARY_KEY) or ""
        prompt = (
            f"Summarise this conversation excerpt in ≤{SUMMARY_MAX_CHARS} chars. "
            f"Existing summary to extend: {existing}\n\nExcerpt:\n{text}"
        )
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            model=model,
        )
        summary = (resp.content or "").strip()[:SUMMARY_MAX_CHARS]
        uf.upsert(_SUMMARY_KEY, summary)

    def get_context(self, query: str, system_prompt: str) -> tuple[list[BaseMessage], int]:
        """Build message list for the agent. Returns (messages, chars_used)."""
        chars_used = 0

        # 1. System message
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        chars_used += len(system_prompt)

        # 2. Rolling summary (if exists)
        summary = uf.get(_SUMMARY_KEY)
        if summary:
            block = f"[Conversation so far]\n{summary}"
            messages.append(SystemMessage(content=block))
            chars_used += len(block)

        # 3. Relevant user facts
        facts_block = uf.get_relevant(query, budget=FACTS_MAX_CHARS)
        if facts_block:
            block = f"[Known user facts]\n{facts_block}"
            messages.append(SystemMessage(content=block))
            chars_used += len(block)

        # 4. Recent verbatim turns
        with self._lock:
            recent = list(self._turns[-CONVO_WINDOW:])

        for turn in recent:
            messages.append(turn)
            chars_used += len(turn.content or "")

        return messages, chars_used

    def clear(self) -> None:
        with self._lock:
            self._turns = []
        uf.delete(_SUMMARY_KEY)
