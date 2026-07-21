from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Generator

from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config.settings import MAX_REACT_STEPS, CTX_BUDGET_CHARS
from serving.llm_client import LLMClient, inject_no_think_lc
from memory.convo import ConvoMemory
import memory.user_facts as uf
from tools.registry import lc_tools_for_routes, TOOL_REGISTRY
from orchestrator.router import classify
from orchestrator.prompts import SYSTEM_PROMPT, FACT_EXTRACT_PROMPT

_llm = LLMClient()


@dataclass
class AgentResponse:
    answer: str
    citations: list[str] = field(default_factory=list)
    steps: int = 0
    routes: list[str] = field(default_factory=list)


def _extract_citations(text: str) -> list[str]:
    return re.findall(r"\[(?:doc|web):[^\]]+\]", text)


def _extract_facts(assistant_msg: str) -> None:
    """Parse new user facts from the assistant's response and upsert them."""
    prompt = FACT_EXTRACT_PROMPT.format(message=assistant_msg[:1000])
    resp = _llm.chat([{"role": "user", "content": prompt}], model=None)
    raw = (resp.content or "").strip()
    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            for k, v in facts.items():
                if k and v:
                    uf.upsert(str(k), str(v))
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass


def _run_tool_call(tc: dict, remaining: int) -> tuple[ToolMessage, int]:
    """Dispatch one LangChain tool call. Returns (ToolMessage, remaining_budget)."""
    name = tc["name"]
    args = tc.get("args") or {}

    tool = TOOL_REGISTRY.get(name)
    if tool:
        try:
            result = tool.handler(**args)
        except Exception as exc:
            result = f"[tool error: {exc}]"
    else:
        result = f"[unknown tool: {name}]"

    result_str = str(result)[:max(remaining, 200)]
    remaining -= len(result_str)
    logger.info(f"tool={name} result_chars={len(result_str)}")

    return ToolMessage(content=result_str, tool_call_id=tc.get("id") or "", name=name), remaining


class Agent:
    def __init__(self, convo: ConvoMemory) -> None:
        self.convo = convo

    def run(self, query: str, model: str | None = None, think: bool = False, fallback_model: str | None = None) -> AgentResponse:
        # 1. Route
        routes = classify(query)
        logger.info(f"query={query[:60]!r}  model={model}  fallback={fallback_model}  routes={routes}")

        # 2. Build base context (memory-injected, budget-tracked)
        messages, chars_used = self.convo.get_context(query, SYSTEM_PROMPT)
        remaining = CTX_BUDGET_CHARS - chars_used

        # 3. Append user query
        messages.append(HumanMessage(content=query))
        if not think:
            messages = inject_no_think_lc(messages)

        # 4. Resolve LangChain tools for active routes
        lc_tools = lc_tools_for_routes(routes)
        chat_model = _llm.get_chat_model(model, tools=lc_tools or None, fallback_model=fallback_model)

        # 5. ReAct loop
        steps = 0
        while steps < MAX_REACT_STEPS:
            resp: AIMessage = chat_model.invoke(messages)
            steps += 1

            if not resp.tool_calls:
                # Final answer
                answer = (resp.content or "").strip()
                break

            # Dispatch each tool call
            messages.append(resp)
            for tc in resp.tool_calls:
                tool_msg, remaining = _run_tool_call(tc, remaining)
                messages.append(tool_msg)

            if remaining <= 0:
                messages.append(HumanMessage(
                    content="[context budget reached — answer with information gathered so far]"
                ))
                no_tools_model = _llm.get_chat_model(model, fallback_model=fallback_model)
                final = no_tools_model.invoke(messages)
                answer = (final.content or "").strip()
                steps += 1
                break
        else:
            answer = "I reached the step limit. Here is what I found so far."

        # 6. Update convo memory
        self.convo.add_turn("user", query)
        self.convo.add_turn("assistant", answer)
        self.convo.maybe_summarise(_llm)

        # 7. Extract any new user facts from answer
        _extract_facts(answer)

        return AgentResponse(
            answer=answer,
            citations=_extract_citations(answer),
            steps=steps,
            routes=routes,
        )

    def run_stream(
        self, query: str, model: str | None = None, think: bool = False, fallback_model: str | None = None
    ) -> Generator:
        """Mixed generator. Yields:
          dict {"type": "routing", "routes": [...]}
          dict {"type": "tool", "name": str, "step": int}
          str  — final answer token chunks
          dict {"type": "done", "citations": [...], "routes": [...], "steps": int}
        """
        # 1. Route
        routes = classify(query)
        logger.info(f"[stream] query={query[:60]!r}  model={model}  fallback={fallback_model}  routes={routes}")
        yield {"type": "routing", "routes": routes}

        # 2. Build context
        messages, chars_used = self.convo.get_context(query, SYSTEM_PROMPT)
        remaining = CTX_BUDGET_CHARS - chars_used
        messages.append(HumanMessage(content=query))
        if not think:
            messages = inject_no_think_lc(messages)

        lc_tools = lc_tools_for_routes(routes)

        # 3a. Fast path — no tools needed → stream the answer live token-by-token
        answer_chunks: list[str] = []
        if not lc_tools:
            chat_model = _llm.get_chat_model(model, fallback_model=fallback_model)
            for chunk in chat_model.stream(messages):
                delta = chunk.content or ""
                if delta:
                    answer_chunks.append(delta)
                    yield delta
            answer = "".join(answer_chunks).strip()
            self.convo.add_turn("user", query)
            self.convo.add_turn("assistant", answer)
            self.convo.maybe_summarise(_llm)
            yield {
                "type": "done",
                "citations": _extract_citations(answer),
                "routes": routes,
                "steps": 1,
            }
            # Fact extraction off the critical path — UI already has its answer
            threading.Thread(target=_extract_facts, args=(answer,), daemon=True).start()
            return

        # 3b. ReAct loop — blocking for tool steps
        chat_model = _llm.get_chat_model(model, tools=lc_tools, fallback_model=fallback_model)
        steps = 0
        while steps < MAX_REACT_STEPS:
            resp: AIMessage = chat_model.invoke(messages)
            steps += 1

            if not resp.tool_calls:
                # Yield content already received — avoids a second model call
                content = resp.content or ""
                answer_chunks.append(content)
                yield content
                break

            messages.append(resp)
            for tc in resp.tool_calls:
                yield {"type": "tool", "name": tc["name"], "step": steps}
                tool_msg, remaining = _run_tool_call(tc, remaining)
                messages.append(tool_msg)

            if remaining <= 0:
                messages.append(HumanMessage(
                    content="[context budget reached — answer with information gathered so far]"
                ))
                no_tools_model = _llm.get_chat_model(model, fallback_model=fallback_model)
                for chunk in no_tools_model.stream(messages):
                    delta = chunk.content or ""
                    if delta:
                        answer_chunks.append(delta)
                        yield delta
                steps += 1
                break
        else:
            fallback = "I reached the step limit. Here is what I found so far."
            answer_chunks.append(fallback)
            yield fallback

        answer = "".join(answer_chunks).strip()

        # 4. Update convo memory
        self.convo.add_turn("user", query)
        self.convo.add_turn("assistant", answer)
        self.convo.maybe_summarise(_llm)

        yield {
            "type": "done",
            "citations": _extract_citations(answer),
            "routes": routes,
            "steps": steps,
        }

        # 5. Extract facts off the critical path — UI already has its answer
        threading.Thread(target=_extract_facts, args=(answer,), daemon=True).start()
