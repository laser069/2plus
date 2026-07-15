from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Generator

from loguru import logger

from config.settings import MAX_REACT_STEPS, CTX_BUDGET_CHARS
from serving.llm_client import LLMClient
from memory.convo import ConvoMemory
import memory.user_facts as uf
from tools.registry import tools_for_routes, TOOL_REGISTRY
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


class Agent:
    def __init__(self, convo: ConvoMemory) -> None:
        self.convo = convo

    def run(self, query: str, model: str | None = None, think: bool = False, fallback_model: str | None = None) -> AgentResponse:
        # 1. Route
        routes = classify(query)
        logger.info(f"routes={routes} query={query[:80]!r}")

        # 2. Build base context (memory-injected, budget-tracked)
        messages, chars_used = self.convo.get_context(query, SYSTEM_PROMPT)
        remaining = CTX_BUDGET_CHARS - chars_used

        # 3. Append user query
        messages.append({"role": "user", "content": query})

        # 4. Resolve tool schemas for active routes
        _, tool_schemas = tools_for_routes(routes)

        # 5. ReAct loop
        steps = 0
        while steps < MAX_REACT_STEPS:
            resp = _llm.chat(
                messages,
                model=model,
                think=think,
                tools=tool_schemas if tool_schemas else None,
                fallback_model=fallback_model,
            )
            steps += 1

            if not resp.tool_calls:
                # Final answer
                answer = resp.content.strip()
                break

            # Dispatch each tool call
            messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": resp.tool_calls})
            for tc in resp.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

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

                messages.append({
                    "role": "tool",
                    "content": result_str,
                    "name": name,
                })
                logger.info(f"tool={name} result_chars={len(result_str)}")

            if remaining <= 0:
                messages.append({
                    "role": "user",
                    "content": "[context budget reached — answer with information gathered so far]",
                })
                final = _llm.chat(messages, model=model, think=think, fallback_model=fallback_model)
                answer = final.content.strip()
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
        """
        Mixed generator. Yields:
          dict {"type": "routing", "routes": [...]}
          dict {"type": "tool", "name": str, "step": int}
          str  — final answer token chunks
          dict {"type": "done", "citations": [...], "routes": [...], "steps": int}
        """
        # 1. Route
        routes = classify(query)
        logger.info(f"routes={routes} query={query[:80]!r}")
        yield {"type": "routing", "routes": routes}

        # 2. Build context
        messages, chars_used = self.convo.get_context(query, SYSTEM_PROMPT)
        remaining = CTX_BUDGET_CHARS - chars_used
        messages.append({"role": "user", "content": query})
        _, tool_schemas = tools_for_routes(routes)

        # 3. ReAct loop — blocking for tool steps
        steps = 0
        answer_chunks: list[str] = []
        while steps < MAX_REACT_STEPS:
            resp = _llm.chat(
                messages,
                model=model,
                think=think,
                tools=tool_schemas if tool_schemas else None,
                fallback_model=fallback_model,
            )
            steps += 1

            if not resp.tool_calls:
                # Stream the final answer
                messages.append({"role": "assistant", "content": resp.content or ""})
                for chunk in _llm.chat_stream(messages[:-1], model=model, think=think, fallback_model=fallback_model):
                    answer_chunks.append(chunk)
                    yield chunk
                break

            messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": resp.tool_calls})
            for tc in resp.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                yield {"type": "tool", "name": name, "step": steps}

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
                messages.append({"role": "tool", "content": result_str, "name": name})
                logger.info(f"tool={name} result_chars={len(result_str)}")

            if remaining <= 0:
                messages.append({
                    "role": "user",
                    "content": "[context budget reached — answer with information gathered so far]",
                })
                for chunk in _llm.chat_stream(messages, model=model, think=think, fallback_model=fallback_model):
                    answer_chunks.append(chunk)
                    yield chunk
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

        # 5. Extract facts
        _extract_facts(answer)

        yield {
            "type": "done",
            "citations": _extract_citations(answer),
            "routes": routes,
            "steps": steps,
        }
