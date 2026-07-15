from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter
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


def _bg(fn, *args) -> None:
    """Fire-and-forget daemon thread."""
    threading.Thread(target=fn, args=args, daemon=True).start()


def _dispatch_tool(name: str, args: dict, remaining: int) -> tuple[str, str]:
    """Execute one tool call. Returns (name, result_str). Pure — no shared state mutation."""
    tool = TOOL_REGISTRY.get(name)
    if tool:
        try:
            result = tool.handler(**args)
        except Exception as exc:
            result = f"[tool error: {exc}]"
    else:
        result = f"[unknown tool: {name}]"
    result_str = str(result)[:max(remaining, 200)]
    logger.info(f"tool={name} result_chars={len(result_str)}")
    return name, result_str


class Agent:
    def __init__(self, convo: ConvoMemory) -> None:
        self.convo = convo

    def run(self, query: str, model: str | None = None, think: bool = False, fallback_model: str | None = None) -> AgentResponse:
        t0 = perf_counter()

        # 1. Route
        routes = classify(query)
        t_router = perf_counter()
        logger.info(f"query={query[:60]!r}  model={model}  fallback={fallback_model}  routes={list(routes)}")

        # 2. Build base context (memory-injected, budget-tracked)
        messages, chars_used = self.convo.get_context(query, SYSTEM_PROMPT)
        remaining = CTX_BUDGET_CHARS - chars_used
        t_context = perf_counter()

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
                answer = resp.content.strip()
                break

            messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": resp.tool_calls})

            # Parallel tool dispatch
            tool_inputs = []
            for tc in resp.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_inputs.append((name, args))

            if len(tool_inputs) == 1:
                name, args = tool_inputs[0]
                t_name, result_str = _dispatch_tool(name, args, remaining)
                remaining -= len(result_str)
                messages.append({"role": "tool", "content": result_str, "name": t_name})
            else:
                with ThreadPoolExecutor() as ex:
                    futures = {ex.submit(_dispatch_tool, n, a, remaining): n for n, a in tool_inputs}
                    for fut in as_completed(futures):
                        t_name, result_str = fut.result()
                        remaining -= len(result_str)
                        messages.append({"role": "tool", "content": result_str, "name": t_name})

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

        total_ms = (perf_counter() - t0) * 1000
        router_ms = (t_router - t0) * 1000
        ctx_ms = (t_context - t_router) * 1000
        logger.info(
            f"[perf] router={router_ms:.0f}ms ctx={ctx_ms:.0f}ms "
            f"total={total_ms:.0f}ms routes={list(routes)} steps={steps}"
        )

        # 6. Update convo memory (background)
        _bg(self.convo.add_turn, "user", query)
        _bg(self.convo.add_turn, "assistant", answer)
        _bg(self.convo.maybe_summarise, _llm)

        # 7. Extract any new user facts from answer (background)
        _bg(_extract_facts, answer)

        return AgentResponse(
            answer=answer,
            citations=_extract_citations(answer),
            steps=steps,
            routes=list(routes),
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
        t0 = perf_counter()

        # 1. Route
        routes = classify(query)
        t_router = perf_counter()
        router_ms = (t_router - t0) * 1000
        logger.info(f"[stream] query={query[:60]!r}  model={model}  fallback={fallback_model}  routes={list(routes)}")
        yield {"type": "routing", "routes": list(routes)}

        # 2. Build context
        messages, chars_used = self.convo.get_context(query, SYSTEM_PROMPT)
        remaining = CTX_BUDGET_CHARS - chars_used
        messages.append({"role": "user", "content": query})
        _, tool_schemas = tools_for_routes(routes)
        t_context = perf_counter()
        ctx_ms = (t_context - t_router) * 1000

        # 3. ReAct loop
        steps = 0
        answer_chunks: list[str] = []
        t_first_tok: float | None = None

        # Fast path: direct route → skip blocking chat(), stream immediately
        if list(routes) == ["direct"]:
            for chunk in _llm.chat_stream(messages, model=model, think=think, fallback_model=fallback_model):
                if t_first_tok is None:
                    t_first_tok = perf_counter()
                answer_chunks.append(chunk)
                yield chunk
            steps = 1
        else:
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
                    # Model gave final answer — yield it directly (already generated)
                    content = resp.content or ""
                    if t_first_tok is None:
                        t_first_tok = perf_counter()
                    answer_chunks.append(content)
                    yield content
                    break

                messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": resp.tool_calls})

                # Parallel tool dispatch
                tool_inputs = []
                for tc in resp.tool_calls:
                    name = tc.function.name
                    args = tc.function.arguments or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    yield {"type": "tool", "name": name, "step": steps}
                    tool_inputs.append((name, args))

                if len(tool_inputs) == 1:
                    name, args = tool_inputs[0]
                    t_name, result_str = _dispatch_tool(name, args, remaining)
                    remaining -= len(result_str)
                    messages.append({"role": "tool", "content": result_str, "name": t_name})
                else:
                    with ThreadPoolExecutor() as ex:
                        futures = {ex.submit(_dispatch_tool, n, a, remaining): n for n, a in tool_inputs}
                        for fut in as_completed(futures):
                            t_name, result_str = fut.result()
                            remaining -= len(result_str)
                            messages.append({"role": "tool", "content": result_str, "name": t_name})

                if remaining <= 0:
                    messages.append({
                        "role": "user",
                        "content": "[context budget reached — answer with information gathered so far]",
                    })
                    for chunk in _llm.chat_stream(messages, model=model, think=think, fallback_model=fallback_model):
                        if t_first_tok is None:
                            t_first_tok = perf_counter()
                        answer_chunks.append(chunk)
                        yield chunk
                    steps += 1
                    break
            else:
                fallback = "I reached the step limit. Here is what I found so far."
                answer_chunks.append(fallback)
                yield fallback

        answer = "".join(answer_chunks).strip()

        t_done = perf_counter()
        ttft_ms = (t_first_tok - t0) * 1000 if t_first_tok else None
        total_ms = (t_done - t0) * 1000
        logger.info(
            f"[perf] router={router_ms:.0f}ms ctx={ctx_ms:.0f}ms "
            f"ttft={ttft_ms:.0f}ms total={total_ms:.0f}ms "
            f"routes={list(routes)} steps={steps}"
            if ttft_ms is not None else
            f"[perf] router={router_ms:.0f}ms ctx={ctx_ms:.0f}ms "
            f"total={total_ms:.0f}ms routes={list(routes)} steps={steps}"
        )

        # 4. Update convo memory (background — non-blocking)
        _bg(self.convo.add_turn, "user", query)
        _bg(self.convo.add_turn, "assistant", answer)
        _bg(self.convo.maybe_summarise, _llm)

        # 5. Extract facts (background — non-blocking)
        _bg(_extract_facts, answer)

        yield {
            "type": "done",
            "citations": _extract_citations(answer),
            "routes": list(routes),
            "steps": steps,
        }
