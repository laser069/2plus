import time
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import ollama
from loguru import logger

from config.settings import (
    MODEL_ROUTER,
    OLLAMA_BASE_URL,
    LOG_DIR,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
_call_log = Path(LOG_DIR) / "calls.jsonl"

logger.add(str(Path(LOG_DIR) / "2plus.log"), rotation="10 MB", retention=3)


# ── Unified tool-call descriptor ─────────────────────────────────────────────

class _Func:
    """Minimal function descriptor compatible with Ollama and OpenAI shapes."""
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """Normalised tool-call object returned by LLMClient regardless of provider."""
    __slots__ = ("function", "id")

    def __init__(self, name: str, arguments: Any, id: str = "") -> None:
        self.function = _Func(name, arguments)
        self.id = id


# ── Public response type ──────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    content: str
    tool_calls: list[Any] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_call(model: str, kind: str, latency_ms: float, success: bool, chars_in: int) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "kind": kind,
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "chars_in": chars_in,
    }
    with open(_call_log, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _is_openrouter(model: str) -> bool:
    """OpenRouter models always use provider/model-name format."""
    return "/" in model


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert canonical message list to OpenAI-compatible format for OpenRouter."""
    result: list[dict] = []
    last_calls: list[tuple[str, str]] = []  # (tool_name, call_id) from last assistant msg

    for msg in messages:
        role = msg["role"]

        if role == "assistant" and msg.get("tool_calls"):
            oai_calls = []
            last_calls = []
            for i, tc in enumerate(msg["tool_calls"]):
                tc_id = getattr(tc, "id", "") or f"call_{i}"
                name = tc.function.name
                args = tc.function.arguments
                if isinstance(args, dict):
                    args = json.dumps(args)
                oai_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args or "{}"},
                })
                last_calls.append((name, tc_id))
            result.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": oai_calls,
            })

        elif role == "tool":
            name = msg.get("name", "")
            tc_id = next((cid for n, cid in last_calls if n == name), "call_0")
            result.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": msg.get("content", ""),
            })

        else:
            result.append({k: v for k, v in msg.items() if k != "tool_calls"})

    return result


def _inject_no_think(messages: list[dict]) -> list[dict]:
    """Prepend /no_think to first user message (Qwen3-specific directive)."""
    msgs = list(messages)
    for i, m in enumerate(msgs):
        if m["role"] == "user" and "/no_think" not in (m.get("content") or ""):
            msgs[i] = {**m, "content": "/no_think\n" + (m["content"] or "")}
            break
    return msgs


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self) -> None:
        self._ollama = ollama.Client(host=OLLAMA_BASE_URL)
        self._openai_client: Any = None  # lazy-init on first OpenRouter call

    @property
    def _or_client(self) -> Any:
        if self._openai_client is None:
            if not OPENROUTER_API_KEY:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set in .env — "
                    "add it to use OpenRouter models."
                )
            from openai import OpenAI
            self._openai_client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
        return self._openai_client

    # ── chat ─────────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        think: bool = False,
    ) -> ChatResponse:
        model = model or MODEL_ROUTER["default"]
        chars_in = sum(len(m.get("content") or "") for m in messages)
        t0 = time.perf_counter()
        success = False
        try:
            if _is_openrouter(model):
                content, tool_calls = self._chat_openrouter(messages, model, tools)
            else:
                content, tool_calls = self._chat_ollama(messages, model, tools, think)
            success = True
            return ChatResponse(
                content=content,
                tool_calls=tool_calls,
                model=model,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            logger.error(f"chat error ({model}): {exc}")
            return ChatResponse(content="", model=model)
        finally:
            _log_call(model, "chat", (time.perf_counter() - t0) * 1000, success, chars_in)

    def _chat_ollama(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        think: bool,
    ) -> tuple[str, list]:
        if not think:
            messages = _inject_no_think(messages)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self._ollama.chat(**kwargs)
        msg = resp.message
        # Wrap Ollama tool calls in _ToolCall so agent accesses uniform interface
        tcs = []
        for tc in (msg.tool_calls or []):
            tcs.append(_ToolCall(tc.function.name, tc.function.arguments or {}))
        return msg.content or "", tcs

    def _chat_openrouter(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
    ) -> tuple[str, list]:
        oai_msgs = _to_openai_messages(messages)
        kwargs: dict[str, Any] = {"model": model, "messages": oai_msgs}
        if tools:
            kwargs["tools"] = tools
        resp = self._or_client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        tcs = []
        for tc in (msg.tool_calls or []):
            args = tc.function.arguments
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, TypeError):
                pass
            tcs.append(_ToolCall(tc.function.name, args or {}, tc.id))
        return content, tcs

    # ── chat_stream ───────────────────────────────────────────────────────────

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        think: bool = False,
    ) -> Generator[str, None, None]:
        """Stream final-answer tokens. Do NOT use when tools are needed."""
        model = model or MODEL_ROUTER["default"]
        chars_in = sum(len(m.get("content") or "") for m in messages)
        t0 = time.perf_counter()
        success = False
        try:
            if _is_openrouter(model):
                yield from self._stream_openrouter(messages, model)
            else:
                yield from self._stream_ollama(messages, model, think)
            success = True
        except Exception as exc:
            logger.error(f"chat_stream error ({model}): {exc}")
            yield f"[stream error: {exc}]"
        finally:
            _log_call(model, "chat_stream", (time.perf_counter() - t0) * 1000, success, chars_in)

    def _stream_ollama(
        self, messages: list[dict], model: str, think: bool
    ) -> Generator[str, None, None]:
        if not think:
            messages = _inject_no_think(messages)
        for chunk in self._ollama.chat(model=model, messages=messages, stream=True):
            delta = (chunk.message.content or "") if chunk.message else ""
            if delta:
                yield delta

    def _stream_openrouter(
        self, messages: list[dict], model: str
    ) -> Generator[str, None, None]:
        oai_msgs = _to_openai_messages(messages)
        stream = self._or_client.chat.completions.create(
            model=model,
            messages=oai_msgs,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    # ── embed ─────────────────────────────────────────────────────────────────

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Embeddings always use the local Ollama model."""
        model = model or MODEL_ROUTER["embed"]
        t0 = time.perf_counter()
        success = False
        try:
            resp = self._ollama.embeddings(model=model, prompt=text)
            success = True
            return resp["embedding"]
        except Exception as exc:
            logger.error(f"embed error: {exc}")
            return []
        finally:
            _log_call(model, "embed", (time.perf_counter() - t0) * 1000, success, len(text))
