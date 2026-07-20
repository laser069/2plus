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
    GROQ_API_KEY,
    GROQ_BASE_URL,
)
from config.logging_config import setup_logging

setup_logging()

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
_call_log = Path(LOG_DIR) / "calls.jsonl"


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


def _provider(model: str) -> str:
    """Resolve which backend serves a model.

    - "groq/<name>"  → Groq (OpenAI-compatible; prefix stripped before send)
    - "<vendor>/<name>" → OpenRouter
    - bare name      → local Ollama
    """
    if model.startswith("groq/"):
        return "groq"
    if "/" in model:
        return "openrouter"
    return "ollama"


def _is_cloud(model: str | None) -> bool:
    """True for any non-Ollama (cloud) model string."""
    return bool(model) and _provider(model) != "ollama"


# Per-provider OpenAI-compatible client config: (api_key, base_url)
_OAI_CONFIG = {
    "openrouter": (OPENROUTER_API_KEY, OPENROUTER_BASE_URL),
    "groq": (GROQ_API_KEY, GROQ_BASE_URL),
}


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
        self._oai_clients: dict[str, Any] = {}  # lazy per-provider (openrouter/groq)

    def _oai_client(self, provider: str) -> Any:
        """Lazily build & cache an OpenAI-compatible client for the provider."""
        if provider not in self._oai_clients:
            api_key, base_url = _OAI_CONFIG[provider]
            if not api_key:
                raise RuntimeError(
                    f"{provider.upper()}_API_KEY is not set in .env — "
                    f"add it to use {provider} models."
                )
            from openai import OpenAI
            self._oai_clients[provider] = OpenAI(base_url=base_url, api_key=api_key)
        return self._oai_clients[provider]

    # ── chat ─────────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        think: bool = False,
        fallback_model: str | None = None,
    ) -> ChatResponse:
        model = model or MODEL_ROUTER["default"]
        chars_in = sum(len(m.get("content") or "") for m in messages)
        t0 = time.perf_counter()
        success = False
        try:
            prov = _provider(model)
            if prov != "ollama":
                content, tool_calls = self._chat_openai(messages, model, tools, prov)
            else:
                try:
                    content, tool_calls = self._chat_ollama(messages, model, tools, think)
                except Exception as ollama_exc:
                    if _is_cloud(fallback_model):
                        logger.warning(f"Ollama failed → falling back to {fallback_model} | {ollama_exc!s:.80}")
                        content, tool_calls = self._chat_openai(
                            messages, fallback_model, tools, _provider(fallback_model)
                        )
                        model = fallback_model
                    else:
                        raise
            latency = (time.perf_counter() - t0) * 1000
            success = True
            tc_count = len(tool_calls)
            logger.info(
                f"chat  {model}  {latency:.0f}ms"
                + (f"  tool_calls={tc_count}" if tc_count else "")
            )
            return ChatResponse(content=content, tool_calls=tool_calls, model=model, latency_ms=latency)
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

    def _chat_openai(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        provider: str,
    ) -> tuple[str, list]:
        oai_msgs = _to_openai_messages(messages)
        api_model = model.split("/", 1)[1] if provider == "groq" else model
        kwargs: dict[str, Any] = {"model": api_model, "messages": oai_msgs}
        if tools:
            kwargs["tools"] = tools
        resp = self._oai_client(provider).chat.completions.create(**kwargs)
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
        fallback_model: str | None = None,
    ) -> Generator[str, None, None]:
        """Stream final-answer tokens. Do NOT use when tools are needed."""
        model = model or MODEL_ROUTER["default"]
        chars_in = sum(len(m.get("content") or "") for m in messages)
        t0 = time.perf_counter()
        success = False
        logger.info(f"stream {model}  start")
        try:
            prov = _provider(model)
            if prov != "ollama":
                yield from self._stream_openai(messages, model, prov)
            else:
                try:
                    yield from self._stream_ollama(messages, model, think)
                except Exception as ollama_exc:
                    if _is_cloud(fallback_model):
                        logger.warning(f"Ollama stream failed → falling back to {fallback_model}")
                        yield from self._stream_openai(
                            messages, fallback_model, _provider(fallback_model)
                        )
                    else:
                        raise ollama_exc
            success = True
            logger.info(f"stream {model}  done  {(time.perf_counter()-t0)*1000:.0f}ms")
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

    def _stream_openai(
        self, messages: list[dict], model: str, provider: str
    ) -> Generator[str, None, None]:
        oai_msgs = _to_openai_messages(messages)
        api_model = model.split("/", 1)[1] if provider == "groq" else model
        stream = self._oai_client(provider).chat.completions.create(
            model=api_model,
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
            latency = (time.perf_counter() - t0) * 1000
            logger.debug(f"embed  {model}  {latency:.0f}ms  chars={len(text)}")
            return resp["embedding"]
        except Exception as exc:
            logger.error(f"embed error: {exc}")
            return []
        finally:
            _log_call(model, "embed", (time.perf_counter() - t0) * 1000, success, len(text))
