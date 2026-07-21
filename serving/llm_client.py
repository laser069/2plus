import time
import json
from pathlib import Path
from typing import Any, Generator

from loguru import logger
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from config.settings import (
    MODEL_ROUTER,
    OLLAMA_BASE_URL,
    LOG_DIR,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
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

from dataclasses import dataclass, field


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
    if "/" in model:
        return "openrouter"
    return "ollama"


def _is_cloud(model: str | None) -> bool:
    """True for any non-Ollama (cloud) model string."""
    return bool(model) and _provider(model) != "ollama"


# Per-provider OpenAI-compatible client config: (api_key, base_url)
_OAI_CONFIG = {
    "openrouter": (OPENROUTER_API_KEY, OPENROUTER_BASE_URL),
}


def _to_lc_messages(messages: list[dict]) -> list[BaseMessage]:
    """Convert canonical message list to LangChain BaseMessage objects."""
    result: list[BaseMessage] = []
    last_calls: list[tuple[str, str]] = []  # (tool_name, call_id) from last assistant msg

    for msg in messages:
        role = msg["role"]

        if role == "system":
            result.append(SystemMessage(content=msg.get("content") or ""))

        elif role == "assistant" and msg.get("tool_calls"):
            lc_calls = []
            last_calls = []
            for i, tc in enumerate(msg["tool_calls"]):
                tc_id = getattr(tc, "id", "") or f"call_{i}"
                name = tc.function.name
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                lc_calls.append({"name": name, "args": args or {}, "id": tc_id})
                last_calls.append((name, tc_id))
            result.append(AIMessage(content=msg.get("content") or "", tool_calls=lc_calls))

        elif role == "assistant":
            result.append(AIMessage(content=msg.get("content") or ""))

        elif role == "tool":
            name = msg.get("name", "")
            tc_id = next((cid for n, cid in last_calls if n == name), "call_0")
            result.append(
                ToolMessage(content=msg.get("content", ""), tool_call_id=tc_id, name=name)
            )

        else:
            result.append(HumanMessage(content=msg.get("content") or ""))

    return result


def _from_lc_tool_calls(ai_msg: AIMessage) -> list[_ToolCall]:
    tcs = []
    for tc in (ai_msg.tool_calls or []):
        tcs.append(_ToolCall(tc["name"], tc.get("args") or {}, tc.get("id") or ""))
    return tcs


def _inject_no_think(messages: list[dict]) -> list[dict]:
    """Prepend /no_think to first user message (Qwen3-specific directive)."""
    msgs = list(messages)
    for i, m in enumerate(msgs):
        if m["role"] == "user" and "/no_think" not in (m.get("content") or ""):
            msgs[i] = {**m, "content": "/no_think\n" + (m["content"] or "")}
            break
    return msgs


def inject_no_think_lc(messages: list[BaseMessage]) -> list[BaseMessage]:
    """LangChain-message counterpart of _inject_no_think, for callers (e.g. the
    ReAct agent loop) that work with BaseMessage lists directly."""
    msgs = list(messages)
    for i, m in enumerate(msgs):
        if isinstance(m, HumanMessage) and "/no_think" not in (m.content or ""):
            msgs[i] = HumanMessage(content="/no_think\n" + (m.content or ""))
            break
    return msgs


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self) -> None:
        self._ollama_chat_models: dict[str, ChatOllama] = {}  # keyed by model name
        self._oai_chat_models: dict[str, ChatOpenAI] = {}     # keyed by "provider::model"
        self._embeddings: dict[str, OllamaEmbeddings] = {}

    def _ollama_chat(self, model: str) -> ChatOllama:
        if model not in self._ollama_chat_models:
            self._ollama_chat_models[model] = ChatOllama(model=model, base_url=OLLAMA_BASE_URL)
        return self._ollama_chat_models[model]

    def _oai_chat(self, model: str, provider: str) -> ChatOpenAI:
        key = f"{provider}::{model}"
        if key not in self._oai_chat_models:
            api_key, base_url = _OAI_CONFIG[provider]
            if not api_key:
                raise RuntimeError(
                    f"{provider.upper()}_API_KEY is not set in .env — "
                    f"add it to use {provider} models."
                )
            self._oai_chat_models[key] = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)
        return self._oai_chat_models[key]

    def _embedder(self, model: str) -> OllamaEmbeddings:
        if model not in self._embeddings:
            self._embeddings[model] = OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
        return self._embeddings[model]

    # ── native LangChain chat model access (for the ReAct agent loop) ──────────

    def get_chat_model(
        self,
        model: str | None = None,
        tools: list | None = None,
        fallback_model: str | None = None,
    ):
        """Return a LangChain chat Runnable bound with `tools`, wired with a
        native `.with_fallbacks()` cloud fallback when `fallback_model` is set.
        Used by the agent's ReAct loop, which works with BaseMessage lists and
        AIMessage.tool_calls directly rather than through chat()/chat_stream()."""
        model = model or MODEL_ROUTER["default"]
        prov = _provider(model)

        if prov != "ollama":
            chat_model = self._oai_chat(model, prov)
            return chat_model.bind_tools(tools) if tools else chat_model

        primary = self._ollama_chat(model)
        if tools:
            primary = primary.bind_tools(tools)

        if fallback_model and _is_cloud(fallback_model):
            fb_prov = _provider(fallback_model)
            fallback = self._oai_chat(fallback_model, fb_prov)
            if tools:
                fallback = fallback.bind_tools(tools)
            return primary.with_fallbacks([fallback])

        return primary

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
        lc_messages = _to_lc_messages(messages)
        chat_model = self._ollama_chat(model)
        if tools:
            chat_model = chat_model.bind_tools(tools)
        resp = chat_model.invoke(lc_messages)
        return resp.content or "", _from_lc_tool_calls(resp)

    def _chat_openai(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        provider: str,
    ) -> tuple[str, list]:
        lc_messages = _to_lc_messages(messages)
        chat_model = self._oai_chat(model, provider)
        if tools:
            chat_model = chat_model.bind_tools(tools)
        resp = chat_model.invoke(lc_messages)
        return resp.content or "", _from_lc_tool_calls(resp)

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
        lc_messages = _to_lc_messages(messages)
        for chunk in self._ollama_chat(model).stream(lc_messages):
            delta = chunk.content or ""
            if delta:
                yield delta

    def _stream_openai(
        self, messages: list[dict], model: str, provider: str
    ) -> Generator[str, None, None]:
        lc_messages = _to_lc_messages(messages)
        for chunk in self._oai_chat(model, provider).stream(lc_messages):
            delta = chunk.content or ""
            if delta:
                yield delta

    # ── embed ─────────────────────────────────────────────────────────────────

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Embeddings always use the local Ollama model."""
        model = model or MODEL_ROUTER["embed"]
        t0 = time.perf_counter()
        success = False
        try:
            vector = self._embedder(model).embed_query(text)
            success = True
            latency = (time.perf_counter() - t0) * 1000
            logger.debug(f"embed  {model}  {latency:.0f}ms  chars={len(text)}")
            return vector
        except Exception as exc:
            logger.error(f"embed error: {exc}")
            return []
        finally:
            _log_call(model, "embed", (time.perf_counter() - t0) * 1000, success, len(text))
