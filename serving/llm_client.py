import time
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ollama
from loguru import logger

from config.settings import MODEL_ROUTER, OLLAMA_BASE_URL, LOG_DIR

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
_call_log = Path(LOG_DIR) / "calls.jsonl"

logger.add(str(Path(LOG_DIR) / "2plus.log"), rotation="10 MB", retention=3)


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[Any] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0


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


class LLMClient:
    def __init__(self):
        self._client = ollama.Client(host=OLLAMA_BASE_URL)

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        think: bool = False,
    ) -> ChatResponse:
        model = model or MODEL_ROUTER["default"]

        # Prepend /no_think to first user message (Qwen3 reads it from user turn)
        if not think:
            msgs = list(messages)
            for i, m in enumerate(msgs):
                if m["role"] == "user" and "/no_think" not in (m.get("content") or ""):
                    msgs[i] = {**m, "content": "/no_think\n" + (m["content"] or "")}
                    break
            messages = msgs

        chars_in = sum(len(m.get("content") or "") for m in messages)
        t0 = time.perf_counter()
        success = False
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            resp = self._client.chat(**kwargs)
            success = True
            msg = resp.message
            return ChatResponse(
                content=msg.content or "",
                tool_calls=msg.tool_calls or [],
                model=model,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            logger.error(f"chat error: {exc}")
            return ChatResponse(content="", model=model)
        finally:
            _log_call(model, "chat", (time.perf_counter() - t0) * 1000, success, chars_in)

    def embed(self, text: str, model: str | None = None) -> list[float]:
        model = model or MODEL_ROUTER["embed"]
        t0 = time.perf_counter()
        success = False
        try:
            resp = self._client.embeddings(model=model, prompt=text)
            success = True
            return resp["embedding"]
        except Exception as exc:
            logger.error(f"embed error: {exc}")
            return []
        finally:
            _log_call(model, "embed", (time.perf_counter() - t0) * 1000, success, len(text))
