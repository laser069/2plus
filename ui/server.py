from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config.settings import MODEL_ROUTER, OPENROUTER_API_KEY
from memory.chat_history import (
    delete_session,
    list_sessions,
    load_session,
    save_message,
)
from memory.convo import ConvoMemory
import memory.user_facts as uf
from orchestrator.agent import Agent, _llm
from rag.ingestion import ingest, list_docs

app = FastAPI(title="2Plus")


@app.on_event("startup")
async def _warm_default_model() -> None:
    """Preload the default chat model into VRAM before accepting traffic, so
    the first real user request doesn't race the warm-up call and queue
    behind it on Ollama's single-model-load serialization."""
    await asyncio.get_running_loop().run_in_executor(None, _llm.warm)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# session_id → (ConvoMemory, Agent)
_sessions: dict[str, tuple[ConvoMemory, Agent]] = {}


def _get_or_create_session(session_id: str) -> tuple[ConvoMemory, Agent]:
    if session_id not in _sessions:
        convo = ConvoMemory()
        for msg in load_session(session_id):
            convo.add_turn(msg["role"], msg["content"])
        agent = Agent(convo)
        _sessions[session_id] = (convo, agent)
    return _sessions[session_id]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/sessions")
async def get_sessions():
    return list_sessions()


class NewSession(BaseModel):
    session_id: str | None = None


@app.post("/sessions")
async def create_session(body: NewSession | None = None):
    sid = (body.session_id if body and body.session_id else None) or str(uuid.uuid4())[:8]
    _get_or_create_session(sid)
    return {"session_id": sid}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    delete_session(session_id)
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    return load_session(session_id)


@app.get("/models")
async def get_models():
    try:
        import ollama as _ollama
        models = [m.model for m in _ollama.list().models]
    except Exception:
        models = [MODEL_ROUTER["default"]]
    return {"models": models, "default": MODEL_ROUTER["default"]}


@app.get("/docs")
async def get_docs():
    return {"docs": list_docs()}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or "upload"
    if filename.endswith(".pdf"):
        text = _pdf_to_text(content)
    else:
        text = content.decode("utf-8", errors="replace")
    n = await asyncio.get_running_loop().run_in_executor(
        None, lambda: ingest(text, doc_id=filename, metadata={"filename": filename})
    )
    return {"filename": filename, "chunks": n}


@app.get("/facts")
async def get_facts():
    facts = {k: v for k, v in uf.get_all().items() if k != "convo_summary"}
    return facts


# ── SSE Chat Stream ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    query: str
    model: str | None = None
    think: bool = False
    fallback_model: str | None = None


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return EventSourceResponse(_stream(req))


async def _stream(req: ChatRequest) -> AsyncGenerator[dict, None]:
    _, agent = _get_or_create_session(req.session_id)

    model = req.model or MODEL_ROUTER["default"]
    fallback = req.fallback_model or None

    save_message(req.session_id, "user", req.query)

    full_answer: list[str] = []
    citations: list[str] = []
    meta: dict = {}

    try:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_agent():
            try:
                for event in agent.run_stream(req.query, model=model, think=req.think, fallback_model=fallback):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        _fut = loop.run_in_executor(None, _run_agent)  # noqa: F841 — kept to prevent GC

        while True:
            event = await queue.get()
            if event is None:
                break
            if isinstance(event, str):
                full_answer.append(event)
                yield {"data": json.dumps({"type": "token", "text": event})}
            elif isinstance(event, dict):
                if event.get("type") == "done":
                    citations = event.get("citations", [])
                    meta = {"routes": event.get("routes", []), "steps": event.get("steps", 0)}
                yield {"data": json.dumps(event)}

    except Exception as exc:
        yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    answer = "".join(full_answer).strip()
    if answer:
        save_message(req.session_id, "assistant", answer, citations=citations, meta=meta)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pdf_to_text(content: bytes) -> str:
    try:
        import importlib
        import io
        pypdf = importlib.import_module("pypdf")
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return content.decode("utf-8", errors="replace")
