# 2Plus — Local AI Assistant

A fully local, privacy-first AI assistant that combines retrieval-augmented generation (RAG) over your own documents, live web browsing, and persistent user memory — all running on your machine with no cloud dependencies or API keys required.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [LangChain Integration](#langchain-integration)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Running 2Plus](#running-2plus)
8. [OpenRouter Integration](#openrouter-integration)
9. [Project Structure](#project-structure)
10. [Module Reference](#module-reference)
11. [Configuration](#configuration)
12. [How It Works](#how-it-works)
13. [Context Budget System](#context-budget-system)
14. [Git Branch Strategy](#git-branch-strategy)
15. [Smoke Tests](#smoke-tests)
16. [Troubleshooting](#troubleshooting)
17. [Roadmap](#roadmap)

---

## Overview

2Plus is built around a **ReAct (Reason + Act)** agent loop that intelligently decides when to search your documents, browse the web, recall stored facts, or answer directly from its own knowledge. It runs primarily on [Ollama](https://ollama.com/) for fully local LLM inference, with optional cloud routing to any model on [OpenRouter](https://openrouter.ai/) when you need stronger reasoning or larger context windows. Chat and embedding calls, the ReAct tool-calling loop, RAG storage, and conversation windowing all run on [LangChain](https://python.langchain.com/) primitives (see [LangChain Integration](#langchain-integration)); routing uses a fast keyword heuristic with an optional cached LLM fallback for ambiguous queries, and Ollama keeps the active model resident in VRAM between calls (`OLLAMA_KEEP_ALIVE`) to avoid cold-load thrash when switching models.

**Primary model:** `qwen3:8b`  
**Embeddings model:** `all-minilm:l6-v2`  
**Vector database:** ChromaDB (local persistent)  
**Fact store:** SQLite  
**Web search:** DuckDuckGo (no API key)  
**LLM layer:** LangChain (`langchain-ollama`, `langchain-openai`)  
**UI:** Streamlit

---

## Features

| Feature | Details |
|---------|---------|
| **Document Q&A (RAG)** | Upload PDF, TXT, or Markdown files and ask questions about their contents. Answers include citations pointing to the source document. |
| **Web browsing** | Searches DuckDuckGo and fetches page content using trafilatura. No API key required. |
| **Persistent memory** | Remembers facts about you (name, preferences, ongoing projects) across sessions using SQLite. Updated facts overwrite stale ones automatically. |
| **Context-aware routing** | A fast router model classifies each query and selects only the tools that are actually needed — avoiding unnecessary tool calls. |
| **Budget-capped context** | A strict character budget system prevents context overflow at 8B model scale. Memory, summaries, and RAG chunks are all size-limited before injection. |
| **Rolling conversation summary** | When conversation history grows long, older turns are compressed into a summary paragraph rather than dropped entirely. |
| **Structured logging** | Every model call is logged to `logs/calls.jsonl` with model name, latency, character count, and success flag. |
| **100% free** | No paid APIs, no cloud accounts, no telemetry. Everything runs on your local machine. |

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    Router                           │
│  (qwen3.5:4b — fast classification)                 │
│  → ["use_rag", "use_browser", "use_memory", "direct"]│
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Context Builder                        │
│  System prompt + rolling summary + relevant facts   │
│  (budget-capped: 6000 chars total)                  │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           ReAct Agent Loop (max 5 steps)            │
│                                                     │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────┐    │
│  │ search_ │  │ fetch_   │  │ search_docs     │    │
│  │ web     │  │ page     │  │ (ChromaDB RAG)  │    │
│  └─────────┘  └──────────┘  └─────────────────┘    │
│                                                     │
│  ┌──────────────┐  ┌──────────────────────────┐    │
│  │ update_      │  │ recall_memory            │    │
│  │ memory       │  │ (SQLite user facts)      │    │
│  └──────────────┘  └──────────────────────────┘    │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                   Answer + Citations                │
│  Post-run: extract new user facts → SQLite upsert   │
│            add turn to convo memory                 │
└─────────────────────────────────────────────────────┘
```

---

## LangChain Integration

2Plus uses [LangChain](https://python.langchain.com/) across the LLM-calling layer, the ReAct agent loop, RAG storage, and conversation memory — while keeping its own hand-rolled router, prompts, SSE streaming contract, and step/budget control flow (LangChain is used for its primitives, not as a full agent framework like LangGraph's `AgentExecutor`).

**What LangChain is used for:**

| Concern | LangChain class | Notes |
|---------|-----------------|-------|
| Local chat (Ollama) | `langchain_ollama.ChatOllama` | `serving/llm_client.py` |
| Cloud chat (OpenRouter) | `langchain_openai.ChatOpenAI` | Points at OpenRouter via `base_url` override |
| Local embeddings | `langchain_ollama.OllamaEmbeddings` | `serving/llm_client.py`, `rag/ingestion.py`, `rag/retrieval.py` |
| ReAct tool-calling loop | `BaseMessage`/`AIMessage`/`ToolMessage`, `.bind_tools()` | `orchestrator/agent.py` operates on LangChain messages natively — no dict↔message conversion at this layer |
| Ollama→cloud fallback | `Runnable.with_fallbacks()` | `LLMClient.get_chat_model()` binds tools then wraps with a native fallback runnable, replacing a manual try/except |
| Tool schemas | `langchain_core.tools.StructuredTool` | `tools/registry.py` builds a pydantic `args_schema` per tool from its JSON-schema `parameters` |
| RAG vector store | `langchain_chroma.Chroma` | `rag/ingestion.py`, `rag/retrieval.py` — replaces the raw `chromadb.PersistentClient` |
| Document chunking | `langchain_text_splitters.RecursiveCharacterTextSplitter` | Replaces fixed-size manual chunking |
| Conversation window | `langchain_core.messages.trim_messages` | `memory/convo.py` — `token_counter=len` makes it a message-count window (matches `CONVO_WINDOW`) instead of a token budget |

**What stays hand-rolled** (no clear win from a framework rewrite):

- The **router** (`orchestrator/router.py`) — a single prompt + regex classification.
- The **ReAct loop's control flow** (`orchestrator/agent.py`) — step limits, character-budget tracking, SSE event yielding, and citation extraction are custom, even though the messages and tool calls flowing through it are now LangChain-native.
- **User facts** (`memory/user_facts.py`) and **chat history persistence** (`memory/chat_history.py`) — raw SQLite key/value and message-log tables; there's no LangChain abstraction that fits these better than direct SQL.
- The **rolling conversation summary** itself (`memory/convo.py`) — still a custom LLM-summarization prompt, LangChain's `trim_messages` only decides *which* messages get folded into it.

Custom behaviors preserved throughout: `/no_think` prompt injection for Qwen3 (`inject_no_think_lc`), structured call logging to `logs/calls.jsonl`, and the exact SSE event contract (`routing`/`tool`/`token`/`done`) the web UI depends on.

---

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running: https://ollama.com/download
- **NVIDIA GPU** (recommended, <8GB VRAM works with quantized models)
- The following Ollama models pulled:

```bash
ollama pull qwen3:8b
ollama pull qwen3.5:4b
ollama pull all-minilm:l6-v2
```

---

## Installation

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd 2Plus
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env if your Ollama runs on a non-default port
```

### 5. Verify Ollama is running

```bash
ollama list
# Should show qwen3:8b, qwen3.5:4b, all-minilm:l6-v2
```

---

## Running 2Plus

### Web UI (recommended)

FastAPI backend with SSE streaming (`ui/server.py`) + a responsive vanilla JS/HTML/CSS frontend (`ui/static/`). Chats persist to SQLite and reload across server restarts; sessions can be created, switched, and deleted from the sidebar (a collapsible off-canvas drawer below 768px).

```bash
uvicorn ui.server:app --reload --port 8000
```

Open your browser at `http://localhost:8000`.

### Streamlit UI (legacy fallback)

```bash
streamlit run ui/app.py
```

Open your browser at `http://localhost:8501`. Kept for parity/fallback; the web UI above is the primary, actively developed interface.

### CLI Chat Mode

```bash
python main.py --chat
```

Interactive terminal chat loop. Type `quit` to exit.

### Smoke Tests

```bash
python main.py --test-llm      # Tests chat and embedding endpoints
python main.py --test-rag      # Tests document ingestion and retrieval
python main.py --test-browser  # Tests DuckDuckGo search and page fetch
```

---

## OpenRouter Integration

2Plus supports routing individual conversations to cloud models via [OpenRouter](https://openrouter.ai/), giving you access to models like Claude, GPT-4o, Gemini, and DeepSeek without replacing the local Ollama setup. Ollama remains the default for all inference; OpenRouter is purely opt-in per conversation.

### Prerequisites

1. Create a free account at [openrouter.ai](https://openrouter.ai/)
2. Generate an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
3. Add credits to your OpenRouter account (pay-as-you-go, no subscription required)

### Setup

Add your API key to the `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
```

No other configuration is needed. The key is loaded automatically on startup.

### Using OpenRouter in the UI

In the Streamlit sidebar, you will find a **Cloud · OpenRouter** section beneath the Ollama model selector. Paste any valid OpenRouter model identifier into the text field:

```
anthropic/claude-3.5-sonnet
openai/gpt-4o
google/gemini-2.0-flash-001
deepseek/deepseek-r1
meta-llama/llama-3.3-70b-instruct
```

The full list of available models and their IDs can be found at [openrouter.ai/models](https://openrouter.ai/models).

**How the override works:**
- When the OpenRouter model field is filled in, it overrides the Ollama model selector for that conversation.
- The active model badge in the top-right of the chat area turns purple to indicate a cloud model is in use.
- Clearing the field immediately reverts back to the Ollama model selected in the dropdown.

### Full Tool Use with Cloud Models

OpenRouter models participate in the full ReAct agent loop — the same reasoning cycle used with local Ollama models. This means cloud models can invoke all five tools:

| Tool | What it does |
|------|-------------|
| `search_web` | DuckDuckGo search |
| `fetch_page` | Extracts text content from a URL |
| `search_docs` | Semantic search over your uploaded documents |
| `update_memory` | Stores a user fact persistently |
| `recall_memory` | Retrieves all stored user facts |

Tool call format is automatically translated between Ollama's native format and the OpenAI-compatible format required by OpenRouter. No changes to prompts or tool definitions are needed.

### What stays local

Even when an OpenRouter model is selected, the following always use local Ollama:

- **Text embeddings** (`all-minilm:l6-v2`) — used for RAG document indexing and retrieval
- **Query routing** (`qwen3.5:4b`) — the fast classifier that picks which tools to activate
- **Conversation summarisation** — compresses older turns to keep context short
- **Fact extraction** — parses new user facts from assistant responses

This design keeps your documents and facts private while optionally using cloud models for the main reasoning step.

### Cost considerations

OpenRouter charges per token on a pay-as-you-go basis. The context budget system (6 000-character default) naturally limits token usage. You can monitor your spending at [openrouter.ai/activity](https://openrouter.ai/activity).

### Troubleshooting OpenRouter

**"OPENROUTER_API_KEY missing in .env" warning in the sidebar**  
The model name field is filled in but the key is not set. Add `OPENROUTER_API_KEY=...` to your `.env` file and restart the Streamlit server.

**`AuthenticationError` in the logs**  
The API key is present but invalid or expired. Generate a new key at [openrouter.ai/keys](https://openrouter.ai/keys).

**`RateLimitError` or `InsufficientCreditsError`**  
Your OpenRouter account balance is depleted. Top up credits at [openrouter.ai/credits](https://openrouter.ai/credits).

**Model returns no tool calls (ReAct loop ends immediately)**  
Some smaller or older models on OpenRouter do not reliably follow function-calling instructions. Switch to a model with strong tool-use capability such as `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, or `google/gemini-2.0-flash-001`.

---

## Project Structure

```
2Plus/
├── config/
│   ├── __init__.py
│   └── settings.py          # All constants: model names, paths, budget limits
│
├── serving/
│   ├── __init__.py
│   └── llm_client.py        # LLMClient: thin Ollama wrapper for chat + embeddings
│
├── memory/
│   ├── __init__.py
│   ├── user_facts.py        # Persistent SQLite key-value store for user facts
│   └── convo.py             # Rolling conversation memory with summary compression
│
├── rag/
│   ├── __init__.py
│   ├── ingestion.py         # Document ingestion: chunk → embed → store in ChromaDB
│   └── retrieval.py         # Semantic retrieval from ChromaDB
│
├── tools/
│   ├── __init__.py
│   ├── browser.py           # DuckDuckGo search + trafilatura page extraction
│   ├── rag_tool.py          # Thin wrapper exposing RAG retrieval as a tool
│   └── registry.py          # Tool dataclass, TOOL_REGISTRY, Ollama schema builder
│
├── orchestrator/
│   ├── __init__.py
│   ├── prompts.py           # System prompt, router prompt, fact extraction prompt
│   ├── router.py            # LLM-based query classifier → route tags
│   └── agent.py             # ReAct loop: routes → context → tools → answer
│
├── ui/
│   ├── app.py               # Streamlit chat UI (legacy fallback)
│   ├── server.py            # FastAPI backend: sessions, SSE chat streaming, uploads
│   └── static/              # Vanilla JS/HTML/CSS web frontend (primary UI)
│       ├── index.html
│       ├── app.js
│       └── style.css
│
├── data/                    # Auto-created: ChromaDB files + SQLite DB
├── logs/                    # Auto-created: calls.jsonl + 2plus.log
│
├── main.py                  # Entry point: smoke tests + CLI chat
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Module Reference

### `config/settings.py`

Central configuration file. All tuneable constants live here.

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_ROUTER["default"]` | `qwen3:8b` | Primary reasoning model |
| `MODEL_ROUTER["fast"]` | `qwen3.5:4b` | Fast model used for routing classification |
| `MODEL_ROUTER["embed"]` | `all-minilm:l6-v2` | Embeddings model |
| `MODEL_ROUTER["cloud"]` | `None` | Cloud model seam — wire in later if needed |
| `MAX_REACT_STEPS` | `5` | Maximum tool-call iterations per query |
| `TOP_K_RETRIEVAL` | `4` | Number of document chunks returned per RAG query |
| `CHUNK_SIZE` | `800` | Characters per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `CTX_BUDGET_CHARS` | `6000` | Total character budget injected per prompt |
| `FACTS_MAX_CHARS` | `800` | Max chars for injected user facts block |
| `SUMMARY_MAX_CHARS` | `600` | Max chars for injected conversation summary |
| `RAG_CHUNK_MAX` | `1200` | Max chars per individual RAG chunk |
| `CONVO_WINDOW` | `6` | Verbatim turns kept before older ones are summarised |

---

### `serving/llm_client.py`

**`LLMClient`**

A thin, provider-agnostic wrapper backed by [LangChain](https://python.langchain.com/) chat models (`ChatOllama`, `ChatOpenAI`) and `OllamaEmbeddings`. All model calls go through this class so the underlying provider — and now the underlying LLM framework — can be swapped in one place. See [LangChain Integration](#langchain-integration) for what LangChain does and doesn't touch.

```python
from serving.llm_client import LLMClient

llm = LLMClient()

# Chat
response = llm.chat(
    messages=[{"role": "user", "content": "Hello"}],
    model="qwen3:8b",   # defaults to MODEL_ROUTER["default"]
    tools=None,         # optional OpenAI-function-format tool list, bound via .bind_tools()
    think=False,        # True enables chain-of-thought (Qwen3 thinking mode)
)
print(response.content)
print(response.tool_calls)  # list of tool call objects if any

# Embeddings
vector = llm.embed("Some text to embed")
# Returns list[float] of dimension 384 (all-minilm)
```

**Behaviour notes:**
- `think=False` (default): prepends `/no_think` to the first user message, disabling Qwen3's chain-of-thought mode for faster responses.
- `think=True`: lets Qwen3 reason internally before answering — useful for complex multi-step problems.
- Canonical `{"role": ..., "content": ...}` message dicts are converted to LangChain `BaseMessage` objects internally; `AIMessage.tool_calls` are converted back to the app's own `_ToolCall`/`ChatResponse` shapes, so callers never see LangChain types directly.
- Every call writes a structured log entry to `logs/calls.jsonl`.

---

### `memory/user_facts.py`

SQLite-backed key-value store for persistent user facts. Facts survive across sessions and are updated in-place (upsert semantics).

```python
import memory.user_facts as uf

uf.upsert("name", "Alice")
uf.upsert("timezone", "UTC+5:30")
uf.upsert("preferred_language", "Python")

print(uf.get("name"))         # "Alice"
print(uf.get_all())           # {"name": "Alice", "timezone": "UTC+5:30", ...}

# Only inject facts relevant to the current query (keyword match, budget-capped)
block = uf.get_relevant("what timezone am I in?", budget=800)
print(block)
# "- timezone: UTC+5:30"

uf.delete("name")
```

---

### `memory/convo.py`

Manages the in-process conversation window and a persistent rolling summary.

```python
from memory.convo import ConvoMemory
from serving.llm_client import LLMClient

convo = ConvoMemory()
llm = LLMClient()

convo.add_turn("user", "My name is Alice.")
convo.add_turn("assistant", "Nice to meet you, Alice!")

# When window exceeds CONVO_WINDOW turns, summarise oldest half
convo.maybe_summarise(llm)

# Build prompt-ready message list with budget tracking
messages, chars_used = convo.get_context(
    query="What is my name?",
    system_prompt="You are 2Plus..."
)
# messages includes: [system, summary (if any), facts (if relevant), ...recent turns]
```

**Summary persistence:** The rolling summary is stored in SQLite under key `"convo_summary"` so it survives application restarts.

---

### `rag/ingestion.py`

Ingests documents into the ChromaDB vector store.

```python
from rag.ingestion import ingest, delete_by_doc_id, list_docs

# Ingest from a file path or raw text string
chunk_count = ingest(
    path_or_text="/path/to/document.txt",  # or raw string
    doc_id="my_document",                  # unique identifier
    metadata={"source": "upload", "author": "Alice"},
)
print(f"Stored {chunk_count} chunks")

# Re-uploading the same doc_id cleanly replaces the previous version
chunk_count = ingest("Updated content...", doc_id="my_document")

# Delete all chunks for a document
delete_by_doc_id("my_document")

# List all ingested documents
print(list_docs())  # ["my_document", "report_2024.txt", ...]
```

**Chunking:** Documents are split into 800-character chunks with 100-character overlap. Each chunk is embedded using `all-minilm:l6-v2` via Ollama before being stored in ChromaDB.

---

### `rag/retrieval.py`

Retrieves relevant document chunks for a query using cosine similarity search.

```python
from rag.retrieval import retrieve

results = retrieve(
    query="What are the payment terms?",
    top_k=4,
    budget_chars=4800,  # total chars returned won't exceed this
)

for chunk in results:
    print(chunk["doc_id"])   # which document
    print(chunk["score"])    # cosine similarity (0–1, higher is better)
    print(chunk["text"])     # chunk content (truncated to RAG_CHUNK_MAX)
```

---

### `tools/registry.py`

Central registry of all tools available to the agent. Tools are plain Python callables wrapped in a `Tool` dataclass with a JSON Schema for Ollama's native function calling.

**Available tools:**

| Tool name | Route tag | Description |
|-----------|-----------|-------------|
| `search_web` | `use_browser` | DuckDuckGo search — returns title, URL, snippet |
| `fetch_page` | `use_browser` | Fetches and extracts text content from a URL |
| `search_docs` | `use_rag` | Semantic search over ingested documents |
| `update_memory` | `use_memory` | Stores a user fact in SQLite |
| `recall_memory` | `use_memory` | Returns all stored user facts |

```python
from tools.registry import tools_for_routes

# Get tools and their Ollama schemas for specific routes
active_tools, schemas = tools_for_routes(["use_browser", "use_rag"])
```

---

### `orchestrator/router.py`

Classifies each user query into one or more route tags using the fast model (`qwen3.5:4b`). This determines which tools are available in the subsequent ReAct loop.

```python
from orchestrator.router import classify

tags = classify("What does my contract say about cancellation?")
# → ["use_rag"]

tags = classify("What is the current price of Bitcoin?")
# → ["use_browser"]

tags = classify("Remember that I prefer concise answers")
# → ["use_memory"]

tags = classify("What is 12 * 8?")
# → ["direct"]
```

If the model returns malformed output, the router falls back to `["direct"]` and logs a warning.

---

### `orchestrator/agent.py`

The main agent loop. Orchestrates routing, context building, tool dispatch, and memory updates.

```python
from memory.convo import ConvoMemory
from orchestrator.agent import Agent

convo = ConvoMemory()
agent = Agent(convo)

result = agent.run("What are the key points in the uploaded contract?")

print(result.answer)      # Final answer text
print(result.citations)   # e.g. ["[doc: contract.pdf]", "[web: https://...]"]
print(result.steps)       # Number of ReAct iterations used
print(result.routes)      # e.g. ["use_rag"]
```

**ReAct loop behaviour:**
1. Router classifies the query → selects active tools
2. Context is built with budget tracking (memory + summary + facts)
3. LLM is called with the active tool schemas
4. If a tool call is returned, the tool is dispatched and the observation is appended
5. Steps repeat until the LLM returns a final answer (no tool call), the step cap is reached, or the context budget is exhausted
6. After the answer is generated, new user facts are extracted and saved to SQLite

---

### `ui/app.py`

The Streamlit-based chat interface.

**Sidebar features:**
- **Document upload:** Upload PDF, TXT, or Markdown files. Each file is chunked, embedded, and stored in ChromaDB automatically.
- **Ingested docs list:** Shows all document IDs currently stored in the vector database.
- **Memory viewer:** Displays all stored user facts (expandable panel).
- **Clear conversation:** Resets the conversation window and deletes the rolling summary.

**Chat area features:**
- Displays the full conversation history
- Each assistant message shows a collapsible **Sources** panel listing citations
- A collapsible **Debug** panel shows the route tags and step count for each response

---

## Configuration

All configuration is in `config/settings.py`. The most commonly adjusted values are:

**Switching to a different Ollama model:**
```python
MODEL_ROUTER = {
    "default": "llama3.1:8b",   # swap qwen3:8b for any Ollama model
    ...
}
```

**Adjusting context budget (if you have more VRAM):**
```python
CTX_BUDGET_CHARS = 12000   # increase for larger context windows
RAG_CHUNK_MAX = 2000        # allow larger individual chunks
```

**Adjusting conversation memory:**
```python
CONVO_WINDOW = 10           # keep more verbatim turns before summarising
SUMMARY_MAX_CHARS = 1000    # allow longer summaries
```

**Environment variables (`.env` file):**
```
OLLAMA_BASE_URL=http://localhost:11434

# Optional — enables cloud model routing via OpenRouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## How It Works

### Document Upload Flow

```
User uploads file
       │
       ▼
Read file content
       │
       ▼
Split into 800-char chunks (100-char overlap)
       │
       ▼
Embed each chunk with all-minilm:l6-v2 (384 dimensions)
       │
       ▼
Store chunks + embeddings in ChromaDB ("2plus_docs" collection)
       │
       ▼
Chunks are available for semantic search immediately
```

### Query Flow

```
User types query
       │
       ▼
Router (qwen3.5:4b) classifies into route tags
       │
       ▼
ConvoMemory builds message list:
  [system prompt]
  [rolling summary — max 600 chars]
  [relevant user facts — max 800 chars]
  [last 6 verbatim turns]
       │
       ▼
ReAct loop (max 5 steps):
  LLM chooses tool or generates answer
  Tool result truncated to remaining budget
  Repeat until final answer or budget/step limit
       │
       ▼
Answer returned with citations
       │
       ▼
Post-processing:
  New facts extracted from answer → SQLite upsert
  Turn added to conversation window
  Oldest turns compressed to summary if window full
```

---

## Context Budget System

One of 2Plus's core design constraints is operating reliably with an 8B parameter model. Larger models degrade significantly when context becomes long and unfocused. The budget system enforces strict limits:

```
Total budget: 6000 characters
├── System prompt:      ~400 chars (fixed)
├── Rolling summary:    max 600 chars
├── User facts block:   max 800 chars (only relevant facts selected)
├── Recent turns:       ~1500 chars (last 6 turns × ~250 chars avg)
└── RAG / web results: remainder (~2700 chars, split across tool observations)
```

Each tool observation is truncated before being appended to the message list, and the agent stops early if the budget is exhausted. This prevents the model from receiving noisy, unfocused context that degrades answer quality.

---

## Git Branch Strategy

Development is organized into one branch per phase:

| Branch | Contents |
|--------|----------|
| `main` | Stable base |
| `feat/phase0-foundation` | Config, LLMClient, requirements |
| `feat/phase1-memory` | SQLite user facts, ConvoMemory |
| `feat/phase2-rag` | ChromaDB ingestion and retrieval |
| `feat/phase3-browser` | DuckDuckGo search, trafilatura fetch, tool registry |
| `feat/phase4-orchestrator` | Prompts, router, ReAct agent |
| `feat/phase5-ui` | Streamlit UI, CLI entry point, bug fixes |

---

## Smoke Tests

Run these to verify your installation is working correctly:

```bash
# 1. Verify Ollama chat and embeddings work
python main.py --test-llm

# 2. Verify ChromaDB ingestion and retrieval
python main.py --test-rag

# 3. Verify DuckDuckGo search and page fetch
python main.py --test-browser
```

Expected output for a passing run:

```
Testing chat (qwen3:8b)...
  response: 'OK'  latency=~80000ms
Testing embed (all-minilm:l6-v2)...
  embedding dims: 384
LLM smoke test PASSED

Ingesting test doc...
  chunks ingested: 2
Retrieving...
  top result score=0.47
RAG smoke test PASSED

Testing DuckDuckGo search...
  results: 3
  first title: Welcome to Python.org
Fetching https://www.python.org/...
  extracted chars: 1173
Browser smoke test PASSED
```

> **Note:** The first LLM call may take 60–90 seconds on a cold Ollama instance while the model loads into VRAM. Subsequent calls are significantly faster.

---

## Troubleshooting

**Ollama not responding**
```bash
# Check Ollama is running
ollama list
# If not: start it
ollama serve
```

**Model not found**
```bash
ollama pull qwen3:8b
ollama pull qwen3.5:4b
ollama pull all-minilm:l6-v2
```

**ChromaDB errors on startup**  
Delete the data directory and let it recreate:
```bash
rm -rf data/chroma
```

**`ddgs` import error**  
The DuckDuckGo package was recently renamed. Reinstall:
```bash
pip uninstall duckduckgo-search -y
pip install ddgs
```

**Slow responses**  
- First query after Ollama starts is slow (model loading). Subsequent queries are faster.
- If consistently slow, try the smaller `qwen3.5:4b` model by setting `MODEL_ROUTER["default"] = "qwen3.5:4b"` in `config/settings.py`.

**Out of memory errors**  
- Ensure no other large models are loaded in Ollama: `ollama ps`
- Use a more aggressively quantized model: `ollama pull qwen3:8b-q4_K_M`

---

## Roadmap

- [x] Streaming responses (SSE) via the FastAPI + web UI stack
- [x] Responsive web UI (off-canvas sidebar drawer below 768px)
- [x] Server-restart-durable sessions (ConvoMemory rehydrated from SQLite on cache miss)
- [ ] Reranker support (`bge-reranker`) for improved RAG quality
- [x] Cloud model routing via OpenRouter (opt-in per conversation)
- [x] LangChain-backed LLM calling layer (`ChatOllama`/`ChatOpenAI`/`OllamaEmbeddings`)
- [x] LangChain-native ReAct tool-calling loop, RAG vector store, and conversation window
- [x] Heuristic query router with cached LLM fallback for ambiguous queries
- [x] Model-swap thrash elimination (`OLLAMA_KEEP_ALIVE`, model warm-on-select, parallel tool dispatch)
- [ ] Vector-based chat history recall for fuzzy "what did we discuss" queries
- [ ] Playwright fallback for JavaScript-heavy pages
- [ ] Multi-user session support (auth; `_sessions` cache is currently single-process/no-auth)
- [ ] Langfuse integration for production observability
