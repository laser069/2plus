from dataclasses import dataclass
from typing import Callable, Any
import json

from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

from tools.browser import search_web, fetch_page
from tools.rag_tool import search_docs
import memory.user_facts as uf


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict        # JSON Schema for the function parameters
    handler: Callable
    routes: list[str]       # which router tags activate this tool


def _ollama_tool_schema(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


TOOL_REGISTRY: dict[str, Tool] = {
    "search_web": Tool(
        name="search_web",
        description=(
            "Search the web using DuckDuckGo. Use when the question requires current "
            "information or facts not in uploaded documents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=lambda query, max_results=5: json.dumps(
            search_web(query, max_results), ensure_ascii=False
        ),
        routes=["use_browser"],
    ),
    "fetch_page": Tool(
        name="fetch_page",
        description=(
            "Fetch and extract the text content of a web page. Use after search_web "
            "when a snippet is not enough and you need the full page."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"},
            },
            "required": ["url"],
        },
        handler=lambda url: fetch_page(url),
        routes=["use_browser"],
    ),
    "search_docs": Tool(
        name="search_docs",
        description=(
            "Search user-uploaded documents. Use when the question may be answered "
            "by uploaded files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 4)",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
        handler=lambda query, top_k=4: json.dumps(
            search_docs(query, top_k), ensure_ascii=False
        ),
        routes=["use_rag"],
    ),
    "update_memory": Tool(
        name="update_memory",
        description=(
            "Save or update a fact about the user. Use when the user shares personal "
            "information, preferences, or ongoing tasks you should remember."
        ),
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short identifier, e.g. 'name'"},
                "value": {"type": "string", "description": "The fact to store"},
            },
            "required": ["key", "value"],
        },
        handler=lambda key, value: (uf.upsert(key, value), f"Saved: {key}={value}")[1],
        routes=["use_memory"],
    ),
    "recall_memory": Tool(
        name="recall_memory",
        description="Retrieve all stored facts about the user.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=lambda: json.dumps(uf.get_all(), ensure_ascii=False),
        routes=["use_memory"],
    ),
}


def tools_for_routes(routes: list[str]) -> tuple[list[Tool], list[dict]]:
    """Return (tool_list, ollama_schema_list) for the given route tags."""
    active = [t for t in TOOL_REGISTRY.values() if any(r in routes for r in t.routes)]
    schemas = [_ollama_tool_schema(t) for t in active]
    return active, schemas


_JSON_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def _args_schema(tool: Tool):
    """Build a pydantic args model from a Tool's JSON-schema parameters, for
    binding as a LangChain StructuredTool."""
    props = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))
    fields = {}
    for name, spec in props.items():
        py_type = _JSON_TYPE_MAP.get(spec.get("type"), str)
        description = spec.get("description", "")
        if name in required:
            fields[name] = (py_type, Field(..., description=description))
        else:
            fields[name] = (py_type, Field(spec.get("default"), description=description))
    return create_model(f"{tool.name}_Args", **fields)


def _to_structured_tool(tool: Tool) -> StructuredTool:
    return StructuredTool.from_function(
        func=tool.handler,
        name=tool.name,
        description=tool.description,
        args_schema=_args_schema(tool),
    )


def lc_tools_for_routes(routes: list[str]) -> list[StructuredTool]:
    """Return LangChain StructuredTool objects for the given route tags, for
    binding via chat_model.bind_tools() in the ReAct agent loop."""
    active = [t for t in TOOL_REGISTRY.values() if any(r in routes for r in t.routes)]
    return [_to_structured_tool(t) for t in active]
