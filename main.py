"""
2Plus — entry point.

Usage:
  uvicorn ui.server:app --reload   # launch the web UI (recommended)
  streamlit run ui/app.py          # launch Streamlit UI (legacy fallback)
  python main.py --test-llm        # smoke test LLM + embeddings
  python main.py --test-rag        # smoke test RAG ingestion + retrieval
  python main.py --test-browser    # smoke test DuckDuckGo search + fetch
  python main.py --chat            # minimal CLI chat loop
"""

import sys
from pathlib import Path

from config.logging_config import setup_logging
setup_logging()


def _test_llm() -> None:
    from serving.llm_client import LLMClient
    llm = LLMClient()
    print("Testing chat (qwen3:8b)…")
    r = llm.chat([{"role": "user", "content": "Reply with: OK"}])
    print(f"  response: {r.content!r}  latency={r.latency_ms:.0f}ms")
    assert r.content, "empty chat response"

    print("Testing embed (all-minilm:l6-v2)…")
    emb = llm.embed("hello world")
    print(f"  embedding dims: {len(emb)}")
    assert len(emb) > 0, "empty embedding"
    print("LLM smoke test PASSED")


def _test_rag() -> None:
    from rag.ingestion import ingest, delete_by_doc_id
    from rag.retrieval import retrieve
    doc_id = "__test_doc__"
    sample = "The quick brown fox jumps over the lazy dog. " * 20
    print("Ingesting test doc…")
    n = ingest(sample, doc_id=doc_id)
    print(f"  chunks ingested: {n}")
    assert n > 0

    print("Retrieving…")
    results = retrieve("quick brown fox")
    print(f"  top result score={results[0]['score'] if results else 'N/A'}")
    assert results, "no results returned"

    delete_by_doc_id(doc_id)
    print("RAG smoke test PASSED")


def _test_browser() -> None:
    from tools.browser import search_web, fetch_page
    print("Testing DuckDuckGo search…")
    results = search_web("Python programming language", max_results=3)
    print(f"  results: {len(results)}")
    assert results, "no search results"
    print(f"  first title: {results[0]['title']}")

    if results:
        url = results[0]["url"]
        print(f"Fetching {url}…")
        text = fetch_page(url)
        print(f"  extracted chars: {len(text)}")
    print("Browser smoke test PASSED")


def _cli_chat() -> None:
    from memory.convo import ConvoMemory
    from orchestrator.agent import Agent
    convo = ConvoMemory()
    agent = Agent(convo)
    print("2Plus CLI — type 'quit' to exit\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        result = agent.run(query)
        print(f"\n2Plus: {result.answer}")
        if result.citations:
            print(f"Sources: {', '.join(result.citations)}")
        print(f"[routes={result.routes} steps={result.steps}]\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--chat"
    dispatch = {
        "--test-llm": _test_llm,
        "--test-rag": _test_rag,
        "--test-browser": _test_browser,
        "--chat": _cli_chat,
    }
    fn = dispatch.get(arg)
    if fn:
        fn()
    else:
        print(__doc__)
