from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from config.settings import (
    CHROMA_PERSIST,
    TOP_K_RETRIEVAL,
    RAG_CHUNK_MAX,
    MODEL_ROUTER,
    OLLAMA_BASE_URL,
)


def _get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=MODEL_ROUTER["embed"], base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name="2plus_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST,
        collection_metadata={"hnsw:space": "cosine"},
    )


def retrieve(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    budget_chars: int = RAG_CHUNK_MAX * TOP_K_RETRIEVAL,
) -> list[dict]:
    """Return top-k chunks relevant to query, each text truncated to RAG_CHUNK_MAX chars.
    Total returned chars stay within budget_chars."""
    try:
        results = _get_vectorstore().similarity_search_with_score(query, k=top_k)
    except Exception as exc:
        logger.error(f"chroma query error: {exc}")
        return []

    chunks, chars_used = [], 0
    for doc, dist in results:
        text = (doc.page_content or "")[:RAG_CHUNK_MAX]
        if chars_used + len(text) > budget_chars:
            break
        chunks.append({
            "text": text,
            "doc_id": doc.metadata.get("doc_id", ""),
            "score": round(1 - dist, 4),   # cosine similarity
            "metadata": doc.metadata,
        })
        chars_used += len(text)

    return chunks
