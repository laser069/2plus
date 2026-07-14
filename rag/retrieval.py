import chromadb
from loguru import logger

from config.settings import CHROMA_PERSIST, TOP_K_RETRIEVAL, RAG_CHUNK_MAX, MODEL_ROUTER
from serving.llm_client import LLMClient

_llm = LLMClient()


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST)
    return client.get_or_create_collection(
        name="2plus_docs",
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    budget_chars: int = RAG_CHUNK_MAX * TOP_K_RETRIEVAL,
) -> list[dict]:
    """Return top-k chunks relevant to query, each text truncated to RAG_CHUNK_MAX chars.
    Total returned chars stay within budget_chars."""
    emb = _llm.embed(query, model=MODEL_ROUTER["embed"])
    if not emb:
        logger.warning("empty embedding for retrieval query")
        return []

    col = _get_collection()
    try:
        results = col.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error(f"chroma query error: {exc}")
        return []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    chunks, chars_used = [], 0
    for text, meta, dist in zip(docs, metas, dists):
        text = (text or "")[:RAG_CHUNK_MAX]
        if chars_used + len(text) > budget_chars:
            break
        chunks.append({
            "text": text,
            "doc_id": meta.get("doc_id", ""),
            "score": round(1 - dist, 4),   # cosine similarity
            "metadata": meta,
        })
        chars_used += len(text)

    return chunks
