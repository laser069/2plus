from rag.retrieval import retrieve
from config.settings import TOP_K_RETRIEVAL, RAG_CHUNK_MAX


def search_docs(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search uploaded documents. Returns list of {text, doc_id, score}."""
    budget = RAG_CHUNK_MAX * top_k
    return retrieve(query, top_k=top_k, budget_chars=budget)
