from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import (
    CHROMA_PERSIST,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MODEL_ROUTER,
    OLLAMA_BASE_URL,
)

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=MODEL_ROUTER["embed"], base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name="2plus_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _load_text(path_or_text: str) -> str:
    p = Path(path_or_text)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    return path_or_text


def delete_by_doc_id(doc_id: str) -> None:
    store = _get_vectorstore()
    results = store._collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        store.delete(ids=results["ids"])
        logger.info(f"deleted {len(results['ids'])} chunks for doc_id={doc_id}")


def ingest(path_or_text: str, doc_id: str, metadata: dict | None = None) -> int:
    """Ingest a document. Re-upload replaces previous version cleanly."""
    metadata = metadata or {}
    delete_by_doc_id(doc_id)

    text = _load_text(path_or_text)
    chunks = [c for c in _splitter.split_text(text) if c.strip()]
    if not chunks:
        return 0

    store = _get_vectorstore()
    ids = [f"{doc_id}::chunk{i}" for i in range(len(chunks))]
    metas = [{"doc_id": doc_id, "chunk_idx": i, **metadata} for i in range(len(chunks))]

    store.add_texts(texts=chunks, metadatas=metas, ids=ids)
    logger.info(f"ingested {len(ids)} chunks for doc_id={doc_id}")

    return len(ids)


def list_docs() -> list[str]:
    """Return unique doc_ids stored in the collection."""
    store = _get_vectorstore()
    results = store._collection.get()
    seen, docs = set(), []
    for meta in results.get("metadatas") or []:
        d = meta.get("doc_id", "")
        if d and d not in seen:
            seen.add(d)
            docs.append(d)
    return sorted(docs)
