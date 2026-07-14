from pathlib import Path

import chromadb
from loguru import logger

from config.settings import CHROMA_PERSIST, CHUNK_SIZE, CHUNK_OVERLAP, MODEL_ROUTER
from serving.llm_client import LLMClient

_llm = LLMClient()


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST)
    return client.get_or_create_collection(
        name="2plus_docs",
        metadata={"hnsw:space": "cosine"},
    )


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def _load_text(path_or_text: str) -> str:
    p = Path(path_or_text)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    return path_or_text


def delete_by_doc_id(doc_id: str) -> None:
    col = _get_collection()
    results = col.get(where={"doc_id": doc_id})
    if results["ids"]:
        col.delete(ids=results["ids"])
        logger.info(f"deleted {len(results['ids'])} chunks for doc_id={doc_id}")


def ingest(path_or_text: str, doc_id: str, metadata: dict | None = None) -> int:
    """Ingest a document. Re-upload replaces previous version cleanly."""
    metadata = metadata or {}
    delete_by_doc_id(doc_id)

    text = _load_text(path_or_text)
    chunks = _chunk(text)
    if not chunks:
        return 0

    col = _get_collection()
    ids, embeddings, documents, metas = [], [], [], []

    for i, chunk in enumerate(chunks):
        emb = _llm.embed(chunk, model=MODEL_ROUTER["embed"])
        if not emb:
            logger.warning(f"empty embedding for chunk {i} of {doc_id}")
            continue
        chunk_id = f"{doc_id}::chunk{i}"
        ids.append(chunk_id)
        embeddings.append(emb)
        documents.append(chunk)
        metas.append({"doc_id": doc_id, "chunk_idx": i, **metadata})

    if ids:
        col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metas)
        logger.info(f"ingested {len(ids)} chunks for doc_id={doc_id}")

    return len(ids)


def list_docs() -> list[str]:
    """Return unique doc_ids stored in the collection."""
    col = _get_collection()
    results = col.get()
    seen, docs = set(), []
    for meta in results.get("metadatas") or []:
        d = meta.get("doc_id", "")
        if d and d not in seen:
            seen.add(d)
            docs.append(d)
    return sorted(docs)
