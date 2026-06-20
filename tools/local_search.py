"""
Local ChromaDB search backend using Mistral embeddings.
Mirrors the interface expected by scripts/seed_local_docs.py.
"""
from __future__ import annotations

import logging
import os

import chromadb
from chromadb.config import Settings
from mistralai import Mistral

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "policy_docs"
_EMBED_MODEL     = "mistral-embed"
_DB_PATH         = os.getenv("CHROMA_DB_PATH", "./.chromadb")

_client:     chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_mistral:    Mistral | None = None


def _get_mistral() -> Mistral:
    global _mistral
    if _mistral is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable not set")
        _mistral = Mistral(api_key=api_key)
    return _mistral


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    client = _get_mistral()
    response = client.embeddings.create(model=_EMBED_MODEL, inputs=texts)
    return [item.embedding for item in response.data]


def add_documents(docs: list[dict]) -> int:
    """Embed and upsert documents into ChromaDB. Returns count added."""
    collection = _get_collection()
    ids       = [d["id"] for d in docs]
    contents  = [d["content"] for d in docs]
    metadatas = [
        {
            "domain":      d.get("domain", ""),
            "title":       d.get("title", ""),
            "doc_name":    d.get("doc_name", ""),
            "doc_url":     d.get("doc_url", ""),
            "chunk_type":  d.get("chunk_type", "paragraph"),
            "page_number": str(d.get("page_number", 0)),
        }
        for d in docs
    ]

    embeddings = _embed(contents)
    collection.upsert(
        ids=ids,
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(docs)


def search(
    query: str,
    domain: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Return top-n documents for query, optionally filtered by domain."""
    collection = _get_collection()
    query_embedding = _embed([query])[0]

    where = {"domain": domain} if domain else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i, doc_id in enumerate(results["ids"][0]):
        docs.append({
            "id":          doc_id,
            "content":     results["documents"][0][i],
            "score":       1 - results["distances"][0][i],  # cosine similarity
            **results["metadatas"][0][i],
        })
    return docs
