"""
Local vector search — ChromaDB + Mistral embeddings.

Drop-in replacement for the Azure AI Search calls in hybrid_search_tool.py.
Exposes the same public interface:
  hybrid_search(query, domain, top_k, chunk_types) → list[SearchDocument]
  fetch_parent_chunk(parent_id)                     → SearchDocument | None

ChromaDB persists to LOCAL_SEARCH_DB_PATH (default: ./local_data/chroma).
The collection is created automatically on first use. Documents are added via
  add_documents(docs: list[dict]) — call from a seed script or admin route.

If the collection is empty (no documents indexed yet), hybrid_search returns
an empty list and the retrieval loop will exhaust its attempts and return a
low-confidence "failure" response. That's the correct local-dev behaviour —
add some test documents to get real answers.

Embedding model: Mistral mistral-embed (1024-dim vectors).
"""
from __future__ import annotations

import logging
from functools import lru_cache

import chromadb
from chromadb import Collection

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.models import Domain
from shared.retry import llm_retry
# Re-export SearchDocument so hybrid_search_tool.py callers still work.
from tools.hybrid_search_tool import SearchDocument  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

_VALID_DOMAINS: frozenset[str] = frozenset(d.value for d in Domain)


# ── ChromaDB client + collection ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    client = chromadb.PersistentClient(path=settings.LOCAL_SEARCH_DB_PATH)
    collection = client.get_or_create_collection(
        name=settings.LOCAL_SEARCH_COLLECTION,
        # Use cosine distance (equivalent to dot-product similarity for
        # normalised Mistral embeddings).
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "local_search_collection name=%s count=%d path=%s",
        collection.name, collection.count(), settings.LOCAL_SEARCH_DB_PATH,
    )
    return collection


# ── Embedding ─────────────────────────────────────────────────────────────────

@llm_retry
def _embed(text: str) -> list[float]:
    resp = get_openai_client().embeddings.create(
        input=text,
        model=settings.MISTRAL_EMBEDDING_MODEL,
    )
    return resp.data[0].embedding


# ── Public search interface ───────────────────────────────────────────────────

def hybrid_search(
    query: str,
    domain: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
) -> list[SearchDocument]:
    """
    Semantic search using Mistral embeddings + ChromaDB cosine similarity.

    Filters by domain and optional chunk_types via ChromaDB metadata filters.
    Returns SearchDocument list sorted by descending score.

    Note: ChromaDB returns distance (0=identical, 2=opposite); we convert to
    a similarity score in [0, 1] as: similarity = 1 - distance/2
    """
    if domain not in _VALID_DOMAINS:
        logger.error("local_search_invalid_domain domain=%r", domain)
        return []

    k = top_k or settings.RETRIEVAL_TOP_K
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning(
            "local_search_empty_collection — no documents indexed yet. "
            "Add documents with tools.local_search.add_documents()."
        )
        return []

    try:
        vector = _embed(query)
    except Exception as exc:
        logger.error("local_search_embed_failed: %s", exc)
        return []

    # Build ChromaDB where filter.
    where: dict = {"domain": {"$eq": domain}}
    if chunk_types:
        where = {"$and": [where, {"chunk_type": {"$in": chunk_types}}]}

    try:
        results = collection.query(
            query_embeddings=[vector],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("local_search_query_failed domain=%s: %s", domain, exc)
        return []

    ids       = results["ids"][0]
    docs_text = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    search_docs: list[SearchDocument] = []
    for doc_id, content, meta, dist in zip(ids, docs_text, metas, distances):
        # Convert cosine distance → similarity score in [0, 1].
        score = max(0.0, 1.0 - dist / 2.0)
        search_docs.append(SearchDocument(
            id                 = doc_id,
            content            = content,
            source             = meta.get("doc_name") or meta.get("source", "local"),
            domain             = meta.get("domain", domain),
            score              = score,
            parent_id          = meta.get("parent_id", ""),
            chunk_type         = meta.get("chunk_type", "paragraph"),
            doc_name           = meta.get("doc_name", ""),
            doc_url            = meta.get("doc_url", ""),
            file_type          = meta.get("file_type", ""),
            page_number        = int(meta.get("page_number", 0)),
            title              = meta.get("title", ""),
            section_heading    = meta.get("section_heading", ""),
            section_subheading = meta.get("section_subheading", ""),
            table_raw          = meta.get("table_raw", ""),
        ))

    search_docs.sort(key=lambda d: d.score, reverse=True)
    logger.debug(
        "local_search domain=%s docs=%d top_score=%.3f",
        domain, len(search_docs), search_docs[0].score if search_docs else 0.0,
    )
    return search_docs


def fetch_parent_chunk(parent_id: str) -> SearchDocument | None:
    """Retrieve a specific chunk by ID from the local collection."""
    if not parent_id:
        return None
    collection = _get_collection()
    try:
        result = collection.get(ids=[parent_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        content = result["documents"][0]
        meta    = result["metadatas"][0]
        return SearchDocument(
            id      = parent_id,
            content = content,
            source  = meta.get("doc_name") or meta.get("source", "local"),
            domain  = meta.get("domain", ""),
            score   = 1.0,
            parent_id          = "",
            chunk_type         = meta.get("chunk_type", "paragraph"),
            doc_name           = meta.get("doc_name", ""),
            doc_url            = meta.get("doc_url", ""),
            file_type          = meta.get("file_type", ""),
            page_number        = int(meta.get("page_number", 0)),
            title              = meta.get("title", ""),
            section_heading    = meta.get("section_heading", ""),
            section_subheading = meta.get("section_subheading", ""),
            table_raw          = meta.get("table_raw", ""),
        )
    except Exception as exc:
        logger.error("local_fetch_parent_failed id=%s: %s", parent_id, exc)
        return None


# ── Document ingestion helper ─────────────────────────────────────────────────

def add_documents(docs: list[dict]) -> int:
    """
    Add documents to the local ChromaDB collection.

    Each doc dict should have:
      id (str), content (str), domain (str), and optional metadata fields
      (doc_name, doc_url, chunk_type, page_number, title, section_heading,
       section_subheading, table_raw, parent_id, file_type).

    Returns the number of documents successfully added.
    """
    collection = _get_collection()
    added = 0

    for doc in docs:
        doc_id  = doc.get("id", "")
        content = doc.get("content", "")
        if not doc_id or not content:
            logger.warning("add_documents_skip missing id or content: %s", doc.get("id"))
            continue

        try:
            vector = _embed(content)
            meta = {
                k: str(v) for k, v in doc.items()
                if k not in ("id", "content") and v is not None
            }
            collection.upsert(
                ids=[doc_id],
                embeddings=[vector],
                documents=[content],
                metadatas=[meta],
            )
            added += 1
        except Exception as exc:
            logger.error("add_document_failed id=%s: %s", doc_id, exc)

    logger.info("add_documents total=%d added=%d", len(docs), added)
    return added
