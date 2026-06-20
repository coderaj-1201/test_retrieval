"""
Hybrid search — local development version.

Delegates to tools/local_search.py (ChromaDB + Mistral embeddings).
The SearchDocument dataclass and fetch_parent_chunk are defined here so
retrieval_agent.py imports keep working unchanged.

On the production branch this file uses Azure AI Search (BM25 + dense vectors
+ semantic reranker). Locally we use cosine similarity on Mistral embeddings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from shared.models import Domain

logger = logging.getLogger(__name__)

_VALID_DOMAINS: frozenset[str] = frozenset(d.value for d in Domain)


@dataclass(frozen=True)
class SearchDocument:
    """Represents a single retrieved chunk — same schema as production."""
    # Core (always present)
    id: str
    content: str
    source: str
    domain: str
    score: float
    # Extended metadata
    parent_id: str          = ""
    chunk_type: str         = "paragraph"
    doc_name: str           = ""
    doc_url: str            = ""
    file_type: str          = ""
    page_number: int        = 0
    title: str              = ""
    section_heading: str    = ""
    section_subheading: str = ""
    table_raw: str          = ""


def hybrid_search(
    query: str,
    domain: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
) -> list[SearchDocument]:
    """
    Semantic search via ChromaDB + Mistral embeddings.
    Delegates to tools.local_search; returns [] on any error.
    """
    from tools.local_search import hybrid_search as _local_search
    return _local_search(query, domain, top_k, chunk_types)


def fetch_parent_chunk(parent_id: str) -> SearchDocument | None:
    """Fetch a chunk by ID from the local ChromaDB collection."""
    from tools.local_search import fetch_parent_chunk as _local_fetch
    return _local_fetch(parent_id)
