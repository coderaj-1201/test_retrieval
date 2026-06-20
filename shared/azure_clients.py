"""
Local AI client factory — Mistral AI via OpenAI-compatible SDK.

Replaces azure_clients.py for local development. Uses the standard `openai`
Python package pointed at the Mistral API endpoint, so all existing call sites
(chat completions, embeddings) work without change.

Azure Search / Foundry clients are stubbed — they are not used in local mode
(hybrid_search_tool.py is replaced by local_search.py on this branch).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from openai import OpenAI

from shared.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Return an OpenAI-compatible client pointing to Mistral AI.

    The openai SDK is fully compatible with Mistral's v1 endpoint, so all
    chat-completion and embedding calls work without code changes.
    """
    logger.info(
        "mistral_client_init base_url=%s model=%s",
        settings.MISTRAL_BASE_URL, settings.MISTRAL_CHAT_MODEL,
    )
    return OpenAI(
        api_key=settings.MISTRAL_API_KEY.get_secret_value(),
        base_url=settings.MISTRAL_BASE_URL,
        max_retries=0,   # tenacity owns retries
        timeout=60.0,
    )


# ── Stubs for symbols imported by other shared modules ────────────────────────
# These are never called in local mode but must exist so imports don't fail.

def get_foundry_client():
    raise NotImplementedError("Azure AI Foundry is not available in local-run mode.")


def get_search_client():
    raise NotImplementedError(
        "Azure AI Search is not available in local-run mode. "
        "Use local_search.py (ChromaDB) instead."
    )


def get_search_index_client():
    raise NotImplementedError(
        "Azure AI Search index client is not available in local-run mode."
    )
