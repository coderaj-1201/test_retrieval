"""
Azure client factories — managed identity only.

All access uses DefaultAzureCredential. In Azure Container Apps the managed
identity is automatically available; no credential configuration is required
beyond assigning the identity the correct RBAC roles (see infra/PERMISSIONS.md).

Required roles per service:
  Azure AI Foundry  : Azure AI Developer
  Azure OpenAI      : Cognitive Services OpenAI User
  Azure AI Search   : Search Index Data Reader + Search Index Data Contributor
  Cosmos DB         : Cosmos DB Built-in Data Contributor (data-plane, not IAM)
  Service Bus       : Azure Service Bus Data Sender
"""
from __future__ import annotations

import logging
from functools import lru_cache

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import OpenAI

from shared.config import settings

logger = logging.getLogger(__name__)


def _credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


@lru_cache(maxsize=1)
def get_foundry_client() -> AIProjectClient:
    return AIProjectClient(
        endpoint   = str(settings.AZURE_FOUNDRY_PROJECT_ENDPOINT),
        credential = _credential(),
    )


@lru_cache(maxsize=1)
def _openai_token_provider():
    """Cached token provider callable — NOT the token itself. Tokens auto-refresh on call."""
    return get_bearer_token_provider(_credential(), "https://ai.azure.com/.default")


@lru_cache(maxsize=1)
def _claude_token_provider():
    """Token provider for Claude on Azure AI Foundry (Cognitive Services scope)."""
    return get_bearer_token_provider(_credential(), "https://cognitiveservices.azure.com/.default")


def get_openai_client() -> OpenAI:
    """
    Returns a cached OpenAI client. The managed-identity token is fetched fresh
    on each call via the token provider (tokens are cached internally by the
    azure-identity SDK for ~1h and auto-refreshed), but the heavy client object
    is not re-created on every call.
    """
    endpoint = str(settings.AZURE_OPENAI_ENDPOINT).rstrip("/")
    token = _openai_token_provider()()   # cheap — SDK returns cached token until near expiry
    client = _openai_client_cache.get("client")
    if client is None:
        logger.info("openai_auth=managed_identity endpoint=%s", endpoint)
        client = OpenAI(
            base_url    = endpoint,
            api_key     = token,
            max_retries = 0,
            timeout     = 600,
        )
        _openai_client_cache["client"] = client
    else:
        # Refresh the api_key in-place so it always holds a valid token.
        client.api_key = token
    return client


_openai_client_cache: dict = {}


_claude_client_cache: dict = {}


def get_claude_client():
    """
    Returns a cached AnthropicFoundry client for the Claude deployment.
    Returns None if CLAUDE_ENDPOINT is not configured.
    The managed-identity bearer token is used as the api_key (refreshed each call).
    """
    if not settings.CLAUDE_ENDPOINT:
        return None
    try:
        from anthropic import AnthropicFoundry  # type: ignore[import-untyped]
    except ImportError:
        logger.error("anthropic package not installed — cannot use Claude routing")
        return None

    endpoint = str(settings.CLAUDE_ENDPOINT).rstrip("/")
    client   = _claude_client_cache.get("client")
    if client is None:
        if settings.CLAUDE_API_KEY:
            logger.info("claude_auth=api_key endpoint=%s", endpoint)
            client = AnthropicFoundry(
                api_key  = settings.CLAUDE_API_KEY.get_secret_value(),
                base_url = endpoint,
            )
        else:
            logger.info("claude_auth=managed_identity endpoint=%s", endpoint)
            client = AnthropicFoundry(
                azure_ad_token_provider = _claude_token_provider(),
                base_url                = endpoint,
            )
        _claude_client_cache["client"] = client
    return client


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    logger.info("search_auth=managed_identity")
    return SearchClient(
        endpoint   = str(settings.AZURE_SEARCH_ENDPOINT),
        index_name = settings.AZURE_SEARCH_INDEX,
        credential = _credential(),
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint   = str(settings.AZURE_SEARCH_ENDPOINT),
        credential = _credential(),
    )
