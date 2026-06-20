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
def get_openai_client() -> OpenAI:
    endpoint       = str(settings.AZURE_OPENAI_ENDPOINT).rstrip("/")
    token_provider = get_bearer_token_provider(_credential(), "https://ai.azure.com/.default")
    logger.info("openai_auth=managed_identity endpoint=%s", endpoint)
    return OpenAI(
        base_url    = endpoint,
        api_key     = token_provider(),
        max_retries = 0,
        timeout     = 30.0,
    )


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
