"""
Azure client factories — managed identity only.

All Azure service access uses DefaultAzureCredential.
No API keys are accepted or used anywhere in this module.

In Azure Container Apps the managed identity is automatically available to the
process — no credential configuration needed beyond assigning the identity the
correct RBAC roles:
  OpenAI   : Cognitive Services OpenAI User
  Search   : Search Index Data Reader (+ Contributor for index management)
  Cosmos   : Cosmos DB Built-in Data Contributor
  Foundry  : Azure AI Developer
"""
from __future__ import annotations

import logging
from functools import lru_cache

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI

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
def get_openai_client() -> AzureOpenAI:
    endpoint = str(settings.AZURE_OPENAI_ENDPOINT).rstrip("/")
    logger.info("openai_auth=managed_identity endpoint=%s", endpoint)
    token_provider = get_bearer_token_provider(
        _credential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint          = endpoint,
        azure_ad_token_provider = token_provider,
        api_version             = settings.AZURE_OPENAI_API_VERSION,
        # max_retries=0: tenacity owns retries; SDK default of 2 would
        # multiply attempts (2 × tenacity 3 = 6 calls).
        max_retries = 0,
        timeout     = 30.0,
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    logger.info("search_auth=managed_identity")
    return SearchClient(
        endpoint    = str(settings.AZURE_SEARCH_ENDPOINT),
        index_name  = settings.AZURE_SEARCH_INDEX,
        credential  = _credential(),
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint   = str(settings.AZURE_SEARCH_ENDPOINT),
        credential = _credential(),
    )
