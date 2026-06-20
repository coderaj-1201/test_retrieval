"""
Azure client factories.

In production (Azure Container Apps) all access uses DefaultAzureCredential
(managed identity). For local development, set the optional *_KEY / *_API_KEY
environment variables in .env to bypass managed identity entirely.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
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
    if settings.AZURE_OPENAI_API_KEY:
        logger.info("openai_auth=api_key endpoint=%s", endpoint)
        return AzureOpenAI(
            azure_endpoint = endpoint,
            api_key        = settings.AZURE_OPENAI_API_KEY.get_secret_value(),
            api_version    = settings.AZURE_OPENAI_API_VERSION,
            max_retries    = 0,
            timeout        = 30.0,
        )
    logger.info("openai_auth=managed_identity endpoint=%s", endpoint)
    token_provider = get_bearer_token_provider(
        _credential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint          = endpoint,
        azure_ad_token_provider = token_provider,
        api_version             = settings.AZURE_OPENAI_API_VERSION,
        max_retries = 0,
        timeout     = 30.0,
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    if settings.AZURE_SEARCH_API_KEY:
        logger.info("search_auth=api_key")
        return SearchClient(
            endpoint   = str(settings.AZURE_SEARCH_ENDPOINT),
            index_name = settings.AZURE_SEARCH_INDEX,
            credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value()),
        )
    logger.info("search_auth=managed_identity")
    return SearchClient(
        endpoint    = str(settings.AZURE_SEARCH_ENDPOINT),
        index_name  = settings.AZURE_SEARCH_INDEX,
        credential  = _credential(),
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    if settings.AZURE_SEARCH_API_KEY:
        return SearchIndexClient(
            endpoint   = str(settings.AZURE_SEARCH_ENDPOINT),
            credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value()),
        )
    return SearchIndexClient(
        endpoint   = str(settings.AZURE_SEARCH_ENDPOINT),
        credential = _credential(),
    )
