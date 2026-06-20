"""
Cosmos DB client factory + container accessors — managed identity only.

Authentication uses DefaultAzureCredential. The managed identity must be assigned
the Cosmos DB Built-in Data Contributor role on the Cosmos account (data-plane RBAC,
not Azure IAM). See infra/PERMISSIONS.md for the exact az CLI command.

Partition key layout (must match provisioned Cosmos containers):
  chat-history     → /conversation_id
  feedback         → /question_id
  sessions         → /conversation_id
  long-term-memory → /user_id
"""
from __future__ import annotations

import logging
from functools import lru_cache

from azure.cosmos import CosmosClient, ContainerProxy, exceptions as cosmos_exc
from azure.identity import DefaultAzureCredential

from shared.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cosmos_client() -> CosmosClient:
    logger.info("cosmos_auth=managed_identity")
    return CosmosClient(
        url        = str(settings.COSMOS_ENDPOINT),
        credential = DefaultAzureCredential(),
    )


@lru_cache(maxsize=1)
def _get_database():
    client = get_cosmos_client()
    try:
        db = client.get_database_client(settings.COSMOS_DATABASE)
        db.read()
        return db
    except cosmos_exc.CosmosResourceNotFoundError:
        raise RuntimeError(
            f"Cosmos database '{settings.COSMOS_DATABASE}' does not exist. "
            "Run scripts/setup_cosmos.py before deploying."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Cosmos DB: {exc}") from exc


def _get_container(container_name: str) -> ContainerProxy:
    db = _get_database()
    try:
        container = db.get_container_client(container_name)
        container.read()
        return container
    except cosmos_exc.CosmosResourceNotFoundError:
        raise RuntimeError(
            f"Cosmos container '{container_name}' does not exist in database "
            f"'{settings.COSMOS_DATABASE}'. Run scripts/setup_cosmos.py."
        )


@lru_cache(maxsize=1)
def get_chat_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_CHAT)


@lru_cache(maxsize=1)
def get_feedback_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_FEEDBACK)


@lru_cache(maxsize=1)
def get_sessions_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_SESSIONS)


@lru_cache(maxsize=1)
def get_ltm_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_LTM)


def probe_cosmos() -> None:
    """Call once at startup. Forces container caching and fails loudly on misconfiguration."""
    for fn, label in [
        (get_chat_container,     settings.COSMOS_CONTAINER_CHAT),
        (get_feedback_container, settings.COSMOS_CONTAINER_FEEDBACK),
        (get_sessions_container, settings.COSMOS_CONTAINER_SESSIONS),
        (get_ltm_container,      settings.COSMOS_CONTAINER_LTM),
    ]:
        fn()
        logger.info("cosmos_probe_ok container=%s", label)

    try:
        sessions_props = get_sessions_container().read()
        ttl_setting = sessions_props.get("resource", {}).get("defaultTtl")
        if ttl_setting is None:
            logger.warning(
                "cosmos_sessions_ttl_not_enabled: container '%s' has no defaultTtl. "
                "Session documents will never expire.",
                settings.COSMOS_CONTAINER_SESSIONS,
            )
        else:
            logger.info("cosmos_sessions_ttl_ok default_ttl=%s", ttl_setting)
    except Exception as exc:
        logger.warning("cosmos_sessions_ttl_check_failed: %s", exc)


def upsert_document(container: ContainerProxy, doc: dict) -> None:
    try:
        container.upsert_item(body=doc)
    except cosmos_exc.CosmosHttpResponseError as exc:
        logger.error(
            "cosmos_upsert_failed container=%s id=%s status=%s: %s",
            container.id, doc.get("id"), exc.status_code, exc.message,
        )
    except Exception as exc:
        logger.error(
            "cosmos_upsert_unexpected container=%s id=%s: %s",
            container.id, doc.get("id"), exc,
        )


def get_document(container: ContainerProxy, item_id: str, partition_key: str) -> dict | None:
    try:
        return container.read_item(item=item_id, partition_key=partition_key)
    except cosmos_exc.CosmosResourceNotFoundError:
        return None
    except cosmos_exc.CosmosHttpResponseError as exc:
        logger.error(
            "cosmos_read_failed container=%s id=%s status=%s: %s",
            container.id, item_id, exc.status_code, exc.message,
        )
        return None
    except Exception as exc:
        logger.error(
            "cosmos_read_unexpected container=%s id=%s: %s",
            container.id, item_id, exc,
        )
        return None


def query_documents(
    container: ContainerProxy,
    query: str,
    params: list[dict],
    partition_key: str | None = None,
) -> list[dict]:
    cross_partition = partition_key is None
    if cross_partition:
        logger.warning(
            "cosmos_cross_partition_query container=%s query=%.80s",
            container.id, query,
        )
    try:
        kwargs: dict = {"query": query, "parameters": params}
        if cross_partition:
            kwargs["enable_cross_partition_query"] = True
        else:
            kwargs["partition_key"] = partition_key
        return list(container.query_items(**kwargs))
    except cosmos_exc.CosmosHttpResponseError as exc:
        logger.error(
            "cosmos_query_failed container=%s status=%s query=%.80s: %s",
            container.id, exc.status_code, query, exc.message,
        )
        return []
    except Exception as exc:
        logger.error(
            "cosmos_query_unexpected container=%s query=%.80s: %s",
            container.id, query, exc,
        )
        return []
