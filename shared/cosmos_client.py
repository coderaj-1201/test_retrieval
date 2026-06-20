"""
SQLite-backed drop-in replacement for cosmos_client.py.

Exposes the same public API as the Cosmos version so no agent or memory code
needs to change:
  get_chat_container()     → SQLiteContainer("chat_history")
  get_feedback_container() → SQLiteContainer("feedback")
  get_sessions_container() → SQLiteContainer("sessions")
  get_ltm_container()      → SQLiteContainer("long_term_memory")
  upsert_document(container, doc)
  get_document(container, item_id, partition_key)
  query_documents(container, query, params, partition_key=None)
  probe_cosmos()

Each "container" is a single SQLite table with columns:
  id TEXT PRIMARY KEY, partition_key TEXT, data TEXT (JSON blob)

Cosmos SQL queries are NOT translated — instead query_documents() does a
full table scan, deserializes each row, and filters in Python. This is fine
for local dev volumes (hundreds of records, not millions).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from threading import Lock

from shared.config import settings

logger = logging.getLogger(__name__)

# One global lock keeps multi-threaded FastAPI workers from corrupting the DB.
_db_lock = Lock()


# ── SQLite container proxy ─────────────────────────────────────────────────────

class SQLiteContainer:
    """Mimics azure.cosmos.ContainerProxy for local dev."""

    def __init__(self, table_name: str) -> None:
        self.id = table_name          # .id mirrors ContainerProxy.id used in logs
        self._table = table_name
        self._db_path = settings.SQLITE_DB_PATH
        self._ensure_table()

    # Called by probe_cosmos() to verify the container is accessible.
    def read(self) -> dict:
        return {"id": self.id}

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with _db_lock, self._conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self._table}" (
                    id            TEXT PRIMARY KEY,
                    partition_key TEXT,
                    data          TEXT NOT NULL
                )
            """)
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{self._table}_pk '
                f'ON "{self._table}" (partition_key)'
            )

    def upsert_item(self, *, body: dict) -> dict:
        """Insert or replace a document. 'id' field is required."""
        doc_id = body.get("id", "")
        pk     = body.get("conversation_id") or body.get("user_id") or body.get("question_id") or doc_id
        with _db_lock, self._conn() as conn:
            conn.execute(
                f'INSERT OR REPLACE INTO "{self._table}" (id, partition_key, data) VALUES (?,?,?)',
                (doc_id, pk, json.dumps(body)),
            )
        return body

    def read_item(self, *, item: str, partition_key: str) -> dict:
        """Return the document or raise KeyError if not found."""
        with _db_lock, self._conn() as conn:
            row = conn.execute(
                f'SELECT data FROM "{self._table}" WHERE id = ?', (item,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Document '{item}' not found in '{self._table}'")
        return json.loads(row["data"])

    def query_items(
        self,
        *,
        query: str,
        parameters: list[dict],
        partition_key: str | None = None,
        enable_cross_partition_query: bool = False,
    ) -> list[dict]:
        """
        Full-table scan with Python-level filtering.

        Cosmos SQL is not parsed — we load all rows (optionally scoped to a
        partition) and filter using the @param values extracted from `parameters`.
        Supports equality filters on top-level string fields only.
        """
        param_map = {p["name"]: p["value"] for p in parameters}

        with _db_lock, self._conn() as conn:
            if partition_key:
                rows = conn.execute(
                    f'SELECT data FROM "{self._table}" WHERE partition_key = ?',
                    (partition_key,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f'SELECT data FROM "{self._table}"'
                ).fetchall()

        docs = [json.loads(r["data"]) for r in rows]

        # Simple filter: look for @param occurrences in the query string and
        # match against the corresponding top-level document field.
        for param_name, param_value in param_map.items():
            if param_name == "@limit":
                continue
            # Strip the leading @ to get the field name used in the query.
            # e.g. "@conv_id" matches "c.conversation_id = @conv_id"
            # We search the query text for "c.<field> = @param_name" patterns.
            field_name = _infer_field(query, param_name)
            if field_name:
                docs = [d for d in docs if str(d.get(field_name, "")) == str(param_value)]

        # Apply LIMIT from @limit param if present.
        limit = int(param_map.get("@limit", 1000))
        return docs[:limit]

    def delete_item(self, *, item: str, partition_key: str) -> None:
        with _db_lock, self._conn() as conn:
            conn.execute(f'DELETE FROM "{self._table}" WHERE id = ?', (item,))


def _infer_field(query: str, param_name: str) -> str | None:
    """
    Extract the document field name for a given @param from a Cosmos SQL string.

    Examples:
      "c.conversation_id = @conv_id" + "@conv_id" → "conversation_id"
      "c.user_id = @user_id"         + "@user_id" → "user_id"
    """
    import re
    # Match "c.<field> = @param_name" or "c.<field> eq @param_name"
    pattern = rf'c\.(\w+)\s+(?:=|eq)\s+{re.escape(param_name)}'
    m = re.search(pattern, query, re.IGNORECASE)
    return m.group(1) if m else None


# ── Cached container accessors (mirrors cosmos_client.py public API) ──────────

@lru_cache(maxsize=1)
def get_chat_container() -> SQLiteContainer:
    return SQLiteContainer(settings.COSMOS_CONTAINER_CHAT)


@lru_cache(maxsize=1)
def get_feedback_container() -> SQLiteContainer:
    return SQLiteContainer(settings.COSMOS_CONTAINER_FEEDBACK)


@lru_cache(maxsize=1)
def get_sessions_container() -> SQLiteContainer:
    return SQLiteContainer(settings.COSMOS_CONTAINER_SESSIONS)


@lru_cache(maxsize=1)
def get_ltm_container() -> SQLiteContainer:
    return SQLiteContainer(settings.COSMOS_CONTAINER_LTM)


# ── Startup probe ──────────────────────────────────────────────────────────────

def probe_cosmos() -> None:
    """Verify all four SQLite tables are reachable. Logs clearly on failure."""
    for fn, label in [
        (get_chat_container,     settings.COSMOS_CONTAINER_CHAT),
        (get_feedback_container, settings.COSMOS_CONTAINER_FEEDBACK),
        (get_sessions_container, settings.COSMOS_CONTAINER_SESSIONS),
        (get_ltm_container,      settings.COSMOS_CONTAINER_LTM),
    ]:
        fn().read()
        logger.info("sqlite_probe_ok table=%s db=%s", label, settings.SQLITE_DB_PATH)


# ── Generic helpers (same signatures as cosmos_client.py) ─────────────────────

def upsert_document(container: SQLiteContainer, doc: dict) -> None:
    """Fire-and-forget upsert. Logs on failure, never raises."""
    try:
        container.upsert_item(body=doc)
    except Exception as exc:
        logger.error(
            "sqlite_upsert_failed table=%s id=%s: %s",
            container.id, doc.get("id"), exc,
        )


def get_document(container: SQLiteContainer, item_id: str, partition_key: str) -> dict | None:
    try:
        return container.read_item(item=item_id, partition_key=partition_key)
    except KeyError:
        return None
    except Exception as exc:
        logger.error(
            "sqlite_read_failed table=%s id=%s: %s", container.id, item_id, exc
        )
        return None


def query_documents(
    container: SQLiteContainer,
    query: str,
    params: list[dict],
    partition_key: str | None = None,
) -> list[dict]:
    """Execute a parameterised query (Python-level filtering of SQLite rows)."""
    try:
        return container.query_items(
            query=query,
            parameters=params,
            partition_key=partition_key,
        )
    except Exception as exc:
        logger.error(
            "sqlite_query_failed table=%s query=%.80s: %s", container.id, query, exc
        )
        return []
