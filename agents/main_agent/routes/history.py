"""
GET /chat-history route.

Returns persisted chat turns for a conversation or user. Used by the
front-end to render conversation history on reload.

The chat-history container is partitioned by /conversation_id, so:
  - Querying by conversation_id → efficient single-partition scan.
  - Querying by user_id only    → cross-partition scan (admin/analytics path).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Query
from fastapi.responses import Response

from shared.cosmos_client import get_chat_container, query_documents
from shared.logging_config import bind_context, get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/chat-history")
async def chat_history(
    conversation_id: str | None = Query(default=None),
    user_id:         str        = Query(default="anonymous"),
    limit:           int        = Query(default=20, ge=1, le=100),
) -> Response:
    """Return chat history records, newest first."""
    bind_context(agent="main", conversation_id=conversation_id or "", user_id=user_id)
    logger.info(
        "chat_history_requested conversation_id=%s user_id=%s",
        conversation_id, user_id,
    )

    if conversation_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.conversation_id = @conv_id AND c.type = 'chat_history' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@conv_id", "value": conversation_id},
            {"name": "@limit",   "value": limit},
        ]
        docs = query_documents(
            get_chat_container(), cosmos_query, params, partition_key=conversation_id
        )
    else:
        # Cross-partition — admin path only.
        cosmos_query = (
            "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'chat_history' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit",   "value": limit},
        ]
        docs = query_documents(get_chat_container(), cosmos_query, params)

    clean = [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs]
    return Response(
        content=json.dumps({"count": len(clean), "history": clean}),
        media_type="application/json",
    )
