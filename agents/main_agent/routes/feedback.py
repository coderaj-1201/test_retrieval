"""
Feedback routes for the Main Agent.

  POST /feedback  — submit a thumbs-up/down rating + optional comment
  GET  /feedback  — retrieve feedback records (scoped by question, answer,
                    conversation, or user)

Ownership check on POST: a user can only rate their own answers. This
prevents spoofing by modifying the Adaptive Card payload.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Query
from fastapi.responses import Response

from shared.cosmos_client import (
    get_chat_container, get_document, get_feedback_container,
    query_documents, upsert_document,
)
from shared.logging_config import bind_context, get_logger
from shared.models import FeedbackRecord

from agents.main_agent.schemas import FeedbackBody

logger = get_logger(__name__)
router = APIRouter()


@router.post("/feedback")
async def feedback_post(body: FeedbackBody) -> Response:
    """Submit a rating for a specific answer. Validates ownership before writing."""
    bind_context(
        agent="main",
        conversation_id=body.conversation_id,
        user_id=body.user_id,
        question_id=body.question_id,
    )
    logger.info(
        "feedback_received question_id=%s answer_id=%s rating=%s",
        body.question_id, body.answer_id, body.rating,
    )

    # Validate ownership — prevent rating someone else's answer.
    existing_answer = await __import__("asyncio").to_thread(
        get_document, get_chat_container(), body.question_id, body.conversation_id
    )
    if not existing_answer:
        logger.warning(
            "feedback_invalid_answer_id question_id=%s user_id=%s",
            body.question_id, body.user_id,
        )
        return Response(
            content=json.dumps({"status": "error", "detail": "Answer not found."}),
            media_type="application/json",
            status_code=404,
        )
    if existing_answer.get("user_id") != body.user_id:
        logger.warning(
            "feedback_ownership_mismatch question_id=%s claimant=%s owner=%s",
            body.question_id, body.user_id, existing_answer.get("user_id"),
        )
        return Response(
            content=json.dumps({"status": "error", "detail": "Forbidden."}),
            media_type="application/json",
            status_code=403,
        )

    record = FeedbackRecord(
        id=f"fb-{body.answer_id}",   # fixed ID — Cosmos upsert overwrites on re-submit
        question_id=body.question_id,
        answer_id=body.answer_id,
        user_id=body.user_id,
        conversation_id=body.conversation_id,
        rating=body.rating,
        comment=body.comment,
    )
    upsert_document(get_feedback_container(), record.to_dict())
    return Response(
        content=json.dumps({
            "status":      "ok",
            "feedback_id": record.id,
            "question_id": body.question_id,
            "answer_id":   body.answer_id,
            "rating":      body.rating,
            "timestamp":   record.timestamp,
        }),
        media_type="application/json",
    )


@router.get("/feedback")
async def feedback_get(
    answer_id:       str | None = Query(default=None),
    question_id:     str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    user_id:         str        = Query(default="anonymous"),
    limit:           int        = Query(default=20, ge=1, le=100),
) -> Response:
    """Retrieve feedback records. Scope by question_id for single-partition queries."""
    bind_context(agent="main", user_id=user_id)

    # Prefer partition-scoped queries (question_id) when available.
    if question_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.question_id = @question_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@question_id", "value": question_id},
            {"name": "@limit",       "value": limit},
        ]
        docs = query_documents(
            get_feedback_container(), cosmos_query, params, partition_key=question_id
        )
    elif answer_id:
        # Cross-partition — admin/analytics path only.
        cosmos_query = (
            "SELECT * FROM c WHERE c.answer_id = @answer_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@answer_id", "value": answer_id},
            {"name": "@limit",     "value": limit},
        ]
        docs = query_documents(get_feedback_container(), cosmos_query, params)
    elif conversation_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.conversation_id = @conv_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@conv_id", "value": conversation_id},
            {"name": "@limit",   "value": limit},
        ]
        docs = query_documents(get_feedback_container(), cosmos_query, params)
    else:
        cosmos_query = (
            "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit",   "value": limit},
        ]
        docs = query_documents(get_feedback_container(), cosmos_query, params)

    clean = [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs]

    summary: dict | None = None
    if answer_id or question_id:
        counts: dict[str, int] = {}
        for d in clean:
            r = d.get("rating", "")
            counts[r] = counts.get(r, 0) + 1
        summary = {"total": len(clean), "by_rating": counts}

    return Response(
        content=json.dumps({"count": len(clean), "summary": summary, "feedback": clean}),
        media_type="application/json",
    )
