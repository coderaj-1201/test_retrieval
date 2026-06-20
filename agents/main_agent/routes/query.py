"""
POST /query route — main RAG entry point.

Accepts a user question, runs the main_agent_workflow, and returns a
structured QueryResponse. Also handles:
  - Rate limiting (429 with Retry-After header)
  - Prompt injection detection (log warning, do not block)
  - Conversation ID generation when none is supplied
"""
from __future__ import annotations

import json
import re
import uuid

from fastapi import APIRouter
from fastapi.responses import Response

from shared.logging_config import bind_context, get_logger
from shared.models import QueryResponse, UserQuery
from shared.rate_limiter import RateLimitExceeded, check_rate_limit

from agents.main_agent.schemas import QueryBody
from agents.main_agent.workflow import main_agent_workflow

logger = get_logger(__name__)
router = APIRouter()

# Patterns that signal likely prompt injection attempts — logged as WARNING
# but not blocked (LLM guardrails handle the actual defence).
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|above|prior)\s+instructions"
    r"|disregard\s+(all\s+)?instructions"
    r"|you\s+are\s+now\s+"
    r"|system\s+prompt"
    r"|jailbreak)",
    re.IGNORECASE,
)


@router.post("/query")
async def query(body: QueryBody) -> Response:
    """Receive a user query, run the workflow, and return the answer."""
    try:
        check_rate_limit(body.user_id)
    except RateLimitExceeded as exc:
        logger.warning(
            "rate_limit_exceeded user_id=%s retry_after=%.1f",
            body.user_id, exc.retry_after,
        )
        return Response(
            content=json.dumps({
                "error":       "rate_limit_exceeded",
                "retry_after": exc.retry_after,
                "message":     f"Too many requests. Please wait {exc.retry_after}s.",
            }),
            media_type="application/json",
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
        )

    if _INJECTION_PATTERNS.search(body.text):
        logger.warning(
            "potential_prompt_injection_detected user_id=%s text_preview=%.80s",
            body.user_id, body.text,
        )

    conversation_id = body.conversation_id or str(uuid.uuid4())
    user_query = UserQuery(
        text=body.text,
        conversation_id=conversation_id,
        user_id=body.user_id,
        question_id=body.idempotency_key or f"q-{uuid.uuid4().hex[:12]}",
    )
    bind_context(
        agent="main",
        conversation_id=conversation_id,
        user_id=body.user_id,
        question_id=user_query.question_id,
    )
    logger.info("query_received text_preview=%.80s", body.text)

    result_obj = await main_agent_workflow.run(user_query)
    outputs = result_obj.get_outputs()
    response: QueryResponse = outputs[0] if outputs else QueryResponse(
        question_id=user_query.question_id,
        answer_id="",
        conversation_id=conversation_id,
        user_id=body.user_id,
        status="error",
        answer="Internal error.",
        domain="",
        confidence=0.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=None,
    )

    logger.info(
        "query_complete question_id=%s status=%s confidence=%.3f",
        response.question_id, response.status, response.confidence,
    )
    return Response(
        content=json.dumps(response.to_dict()),
        media_type="application/json",
    )
