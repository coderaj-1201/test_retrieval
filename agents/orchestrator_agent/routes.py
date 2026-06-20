"""
HTTP routes for the Orchestrator Agent.

  POST /orchestrate  — main entry point called by the Main Agent
  GET  /health/live  — liveness probe (always 200 while running)
  GET  /health/ready — readiness probe (Cosmos + circuit breaker state)
  GET  /health       — alias for /health/ready
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import Response

from shared.logging_config import bind_context, get_logger
from shared.models import FinalResponse, OrchestratorInput, UserQuery

import agents.orchestrator_agent.retrieval as _retrieval

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health/live")
async def liveness() -> dict:
    """Always 200 while the process is alive."""
    return {"status": "alive", "agent": "orchestrator"}


@router.get("/health/ready")
async def readiness() -> Response:
    """200 only when Cosmos is reachable and the circuit breaker is not open."""
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        from shared.cosmos_client import get_chat_container
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {type(exc).__name__}"
        overall_ok = False

    cb_state = _retrieval._retrieval_breaker.to_dict()
    checks["retrieval_circuit"] = cb_state["state"]
    if cb_state["state"] == "open":
        overall_ok = False

    return Response(
        content=json.dumps({
            "status": "ready" if overall_ok else "degraded",
            "agent":  "orchestrator",
            "checks": checks,
        }),
        media_type="application/json",
        status_code=200 if overall_ok else 503,
    )


@router.get("/health")
async def health() -> dict:
    """Alias for /health/ready — kept for backward compatibility."""
    return {"status": "healthy", "agent": "orchestrator"}


@router.post("/orchestrate")
async def orchestrate(raw: Request) -> Response:
    """
    Accept a query from the Main Agent, run orchestrator_workflow, return FinalResponse.

    The Main Agent sends session/LTM context as plain strings in the body;
    session memory and turn_texts are passed separately via OrchestratorInput
    when calling the workflow from within the same process (test/direct use).
    Over HTTP (normal production path) session and turn_texts arrive as None.
    """
    from agents.orchestrator_agent.workflow import orchestrator_workflow

    body        = await raw.json()
    session_ctx = body.pop("session_context", "")
    ltm_ctx     = body.pop("ltm_context", "")

    user_query = UserQuery(
        text=body.get("text", ""),
        conversation_id=body.get("conversation_id", ""),
        user_id=body.get("user_id", ""),
        question_id=body.get("question_id", ""),
    )
    bind_context(
        agent="orchestrator",
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
    )

    try:
        result_obj = await orchestrator_workflow.run(OrchestratorInput(
            user_query=user_query,
            session_context=session_ctx,
            ltm_context=ltm_ctx,
            session=None,
            turn_texts=None,
        ))
        outputs = result_obj.get_outputs()
        final: FinalResponse = outputs[0] if outputs else FinalResponse(
            status="failure", answer="",
            domain=None,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
        )
    except Exception as exc:
        logger.error("orchestrate_endpoint_error: %s", exc, exc_info=True)
        final = FinalResponse(
            status="error", answer="",
            domain=None,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
        )

    return Response(
        content=json.dumps(final.to_dict()),
        media_type="application/json",
    )
