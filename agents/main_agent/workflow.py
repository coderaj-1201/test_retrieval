"""
Main Agent core workflow.

Contains:
  - call_orchestrator  — @step that sends the query to the Orchestrator via HTTP
  - main_agent_workflow — @workflow that orchestrates memory, routing, persistence
  - LTM background-update helpers (_run_ltm_update, _ltm_task_done_callback)

Flow per request:
  1. Short-circuit escalation tokens (raise_ticket / connect_sme).
  2. Load session memory + LTM from Cosmos.
  3. Batch-fetch turn texts so Orchestrator has context without re-querying Cosmos.
  4. Call Orchestrator (circuit-breaker-protected).
  5. Persist chat record to Cosmos.
  6. Append turn to session (updates off_topic_streak).
  7. Fire LTM update in background every N turns.
"""
from __future__ import annotations

import asyncio
import uuid

import httpx
from agent_framework import step, workflow

from shared.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.config import settings
from shared.cosmos_client import get_chat_container, upsert_document
from shared.logging_config import get_logger
from shared.memory import (
    append_turn,
    fetch_turn_texts,
    format_ltm_context,
    format_session_context,
    load_ltm,
    load_session,
    update_ltm,
)
from shared.models import (
    ChatHistoryRecord,
    ConversationTurn,
    Domain,
    FinalResponse,
    OrchestratorInput,
    QueryResponse,
    UserQuery,
)
from shared.telemetry import record_attempts, record_confidence, record_query

from agents.main_agent.escalation import handle_connect_sme, handle_raise_ticket

logger = get_logger(__name__)

# Module-level shared HTTP client — set during FastAPI lifespan startup.
# Using a module-level reference avoids importing `app` (circular).
_http: httpx.AsyncClient | None = None

_orchestrator_breaker = CircuitBreaker(
    name="orchestrator-agent", fail_max=3, reset_timeout=30
)

# Escalation action metadata surfaced to the front-end so it can render cards.
_ESCALATION_OPTIONS = {
    "raise_ticket": {
        "action": "raise_ticket",
        "description": "Raise a support ticket",
        "reply_with": "raise_ticket",
        "sla": settings.ESCALATION_SLA_TICKET,
    },
    "connect_sme": {
        "action": "connect_sme",
        "description": "Connect with a Subject Matter Expert",
        "reply_with": "connect_sme",
        "sla": settings.ESCALATION_SLA_SME,
    },
}


def _internal_headers() -> dict[str, str]:
    """Return the X-Internal-Secret header if a secret is configured."""
    secret = (
        settings.INTERNAL_API_SECRET.get_secret_value()
        if settings.INTERNAL_API_SECRET is not None
        else None
    )
    return {"X-Internal-Secret": secret} if secret else {}


async def _do_orchestrate(payload: dict) -> dict:
    """Raw HTTP POST to the Orchestrator — wrapped by the circuit breaker."""
    global _http
    client = _http or httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=120.0))
    headers = {**_internal_headers(), "X-Request-ID": payload.get("question_id", "")}
    resp = await client.post(
        f"{str(settings.ORCHESTRATOR_URL).rstrip('/')}/orchestrate",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


@step
async def call_orchestrator(inp: OrchestratorInput) -> FinalResponse:
    """
    Send a query to the Orchestrator and parse the response into a FinalResponse.

    Args:
        inp: OrchestratorInput containing the query, session/LTM context, and
             pre-fetched turn texts.

    Raises:
        CircuitOpenError: when the circuit breaker is tripped after repeated failures.
    """
    user_query = inp.user_query
    payload = {
        "text":            user_query.text,
        "conversation_id": user_query.conversation_id,
        "user_id":         user_query.user_id,
        "question_id":     user_query.question_id,
        "session_context": inp.session_context,
        "ltm_context":     inp.ltm_context,
    }
    try:
        data = await _orchestrator_breaker.call(_do_orchestrate, payload)
        domain_val = data.get("domain") or ""
        try:
            domain = Domain(domain_val.lower()) if domain_val else None
        except ValueError:
            domain = None
        return FinalResponse(
            status=data.get("status", "failure"),
            answer=data.get("answer", ""),
            domain=domain,
            sources=data.get("sources", []),
            confidence=float(data.get("confidence", 0.0)),
            attempts_used=int(data.get("attempts_used", 0)),
            conversation_id=data.get("conversation_id", user_query.conversation_id),
            user_id=data.get("user_id", user_query.user_id),
            question_id=data.get("question_id", user_query.question_id),
            answer_id=data.get("answer_id", f"ans-{uuid.uuid4().hex[:12]}"),
            tools_used=data.get("tools_used", []),
            show_citations=bool(data.get("show_citations", False)),
            citations=data.get("citations", []),
            response_type=data.get("response_type", ""),
        )
    except CircuitOpenError as exc:
        logger.error(
            "orchestrator_circuit_open retry_after=%.1f question_id=%s",
            exc.retry_after, user_query.question_id,
        )
        raise


@workflow(name="main_agent_workflow")
async def main_agent_workflow(user_query: UserQuery) -> QueryResponse:
    """
    Top-level workflow for every user query.

    Steps:
      1. Route escalation tokens directly to escalation handlers.
      2. Load session memory and LTM from Cosmos.
      3. Fetch turn texts for context window.
      4. Call Orchestrator (returns FinalResponse).
      5. Persist ChatHistoryRecord to Cosmos.
      6. Update session memory (streak tracking).
      7. Trigger LTM background update every N turns.
    """
    text_lower = user_query.text.strip().lower()

    # Direct escalation — no LLM call needed.
    if text_lower == "raise_ticket":
        return await handle_raise_ticket(
            user_id=user_query.user_id,
            conversation_id=user_query.conversation_id,
            question_id=user_query.question_id,
            question_text=user_query.text,
            domain="",
        )
    if text_lower == "connect_sme":
        return await handle_connect_sme(
            user_id=user_query.user_id,
            conversation_id=user_query.conversation_id,
            question_id=user_query.question_id,
            question_text=user_query.text,
            domain="",
        )

    session = await load_session(user_query.conversation_id, user_query.user_id)
    ltm = await load_ltm(user_query.user_id)

    # Eagerly fetch all turn texts so the Orchestrator has them for:
    # classifier context, reformat shortcut, and whole-chat summary.
    all_question_ids = [t.question_id for t in session.turns]
    turn_texts = await fetch_turn_texts(user_query.conversation_id, all_question_ids)
    session_context = format_session_context(session, turn_texts)

    try:
        final: FinalResponse = await call_orchestrator(OrchestratorInput(
            user_query=user_query,
            session_context=session_context,
            ltm_context=format_ltm_context(ltm),
            session=session,
            turn_texts=turn_texts,
        ))
    except Exception as exc:
        logger.error("orchestrator_call_failed: %s", exc, exc_info=True)
        return QueryResponse(
            question_id=user_query.question_id,
            answer_id=f"ans-{uuid.uuid4().hex[:12]}",
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            status="error",
            answer="Service temporarily unavailable. Please try again.",
            domain="",
            confidence=0.0,
            attempts_used=0,
            tools_used=[],
            sources=[],
            escalation_options=None,
        )

    # "error" means no usable model output at all; "failure" still has an answer.
    has_answer = final.status != "error"
    domain_str = (
        final.domain.value.upper()
        if isinstance(final.domain, Domain)
        else (final.domain or "")
    )
    domain_metric = (
        final.domain.value if isinstance(final.domain, Domain) else (final.domain or "unknown")
    ).lower()

    record_query(domain=domain_metric, status=final.status, tool=final.tools_used[-1] if final.tools_used else "")
    record_confidence(confidence=final.confidence, domain=domain_metric, status=final.status)
    record_attempts(attempts=final.attempts_used, domain=domain_metric, status=final.status)

    response = QueryResponse(
        question_id=user_query.question_id,
        answer_id=final.answer_id,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        status=final.status,
        answer=final.answer if has_answer else "",
        domain=domain_str,
        confidence=final.confidence,
        attempts_used=final.attempts_used,
        tools_used=final.tools_used,
        sources=final.sources,
        escalation_options=None,
        show_citations=final.show_citations,
        citations=final.citations,
    )

    # Persist synchronously before returning — feedback submitted immediately
    # after the response needs to find a valid answer record in Cosmos.
    await asyncio.to_thread(upsert_document, get_chat_container(), ChatHistoryRecord(
        id=user_query.question_id,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
        answer_id=final.answer_id,
        question=user_query.text,
        answer=final.answer,
        domain=domain_str,
        confidence=final.confidence,
        tools_used=final.tools_used,
        sources=final.sources,
        status=final.status,
        show_citations=final.show_citations,
        citations=final.citations,
    ).to_dict())

    _is_in_domain = final.status in ("success", "failure")
    _is_greeting = final.response_type == "greeting"
    await append_turn(
        session,
        ConversationTurn(
            question_id=user_query.question_id,
            answer_id=final.answer_id,
            domain=domain_str,
            confidence=final.confidence,
            tools_used=final.tools_used,
        ),
        is_in_domain=_is_in_domain,
        is_greeting=_is_greeting,
    )

    if len(session.turns) % settings.LTM_SUMMARY_EVERY_N == 0:
        task = asyncio.create_task(
            _run_ltm_update(user_query.user_id, session),
            name=f"ltm-update-{user_query.user_id}",
        )
        task.add_done_callback(_ltm_task_done_callback)

    return response


def _ltm_task_done_callback(task: asyncio.Task) -> None:
    """Log unhandled exceptions from the background LTM update task."""
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.error(
            "ltm_task_unhandled_exception task=%s: %s",
            task.get_name(), exc, exc_info=exc,
        )


async def _run_ltm_update(user_id: str, session) -> None:
    """Run LTM summarisation with one automatic retry on failure."""
    for attempt in range(1, 3):
        try:
            await update_ltm(user_id, session)
            return
        except Exception as exc:
            logger.error(
                "ltm_update_failed user_id=%s attempt=%d/2: %s",
                user_id, attempt, exc, exc_info=True,
            )
            if attempt < 2:
                await asyncio.sleep(5)
