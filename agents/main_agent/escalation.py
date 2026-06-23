"""
Escalation workflow steps for the Main Agent.

Handles two escalation actions triggered when the user replies with a special
action token (``raise_ticket`` or ``connect_sme``) from an Adaptive Card:

  - handle_raise_ticket  — queues a Zendesk/Service-Bus support ticket
  - handle_connect_sme   — queues an SME connection request

Both steps are idempotency-safe: submitting the same action twice for the
same conversation returns the existing reference rather than creating a
duplicate.
"""
from __future__ import annotations

import asyncio
import uuid

from agent_framework import step

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.cosmos_client import (
    get_chat_container, get_document, get_feedback_container,
    query_documents, upsert_document,
)
from shared.escalation_client import (
    connect_sme as sb_connect_sme,
    is_escalation_configured,
    raise_ticket as sb_raise_ticket,
)
from shared.logging_config import get_logger
from shared.models import QueryResponse
from shared.telemetry import record_escalation

logger = get_logger(__name__)


async def _fetch_escalation_context(
    user_id: str,
    conversation_id: str,
    question_id: str,
) -> tuple[str, list[dict], str, str]:
    """
    Fetch the most recent chat record and any feedback for it.
    Returns (previous_answer, sources, feedback_rating, feedback_comment).
    """
    previous_answer  = ""
    sources: list[dict] = []
    feedback_rating  = ""
    feedback_comment = ""

    chat_record = await asyncio.to_thread(
        get_document, get_chat_container(), question_id, conversation_id
    )
    if chat_record:
        previous_answer = chat_record.get("answer", "")
        raw_sources     = chat_record.get("sources") or chat_record.get("citations") or []
        sources = [
            {
                "title":   s.get("title") or s.get("doc_name") or s.get("source") or "",
                "url":     s.get("url") or s.get("doc_url") or "",
                "excerpt": s.get("excerpt") or "",
            }
            for s in raw_sources[:5]
        ]

    feedback_docs = await asyncio.to_thread(
        query_documents,
        get_feedback_container(),
        "SELECT TOP 1 * FROM c WHERE c.question_id = @qid ORDER BY c.timestamp DESC",
        [{"name": "@qid", "value": question_id}],
        question_id,
    )
    if feedback_docs:
        fb = feedback_docs[0]
        feedback_rating  = fb.get("rating", "")
        feedback_comment = fb.get("comment", "")

    return previous_answer, sources, feedback_rating, feedback_comment


async def _generate_issue_summary(
    question_text: str,
    previous_answer: str,
    feedback_rating: str,
    feedback_comment: str,
    domain: str,
) -> str:
    """Generate a concise LLM summary of why the user is escalating."""
    context_parts = [f"User question: {question_text}"]
    if previous_answer:
        context_parts.append(f"Bot answer (truncated): {previous_answer[:800]}")
    if feedback_rating:
        context_parts.append(f"User feedback rating: {feedback_rating}")
    if feedback_comment:
        context_parts.append(f"User feedback comment: {feedback_comment}")

    prompt = (
        "You are a support triage assistant. Based on the conversation below, "
        "write a concise 2-3 sentence summary describing the user's issue and why "
        "they need human support. Be factual and neutral. Do not invent details.\n\n"
        + "\n".join(context_parts)
    )
    try:
        resp = await asyncio.to_thread(
            lambda: get_openai_client().chat.completions.create(
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("escalation_summary_failed: %s", exc)
        return ""


@step
async def handle_raise_ticket(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
) -> QueryResponse:
    """
    Raise a support ticket via Service Bus / Zendesk.

    Idempotency: one ticket per (user, conversation). Duplicate submissions
    return the original reference without re-queuing.
    """
    ticket_idem_id = f"ticket-{user_id}-{conversation_id}"
    existing = await asyncio.to_thread(
        get_document, get_chat_container(), ticket_idem_id, conversation_id
    )
    if existing:
        existing_ref = existing.get("correlation_id", ticket_idem_id)
        logger.info("ticket_duplicate_suppressed ref=%s user_id=%s", existing_ref, user_id)
        return QueryResponse(
            question_id=question_id,
            answer_id=f"ans-{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            user_id=user_id,
            status="ticket_raised",
            answer=(
                f"A ticket was already raised for this conversation. "
                f"Reference: `{existing_ref}`. No duplicate created."
            ),
            domain=domain,
            confidence=1.0,
            attempts_used=0,
            tools_used=[],
            sources=[],
            escalation_options=None,
        )

    if is_escalation_configured():
        try:
            previous_answer, sources, feedback_rating, feedback_comment = \
                await _fetch_escalation_context(user_id, conversation_id, question_id)
            issue_summary = await _generate_issue_summary(
                question_text, previous_answer, feedback_rating, feedback_comment, domain
            )
            correlation_id = await asyncio.to_thread(
                sb_raise_ticket,
                user_id, conversation_id, question_id, question_text, domain,
                None, None,
                previous_answer, sources, feedback_rating, feedback_comment, issue_summary,
            )
            # Persist idempotency record so duplicate clicks are suppressed.
            await asyncio.to_thread(
                upsert_document, get_chat_container(),
                {"id": ticket_idem_id, "correlation_id": correlation_id,
                 "conversation_id": conversation_id, "user_id": user_id},
            )
            answer = (
                f"Your ticket has been raised. Reference: `{correlation_id}`. "
                f"Expected response within **{settings.ESCALATION_SLA_TICKET}**."
            )
            record_escalation(escalation_type="raise_ticket", domain=domain or "unknown")
            logger.info(
                "ticket_queued correlation_id=%s user_id=%s domain=%s",
                correlation_id, user_id, domain,
            )
        except Exception as exc:
            logger.error("ticket_queue_failed user_id=%s: %s", user_id, exc, exc_info=True)
            correlation_id = f"REF-PENDING-{uuid.uuid4().hex[:6].upper()}"
            answer = (
                "Your escalation has been received but could not be queued automatically. "
                "Please contact the support team directly. "
                f"Reference: `{correlation_id}`."
            )
    else:
        logger.error(
            "ticket_queue_skipped: Service Bus not configured. "
            "Set AZURE_SERVICE_BUS_NAMESPACE."
        )
        correlation_id = f"REF-UNCONFIGURED-{uuid.uuid4().hex[:6].upper()}"
        answer = (
            "Escalation is not yet fully configured. "
            "Please contact your support team directly. "
            f"Reference: `{correlation_id}`."
        )

    return QueryResponse(
        question_id=f"q-{uuid.uuid4().hex[:12]}",
        answer_id=f"ans-{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        user_id=user_id,
        status="ticket_raised",
        answer=answer,
        domain=domain,
        confidence=1.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=None,
    )


@step
async def handle_connect_sme(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
) -> QueryResponse:
    """Queue an SME connection request via Service Bus."""
    if is_escalation_configured():
        try:
            previous_answer, sources, feedback_rating, feedback_comment = \
                await _fetch_escalation_context(user_id, conversation_id, question_id)
            issue_summary = await _generate_issue_summary(
                question_text, previous_answer, feedback_rating, feedback_comment, domain
            )
            correlation_id = await asyncio.to_thread(
                sb_connect_sme,
                user_id, conversation_id, question_id, question_text, domain,
                None, None,
                previous_answer, sources, feedback_rating, feedback_comment, issue_summary,
            )
            answer = (
                f"You're being connected with an SME. Reference: `{correlation_id}`. "
                f"Expected response within **{settings.ESCALATION_SLA_SME}**."
            )
            record_escalation(escalation_type="connect_sme", domain=domain or "unknown")
            logger.info(
                "sme_connect_queued correlation_id=%s user_id=%s domain=%s",
                correlation_id, user_id, domain,
            )
        except Exception as exc:
            logger.error("sme_queue_failed user_id=%s: %s", user_id, exc, exc_info=True)
            correlation_id = f"REF-PENDING-{uuid.uuid4().hex[:6].upper()}"
            answer = (
                "Your SME request was received but could not be queued automatically. "
                "Please contact the support team directly. "
                f"Reference: `{correlation_id}`."
            )
    else:
        logger.error(
            "sme_queue_skipped: Service Bus not configured. "
            "Set AZURE_SERVICE_BUS_NAMESPACE."
        )
        correlation_id = f"REF-UNCONFIGURED-{uuid.uuid4().hex[:6].upper()}"
        answer = (
            "SME connection is not yet fully configured. "
            "Please contact your support team directly. "
            f"Reference: `{correlation_id}`."
        )

    return QueryResponse(
        question_id=f"q-{uuid.uuid4().hex[:12]}",
        answer_id=f"ans-{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        user_id=user_id,
        status="sme_connecting",
        answer=answer,
        domain=domain,
        confidence=1.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=None,
    )
