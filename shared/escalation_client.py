"""
Escalation client — creates support tickets via Zendesk (primary) or
Azure Service Bus (fallback when Zendesk is not configured).

Priority:
  1. Zendesk configured (ZENDESK_SUBDOMAIN + ZENDESK_API_TOKEN + ZENDESK_USER_EMAIL)
     → ticket created via Zendesk REST API; real Zendesk ticket ID returned.
     ZENDESK_API_TOKEN is a Zendesk-issued token (not Azure); injected into ACA
     from Key Vault via a managed-identity secret reference.
  2. Service Bus configured (AZURE_SERVICE_BUS_NAMESPACE)
     → message enqueued via managed identity; no connection string used.
  3. Neither configured → RuntimeError (surfaced as a 503 to the caller).

The caller always receives a reference string:
  - Zendesk path: "ZD-{ticket_id}"   (real Zendesk ticket number)
  - Service Bus path: "REF-{hex}"     (provisional correlation ID)

Zendesk field mapping:
  raise_ticket → ticket with group_id=ZENDESK_GROUP_ID_TICKET, tag "raise_ticket"
  connect_sme  → ticket with group_id=ZENDESK_GROUP_ID_SME,    tag "connect_sme"
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import httpx

from shared.config import settings

logger = logging.getLogger(__name__)


# ── Zendesk ───────────────────────────────────────────────────────────────────

def _zendesk_configured() -> bool:
    return bool(
        settings.ZENDESK_SUBDOMAIN
        and settings.ZENDESK_API_TOKEN
        and settings.ZENDESK_USER_EMAIL
    )


def _zendesk_create_ticket(
    *,
    subject: str,
    body: str,
    tags: list[str],
    external_id: str,
    group_id: int | None,
    requester_email: str | None = None,
) -> str:
    """
    POST a ticket to Zendesk and return its ID as "ZD-{id}".
    Raises RuntimeError on HTTP errors (caller logs and surfaces to user).
    """
    ticket: dict = {
        "subject":     subject,
        "comment":     {"body": body},
        "tags":        tags,
        "external_id": external_id,
        # Priority normal by default; Zendesk automation can escalate based on tags.
        "priority":    "normal",
    }
    if requester_email:
        ticket["requester"] = {"email": requester_email}
    if group_id:
        ticket["group_id"] = group_id

    subdomain = settings.ZENDESK_SUBDOMAIN
    user_email = settings.ZENDESK_USER_EMAIL
    api_token = settings.ZENDESK_API_TOKEN.get_secret_value()  # type: ignore[union-attr]

    try:
        with httpx.Client(timeout=600) as client:
            resp = client.post(
                f"https://{subdomain}.zendesk.com/api/v2/tickets",
                auth=(f"{user_email}/token", api_token),
                json={"ticket": ticket},
                headers={"Content-Type": "application/json"},
            )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Zendesk API error {exc.response.status_code}: "
            f"{exc.response.text[:300]}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Zendesk request failed: {exc}") from exc

    ticket_id = resp.json()["ticket"]["id"]
    return f"ZD-{ticket_id}"


# ── Service Bus (fallback) ─────────────────────────────────────────────────────

def _sb_configured() -> bool:
    return bool(settings.AZURE_SERVICE_BUS_NAMESPACE)


def _sb_get_sender():
    from azure.servicebus import ServiceBusClient  # type: ignore[import-untyped]
    from azure.identity import DefaultAzureCredential

    sb_client = ServiceBusClient(
        fully_qualified_namespace = settings.AZURE_SERVICE_BUS_NAMESPACE,
        credential                = DefaultAzureCredential(),
    )
    return sb_client.get_queue_sender(queue_name=settings.SB_QUEUE_ESCALATION)


def _sb_send(escalation_type: str, payload: dict, correlation_id: str) -> None:
    from azure.servicebus import ServiceBusMessage  # type: ignore[import-untyped]

    sender = _sb_get_sender()
    with sender:
        msg = ServiceBusMessage(
            body=json.dumps(payload),
            content_type="application/json",
            subject=escalation_type,
            message_id=correlation_id,
            session_id=payload.get("user_id", ""),
        )
        try:
            sender.send_messages(msg)
        except Exception as send_exc:
            logger.error(
                "escalation_sb_send_failed type=%s correlation_id=%s "
                "user_id=%s domain=%s payload=%s",
                escalation_type,
                correlation_id,
                payload.get("user_id"),
                payload.get("domain"),
                json.dumps(payload),
                exc_info=True,
            )
            raise


# ── Ticket body builder ────────────────────────────────────────────────────────

def _build_ticket_body(
    *,
    escalation_type: str,
    domain: str,
    question_text: str,
    conversation_id: str,
    user_id: str,
    question_id: str,
    timestamp: str,
    previous_answer: str | None,
    sources: list[dict] | None,
    feedback_rating: str | None,
    feedback_comment: str | None,
    issue_summary: str | None,
    footer: str = "",
) -> str:
    lines: list[str] = [
        f"=== {escalation_type} ===",
        "",
    ]

    if issue_summary:
        lines += ["--- Issue Summary (AI-generated) ---", issue_summary, ""]

    lines += [
        "--- User Question ---",
        question_text,
        "",
        f"Domain:          {domain}",
        f"User ID:         {user_id}",
        f"Conversation ID: {conversation_id}",
        f"Question ID:     {question_id}",
        f"Timestamp:       {timestamp}",
        "",
    ]

    if previous_answer:
        lines += [
            "--- Bot's Previous Answer ---",
            previous_answer[:2000] + ("..." if len(previous_answer) > 2000 else ""),
            "",
        ]

    if sources:
        lines.append("--- Sources Referenced ---")
        for i, s in enumerate(sources[:5], 1):
            title = s.get("title") or s.get("doc_name") or s.get("source") or "Unknown"
            url   = s.get("url") or s.get("doc_url") or ""
            lines.append(f"  {i}. {title}" + (f" — {url}" if url else ""))
        lines.append("")

    if feedback_rating or feedback_comment:
        lines.append("--- User Feedback ---")
        if feedback_rating:
            lines.append(f"  Rating:  {feedback_rating}")
        if feedback_comment:
            lines.append(f"  Comment: {feedback_comment}")
        lines.append("")

    if footer:
        lines += [footer, ""]

    return "\n".join(lines)


# ── Public interface ───────────────────────────────────────────────────────────

def is_escalation_configured() -> bool:
    """True if at least one escalation channel is ready."""
    return _zendesk_configured() or _sb_configured()


def raise_ticket(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
    user_email: str | None = None,
    previous_answer: str | None = None,
    sources: list[dict] | None = None,
    feedback_rating: str | None = None,
    feedback_comment: str | None = None,
    issue_summary: str | None = None,
) -> str:
    """
    Create a support ticket.
    Returns a reference string: "ZD-{id}" via Zendesk, "REF-{hex}" via SB.
    Raises RuntimeError if no escalation channel is configured.
    """
    correlation_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    if _zendesk_configured():
        subject = f"[{domain.upper()}] Support request from Teams — {question_text[:60]}"
        body = _build_ticket_body(
            escalation_type="Support Request",
            domain=domain,
            question_text=question_text,
            conversation_id=conversation_id,
            user_id=user_id,
            question_id=question_id,
            timestamp=timestamp,
            previous_answer=previous_answer,
            sources=sources,
            feedback_rating=feedback_rating,
            feedback_comment=feedback_comment,
            issue_summary=issue_summary,
        )
        try:
            ref = _zendesk_create_ticket(
                subject=subject,
                body=body,
                tags=["rag-bot", f"domain:{domain.lower()}", "raise_ticket"],
                external_id=correlation_id,
                group_id=settings.ZENDESK_GROUP_ID_TICKET,
                requester_email=user_email,
            )
            logger.info(
                "escalation_ticket_created via=zendesk ref=%s user_id=%s domain=%s",
                ref, user_id, domain,
            )
            return ref
        except Exception as exc:
            logger.error(
                "zendesk_ticket_failed correlation_id=%s user_id=%s domain=%s: %s",
                correlation_id, user_id, domain, exc, exc_info=True,
            )
            if not _sb_configured():
                raise RuntimeError(f"Ticket creation failed and no fallback configured: {exc}") from exc
            logger.warning("escalation_fallback_to_servicebus correlation_id=%s", correlation_id)

    if _sb_configured():
        payload = {
            "type":                   "raise_ticket",
            "correlation_id":         correlation_id,
            "user_id":                user_id,
            "conversation_id":        conversation_id,
            "question_id":            question_id,
            "question_text":          question_text[:1000],
            "domain":                 domain,
            "timestamp":              timestamp,
            "conversation_reference": conversation_reference or {},
            "previous_answer":        (previous_answer or "")[:2000],
            "sources":                sources or [],
            "feedback_rating":        feedback_rating or "",
            "feedback_comment":       feedback_comment or "",
            "issue_summary":          issue_summary or "",
        }
        _sb_send("raise_ticket", payload, correlation_id)
        logger.info(
            "escalation_ticket_queued via=service_bus correlation_id=%s user_id=%s domain=%s",
            correlation_id, user_id, domain,
        )
        return correlation_id

    raise RuntimeError(
        "No escalation channel configured. "
        "Set ZENDESK_SUBDOMAIN + ZENDESK_API_TOKEN + ZENDESK_USER_EMAIL "
        "or set AZURE_SERVICE_BUS_NAMESPACE."
    )


def connect_sme(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
    user_email: str | None = None,
    previous_answer: str | None = None,
    sources: list[dict] | None = None,
    feedback_rating: str | None = None,
    feedback_comment: str | None = None,
    issue_summary: str | None = None,
) -> str:
    """
    Request SME connection via a Zendesk ticket (or Service Bus fallback).
    Returns a reference string.
    Raises RuntimeError if no escalation channel is configured.
    """
    correlation_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    if _zendesk_configured():
        subject = f"[{domain.upper()}] SME connection request — {question_text[:60]}"
        body = _build_ticket_body(
            escalation_type="SME Connection Request",
            domain=domain,
            question_text=question_text,
            conversation_id=conversation_id,
            user_id=user_id,
            question_id=question_id,
            timestamp=timestamp,
            previous_answer=previous_answer,
            sources=sources,
            feedback_rating=feedback_rating,
            feedback_comment=feedback_comment,
            issue_summary=issue_summary,
            footer="Please connect the user with a subject matter expert for this query.",
        )
        try:
            ref = _zendesk_create_ticket(
                subject=subject,
                body=body,
                tags=["rag-bot", f"domain:{domain.lower()}", "connect_sme", "sme-request"],
                external_id=correlation_id,
                group_id=settings.ZENDESK_GROUP_ID_SME,
                requester_email=user_email,
            )
            logger.info(
                "escalation_sme_created via=zendesk ref=%s user_id=%s domain=%s",
                ref, user_id, domain,
            )
            return ref
        except Exception as exc:
            logger.error(
                "zendesk_sme_failed correlation_id=%s user_id=%s domain=%s: %s",
                correlation_id, user_id, domain, exc, exc_info=True,
            )
            if not _sb_configured():
                raise RuntimeError(f"SME connection failed and no fallback configured: {exc}") from exc
            logger.warning("escalation_fallback_to_servicebus correlation_id=%s", correlation_id)

    if _sb_configured():
        payload = {
            "type":                   "connect_sme",
            "correlation_id":         correlation_id,
            "user_id":                user_id,
            "conversation_id":        conversation_id,
            "question_id":            question_id,
            "question_text":          question_text[:1000],
            "domain":                 domain,
            "timestamp":              timestamp,
            "conversation_reference": conversation_reference or {},
            "previous_answer":        (previous_answer or "")[:2000],
            "sources":                sources or [],
            "feedback_rating":        feedback_rating or "",
            "feedback_comment":       feedback_comment or "",
            "issue_summary":          issue_summary or "",
        }
        _sb_send("connect_sme", payload, correlation_id)
        logger.info(
            "escalation_sme_queued via=service_bus correlation_id=%s user_id=%s domain=%s",
            correlation_id, user_id, domain,
        )
        return correlation_id

    raise RuntimeError(
        "No escalation channel configured. "
        "Set ZENDESK_SUBDOMAIN + ZENDESK_API_TOKEN + ZENDESK_USER_EMAIL "
        "or set AZURE_SERVICE_BUS_NAMESPACE."
    )
