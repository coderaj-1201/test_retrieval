"""
Escalation client — local development stub.

In production this sends tickets to Zendesk or Azure Service Bus.
Locally it logs the escalation to stdout and returns a fake reference ID,
so the full escalation flow can be exercised without any external services.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def is_escalation_configured() -> bool:
    """Always True locally — stub is always available."""
    return True


def raise_ticket(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
    user_email: str | None = None,
) -> str:
    """Log a ticket escalation locally and return a fake reference."""
    ref = f"LOCAL-TICKET-{uuid.uuid4().hex[:8].upper()}"
    logger.info(
        "LOCAL_ESCALATION type=raise_ticket ref=%s user_id=%s domain=%s "
        "question_preview=%.60s timestamp=%s",
        ref, user_id, domain, question_text,
        datetime.now(timezone.utc).isoformat(),
    )
    return ref


def connect_sme(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
    user_email: str | None = None,
) -> str:
    """Log an SME connection request locally and return a fake reference."""
    ref = f"LOCAL-SME-{uuid.uuid4().hex[:8].upper()}"
    logger.info(
        "LOCAL_ESCALATION type=connect_sme ref=%s user_id=%s domain=%s "
        "question_preview=%.60s timestamp=%s",
        ref, user_id, domain, question_text,
        datetime.now(timezone.utc).isoformat(),
    )
    return ref
