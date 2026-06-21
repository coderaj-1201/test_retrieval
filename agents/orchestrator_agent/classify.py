"""
Query classification step for the Orchestrator Agent.

Uses the LLM to decide:
  - Is the query in-domain or out-of-scope?
  - If in-domain: which domain, confidence, tool, and is it a follow-up?
  - If out-of-scope: what response_type (greeting/general/offensive/decline/…)?

The ClassifyResult is consumed by workflow.py to choose the routing path.
"""
from __future__ import annotations

import asyncio
import json

from agent_framework import step

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.logging_config import get_logger
from shared.models import ClassifyInput, Domain, RetrievalTool
from shared.retry import llm_retry
from prompts import CLASSIFY_SYSTEM

logger = get_logger(__name__)


class ClassifyResult:
    """Holds the structured output of classify_query."""

    __slots__ = (
        "domain", "domain_confidence", "secondary_domain", "tool", "failed",
        "out_of_scope", "is_followup", "response_type",
    )

    def __init__(
        self,
        domain: Domain | None,
        domain_confidence: float,
        secondary_domain: Domain | None,
        tool: RetrievalTool,
        failed: bool = False,
        out_of_scope: bool = False,
        is_followup: bool = False,
        response_type: str = "decline",
    ) -> None:
        self.domain = domain
        self.domain_confidence = domain_confidence
        self.secondary_domain = secondary_domain
        self.tool = tool
        self.failed = failed
        self.out_of_scope = out_of_scope
        self.is_followup = is_followup
        self.response_type = response_type


@step
async def classify_query(inp: ClassifyInput) -> ClassifyResult:
    """
    Call the LLM classifier to route the query.

    Args:
        inp: ClassifyInput with query text, session context, and LTM context.

    Returns:
        ClassifyResult — never raises; sets failed=True on LLM errors.
    """
    memory_block = "\n\n".join(filter(None, [inp.ltm_context, inp.session_context]))
    user_content = (
        f"{memory_block}\n\nQuestion: {inp.query}"
        if memory_block else
        f"Question: {inp.query}"
    )

    @llm_retry
    def _call_llm():
        return get_openai_client().chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

    try:
        resp = await asyncio.to_thread(_call_llm)
        raw = json.loads(resp.choices[0].message.content)
        logger.info("raw_classification=%s", raw)
    except json.JSONDecodeError as exc:
        logger.error("classify_json_parse_error query=%.60s exc=%s", inp.query, exc)
        return ClassifyResult(None, 0.0, None, RetrievalTool.HYBRID, failed=True)
    except Exception as exc:
        logger.error("classify_llm_error query=%.60s: %s", inp.query, exc, exc_info=True)
        return ClassifyResult(None, 0.0, None, RetrievalTool.HYBRID, failed=True)

    domain_raw = (raw.get("domain") or "").lower()
    _OUT_OF_SCOPE = {"none", "general", "out_of_scope", "unknown", "other", ""}

    if domain_raw in _OUT_OF_SCOPE:
        response_type = (raw.get("response_type") or "decline").lower()
        logger.info(
            "classify_out_of_scope query_preview=%.60s response_type=%s",
            inp.query, response_type,
        )
        return ClassifyResult(
            None, 0.0, None, RetrievalTool.HYBRID,
            out_of_scope=True,
            response_type=response_type,
        )

    try:
        domain = Domain(domain_raw)
    except ValueError:
        logger.warning("unknown_domain value='%s'", domain_raw)
        return ClassifyResult(None, 0.0, None, RetrievalTool.HYBRID, failed=True)

    try:
        domain_confidence = float(raw.get("domain_confidence", 1.0))
        domain_confidence = max(0.0, min(1.0, domain_confidence))
    except (TypeError, ValueError):
        domain_confidence = 1.0

    secondary_domain: Domain | None = None
    sec_raw = (raw.get("secondary_domain") or "none").lower()
    if sec_raw not in ("none", ""):
        try:
            secondary_domain = Domain(sec_raw)
            if secondary_domain == domain:
                secondary_domain = None
        except ValueError:
            secondary_domain = None

    tool_raw = (raw.get("tool") or "hybrid").lower()
    try:
        tool = RetrievalTool(tool_raw)
    except ValueError:
        logger.warning("unknown_tool value='%s' defaulting=hybrid", tool_raw)
        tool = RetrievalTool.HYBRID

    is_followup = bool(raw.get("is_followup", False))

    logger.info(
        "classify_complete domain=%s confidence=%.2f tool=%s is_followup=%s",
        domain, domain_confidence, tool, is_followup,
    )
    return ClassifyResult(
        domain, domain_confidence, secondary_domain, tool,
        is_followup=is_followup,
        response_type="in_domain",
    )
