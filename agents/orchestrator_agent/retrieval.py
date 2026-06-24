"""
Retrieval layer for the Orchestrator Agent.

Wraps HTTP calls to the Retrieval Agent behind:
  - A circuit breaker (_retrieval_breaker) to stop cascading failures.
  - A @step decorator so the MAF framework tracks the call.
  - A safe variant (_call_retrieval_safe) used in cross-domain fanout.
  - A merge helper (_merge_retrieval_results) that deduplicates sources.

The module-level _http client is set by app.py during lifespan startup.
"""
from __future__ import annotations

import dataclasses

import httpx
from agent_framework import step

from shared.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.config import settings
from shared.logging_config import get_logger
from shared.models import Domain, OrchestratorRequest, RetrievalResult, RetrievalTool

logger = get_logger(__name__)

# Shared HTTP client — injected by app.py lifespan.
_http: httpx.AsyncClient | None = None

_retrieval_breaker = CircuitBreaker(name="retrieval-agent", fail_max=3, reset_timeout=30)


def _internal_headers() -> dict[str, str]:
    """Return the X-Internal-Secret header if a secret is configured."""
    from shared.config import settings as _s
    secret = (
        _s.INTERNAL_API_SECRET.get_secret_value()
        if _s.INTERNAL_API_SECRET is not None
        else None
    )
    return {"X-Internal-Secret": secret} if secret else {}


async def _call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
    """Raw HTTP POST to /retrieve on the Retrieval Agent."""
    global _http
    payload = {
        "query":           req.query,
        "domain":          req.domain.value,
        "tool":            req.tool.value,
        "attempt":         req.attempt,
        "conversation_id": req.conversation_id,
        "user_id":         req.user_id,
        "question_id":     req.question_id,
    }
    client = _http or httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=180.0))
    headers = {**_internal_headers(), "X-Request-ID": req.question_id}
    resp = await client.post(
        f"{str(settings.RETRIEVAL_URL).rstrip('/')}/retrieve",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    domain_val = data.get("domain", "")
    try:
        domain = Domain(domain_val) if domain_val else req.domain
    except ValueError:
        domain = req.domain

    tool_val = data.get("tool", "")
    try:
        tool = RetrievalTool(tool_val) if tool_val else req.tool
    except ValueError:
        tool = req.tool

    return RetrievalResult(
        query=data.get("query", req.query),
        domain=domain,
        tool=tool,
        attempt=data.get("attempt", req.attempt),
        answer=data.get("answer", ""),
        confidence=float(data.get("confidence", 0.0)),
        sources=data.get("sources", []),
        conversation_id=data.get("conversation_id", req.conversation_id),
        user_id=data.get("user_id", req.user_id),
        question_id=data.get("question_id", req.question_id),
        show_citations=bool(data.get("show_citations", False)),
        citations=data.get("citations", []),
    )


@step
async def call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
    """
    Circuit-breaker-protected retrieval call.

    Logs and re-raises all exceptions so the caller can decide how to handle
    (retry, fanout, or surface an error response).
    """
    try:
        return await _retrieval_breaker.call(_call_retrieval, req)
    except CircuitOpenError as exc:
        logger.error(
            "retrieval_circuit_open attempt=%d domain=%s retry_after=%.1f",
            req.attempt, req.domain, exc.retry_after,
        )
        raise
    except httpx.TimeoutException:
        logger.error(
            "retrieval_timeout attempt=%d domain=%s tool=%s",
            req.attempt, req.domain, req.tool,
        )
        raise
    except httpx.HTTPStatusError as exc:
        logger.error(
            "retrieval_http_error status=%d attempt=%d",
            exc.response.status_code, req.attempt,
        )
        raise
    except Exception as exc:
        logger.error(
            "retrieval_unexpected_error attempt=%d: %s", req.attempt, exc, exc_info=True
        )
        raise


async def _call_retrieval_safe(req: OrchestratorRequest) -> RetrievalResult | None:
    """Non-raising wrapper used for cross-domain fanout — returns None on failure."""
    try:
        return await call_retrieval(req)
    except Exception as exc:
        logger.error("retrieval_fanout_failed domain=%s: %s", req.domain, exc)
        return None


def _merge_retrieval_results(
    primary: RetrievalResult, secondary: RetrievalResult | None
) -> RetrievalResult:
    """
    Merge two RetrievalResults from a cross-domain fanout.

    Picks the higher-confidence result as the base and deduplicates sources
    by title, capping at 8 total.
    """
    if secondary is None:
        return primary

    base  = primary if primary.confidence >= secondary.confidence else secondary
    other = secondary if base is primary else primary

    seen_titles: set[str] = set()
    merged: list[dict] = []
    for src in sorted(
        base.sources + other.sources,
        key=lambda s: s.get("relevance", 0.0),
        reverse=True,
    ):
        t = src.get("title", "")
        if t not in seen_titles:
            seen_titles.add(t)
            merged.append(src)

    logger.info(
        "fanout_merge primary_conf=%.3f secondary_conf=%.3f merged_sources=%d",
        primary.confidence, secondary.confidence, len(merged),
    )
    return dataclasses.replace(
        base,
        sources=merged[:8],
        confidence=max(primary.confidence, secondary.confidence),
    )
