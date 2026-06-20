"""
Health-check routes for the Main Agent.

  GET /health/live   — liveness probe (always 200 while the process is running)
  GET /health/ready  — readiness probe (200 only when Cosmos + OpenAI are reachable)
  GET /health        — alias for /health/ready (backward compat for older probes)

ACA liveness probes kill and restart the container on non-200; readiness probes
remove the instance from load balancing. Keep liveness trivial so a slow
dependency doesn't cause unnecessary restarts.
"""
from __future__ import annotations

import asyncio
import json

import httpx
from fastapi import APIRouter, status
from fastapi.responses import Response

from shared.config import settings
from shared.cosmos_client import get_chat_container
from shared.logging_config import get_logger

# Imported at call time to avoid a circular import with app.py.
# (app.py sets _orchestrator_breaker which health.py needs to read)
from agents.main_agent import workflow as _workflow_module

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health/live")
async def liveness() -> dict:
    """Always 200 while the process is alive — used for ACA liveness probe."""
    return {"status": "alive", "agent": "main"}


@router.get("/health/ready")
async def readiness() -> Response:
    """200 only when Cosmos, OpenAI, and the Orchestrator are all reachable."""
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {type(exc).__name__}"
        overall_ok = False

    try:
        from shared.azure_clients import get_openai_client
        await asyncio.to_thread(get_openai_client().models.list)
        checks["openai"] = "ok"
    except Exception as exc:
        checks["openai"] = f"error: {type(exc).__name__}"
        overall_ok = False

    try:
        client = _workflow_module._http or httpx.AsyncClient(timeout=5.0)
        r = await client.get(f"{str(settings.ORCHESTRATOR_URL).rstrip('/')}/health/live")
        checks["orchestrator"] = "ok" if r.status_code == 200 else f"status={r.status_code}"
        if r.status_code != 200:
            overall_ok = False
    except Exception as exc:
        checks["orchestrator"] = f"error: {type(exc).__name__}"
        overall_ok = False

    cb = _workflow_module._orchestrator_breaker.to_dict()
    checks["orchestrator_circuit"] = cb["state"]
    if cb["state"] == "open":
        overall_ok = False

    http_status = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(
        content=json.dumps({
            "status": "ready" if overall_ok else "degraded",
            "agent": "main",
            "checks": checks,
        }),
        media_type="application/json",
        status_code=http_status,
    )


@router.get("/health")
async def health() -> Response:
    """Alias for /health/ready — kept for backward compatibility."""
    return await readiness()
