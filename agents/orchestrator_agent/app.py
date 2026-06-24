"""
FastAPI application factory for the Orchestrator Agent.

Responsibilities:
  - Creates the FastAPI app with lifespan management.
  - Registers InternalAuthMiddleware (validates X-Internal-Secret from Main Agent).
  - Initialises / tears down the shared httpx client used by retrieval.py.
  - Probes Cosmos on startup.
  - Sets up graceful SIGTERM handling.

Run with:
    uvicorn agents.orchestrator_agent:app --port 8001
"""
from __future__ import annotations

import asyncio
import signal
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI

from shared.auth_middleware import InternalAuthMiddleware
from shared.config import settings
from shared.cosmos_client import probe_cosmos
from shared.logging_config import configure_logging, get_logger

import agents.orchestrator_agent.retrieval as _retrieval

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: open shared HTTP client, probe Cosmos. Shutdown: drain client."""
    _register_sigterm()
    _retrieval._http = httpx.AsyncClient(
        timeout=600,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
    await asyncio.to_thread(probe_cosmos)
    logger.info("orchestrator_agent_started environment=%s", settings.ENVIRONMENT)
    yield
    await _retrieval._http.aclose()
    logger.info("orchestrator_agent_stopped")


def _register_sigterm():
    def _handler(signum, frame):
        logger.info("orchestrator_agent_sigterm_received — draining in-flight requests")
    signal.signal(signal.SIGTERM, _handler)


# ── App assembly ───────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Orchestrator Agent", lifespan=lifespan)
app.add_middleware(InternalAuthMiddleware)

from agents.orchestrator_agent.routes import router
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "agents.orchestrator_agent:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        timeout_graceful_shutdown=60,
    )
