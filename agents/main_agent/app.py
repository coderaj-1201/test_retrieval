"""
FastAPI application factory for the Main Agent.

Responsibilities:
  - Creates the FastAPI app with lifespan management.
  - Registers middleware (CORS, payload size limit).
  - Mounts all route modules.
  - Initialises / tears down the shared httpx client.
  - Sets up signal handling and asyncio error logging.

The app is exported from the package __init__.py so it can be run as:
    uvicorn agents.main_agent:app
"""
from __future__ import annotations

import asyncio
import json
import signal
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from shared.config import settings
from shared.cosmos_client import probe_cosmos
from shared.logging_config import configure_logging, get_logger

import agents.main_agent.workflow as _workflow

configure_logging()
logger = get_logger(__name__)


# ── Payload size guard ─────────────────────────────────────────────────────────

class _ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests larger than 1 MB before their body is read into memory."""
    _MAX_BYTES = 1_048_576

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self._MAX_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "payload_too_large", "max_bytes": self._MAX_BYTES},
            )
        return await call_next(request)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: open shared HTTP client, probe Cosmos. Shutdown: drain client."""
    _register_sigterm()
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_asyncio_exception_handler)

    _workflow._http = httpx.AsyncClient(
        timeout=None,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    await asyncio.to_thread(probe_cosmos)
    logger.info("main_agent_started environment=%s", settings.ENVIRONMENT)
    yield
    await _workflow._http.aclose()
    logger.info("main_agent_stopped")


def _register_sigterm():
    def _handler(signum, frame):
        logger.info("main_agent_sigterm_received — draining in-flight requests")
    signal.signal(signal.SIGTERM, _handler)


def _asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    exc = context.get("exception")
    logger.error(
        "asyncio_unhandled_exception msg=%s exc=%s",
        context.get("message"), exc, exc_info=exc,
    )


# ── App assembly ───────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Main Agent", lifespan=lifespan)

app.add_middleware(_ContentSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ask-ops-bot-frontend-e9dfe0aqgfdcg7e3.southcentralus-01.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules.
from agents.main_agent.routes.health import router as health_router
from agents.main_agent.routes.query import router as query_router
from agents.main_agent.routes.feedback import router as feedback_router
from agents.main_agent.routes.history import router as history_router

app.include_router(health_router)
app.include_router(query_router)
app.include_router(feedback_router)
app.include_router(history_router)


if __name__ == "__main__":
    uvicorn.run("agents.main_agent:app", host="0.0.0.0", port=8000, reload=False)
