"""
Orchestrator Agent package.

Exposes the FastAPI ``app`` so the package can be referenced as:
    uvicorn agents.orchestrator_agent:app
"""
from agents.orchestrator_agent.app import app  # noqa: F401
