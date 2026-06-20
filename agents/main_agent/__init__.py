"""
Main Agent package.

Exposes the FastAPI ``app`` so the package can be referenced as:
    uvicorn agents.main_agent:app
"""
from agents.main_agent.app import app  # noqa: F401
