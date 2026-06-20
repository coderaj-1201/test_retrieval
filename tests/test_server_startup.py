"""
Server startup smoke test.

Verifies that all three agent servers start without import or config errors
and respond 200 on their /health/live endpoints.

Prerequisites:
  - Install requirements: pip install -r requirements.txt
  - All required env vars present (uses the dummy .env written by this module).

Usage:
  pytest tests/test_server_startup.py -v
  # or run directly:
  python tests/test_server_startup.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import httpx
import pytest

# ── Dummy .env content ────────────────────────────────────────────────────────
# Uses local stub URLs so the servers can load config without real Azure creds.
# Cosmos/OpenAI/Search calls are NOT made during startup — only config parsing
# and FastAPI lifespan probes run.  The liveness probe (/health/live) never
# touches external services, so these dummies are sufficient.

_DUMMY_ENV = textwrap.dedent("""\
    ENVIRONMENT=development

    AZURE_FOUNDRY_PROJECT_ENDPOINT=https://dummy.api.azureml.ms
    AZURE_OPENAI_ENDPOINT=https://dummy.openai.azure.com/
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-41-mini
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
    AZURE_OPENAI_API_VERSION=2025-01-01-preview

    AZURE_SEARCH_ENDPOINT=https://dummy.search.windows.net
    AZURE_SEARCH_INDEX=idx-rag
    AZURE_SEARCH_SEMANTIC_CONFIG=rag-semantic-config

    COSMOS_ENDPOINT=https://dummy.documents.azure.com:443/
    COSMOS_DATABASE=dummy-db
    COSMOS_CONTAINER_CHAT=chat-history
    COSMOS_CONTAINER_FEEDBACK=feedback
    COSMOS_CONTAINER_SESSIONS=sessions
    COSMOS_CONTAINER_LTM=long-term-memory

    MAIN_AGENT_URL=http://localhost:8000
    ORCHESTRATOR_URL=http://localhost:8001
    RETRIEVAL_URL=http://localhost:8002

    INTERNAL_API_SECRET=

    MICROSOFT_APP_ID=dummy-app-id
    MICROSOFT_APP_PASSWORD=
    MICROSOFT_APP_TYPE=MultiTenant
    MICROSOFT_APP_TENANT_ID=dummy-tenant-id
    BOT_PORT=3978

    AZURE_SERVICE_BUS_NAMESPACE=
    SB_QUEUE_ESCALATION=escalation-requests

    ZENDESK_SUBDOMAIN=
    ZENDESK_API_TOKEN=
    ZENDESK_USER_EMAIL=
    ZENDESK_GROUP_ID_TICKET=
    ZENDESK_GROUP_ID_SME=

    CONFIDENCE_THRESHOLD=0.65
    CITATION_CONFIDENCE_THRESHOLD=0.40
    MAX_RETRIEVAL_ATTEMPTS=3
    RETRIEVAL_TOP_K=5
    SYNTHESIS_TEMPERATURE=0.0
    MAX_QUERY_LENGTH=2000
    SYNTHESIS_MAX_CONTEXT_CHARS=12000
    SYNTHESIS_MAX_SOURCES=5
    SYNTHESIS_MAX_ANSWER_CHARS=10000
    SYNTHESIS_MAX_TOKENS=6000

    SESSION_MAX_TURNS=10
    SESSION_TTL_SECONDS=604800
    LTM_SUMMARY_EVERY_N=5
    LTM_MAX_SUMMARY_CHARS=3000
    LTM_MAX_FACTS=10

    REDIS_URL=
    RATE_LIMIT_RPM=20
    RATE_LIMIT_BURST=5

    DOMAIN_CONFIDENCE_THRESHOLD=0.6
    ESCALATION_SLA_TICKET=4 business hours
    ESCALATION_SLA_SME=2 business hours

    APPLICATIONINSIGHTS_CONNECTION_STRING=
    LOG_LEVEL=INFO
""")

# ── Server definitions ─────────────────────────────────────────────────────────

_SERVERS = [
    {
        "name":   "main_agent",
        "module": "agents.main_agent:app",
        "port":   8000,
    },
    {
        "name":   "orchestrator_agent",
        "module": "agents.orchestrator_agent:app",
        "port":   8001,
    },
    {
        "name":   "retrieval_agent",
        "module": "agents.retrieval_agent:app",
        "port":   8002,
    },
]


def _repo_root() -> Path:
    return Path(__file__).parent.parent


def _wait_for_server(port: int, timeout: float = 20.0) -> bool:
    """Poll /health/live until it returns 200 or timeout expires."""
    url = f"http://localhost:{port}/health/live"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def dummy_env_file(tmp_path_factory):
    """Write a dummy .env to a temp directory and return its path."""
    d = tmp_path_factory.mktemp("env")
    env_path = d / ".env"
    env_path.write_text(_DUMMY_ENV)
    return env_path


@pytest.fixture(scope="module")
def running_servers(dummy_env_file):
    """
    Start all three agent servers as subprocesses.

    Each server is launched with the dummy .env via the DOT_ENV_FILE
    override path so it doesn't accidentally read a real .env in the repo root.
    Yields a list of (name, port, process) tuples. Terminates all on teardown.
    """
    repo = _repo_root()
    env = {
        **os.environ,
        # Override the working directory so pydantic-settings picks up our dummy .env
        # (pydantic-settings reads .env relative to CWD by default).
    }

    procs = []
    for srv in _SERVERS:
        cmd = [
            sys.executable, "-m", "uvicorn",
            srv["module"],
            "--host", "127.0.0.1",
            "--port", str(srv["port"]),
            "--no-access-log",
        ]
        # Run from the repo root with the dummy .env placed there temporarily.
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env={**env, "PYTHONPATH": str(repo)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append((srv["name"], srv["port"], proc))

    # Give all three servers a chance to start concurrently.
    yield procs

    for name, port, proc in procs:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.parametrize("srv", _SERVERS, ids=[s["name"] for s in _SERVERS])
def test_server_liveness(running_servers, srv):
    """Each server must respond 200 on /health/live within 20 seconds of startup."""
    ok = _wait_for_server(srv["port"], timeout=20.0)
    assert ok, (
        f"{srv['name']} did not respond on port {srv['port']} within 20 s. "
        "Check that the server started without import errors."
    )

    r = httpx.get(f"http://localhost:{srv['port']}/health/live", timeout=5.0)
    assert r.status_code == 200, f"{srv['name']} /health/live returned {r.status_code}"
    body = r.json()
    assert body.get("status") == "alive", f"Unexpected body: {body}"


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick manual run without pytest — prints PASS/FAIL per server."""
    import contextlib

    root = _repo_root()

    # Write dummy .env next to the repo root temporarily.
    env_path = root / ".env.test"
    env_path.write_text(_DUMMY_ENV)

    env = {
        **os.environ,
        "PYTHONPATH": str(root),
    }

    procs = []
    for srv in _SERVERS:
        cmd = [
            sys.executable, "-m", "uvicorn",
            srv["module"],
            "--host", "127.0.0.1",
            "--port", str(srv["port"]),
            "--no-access-log",
        ]
        proc = subprocess.Popen(
            cmd, cwd=str(root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        procs.append((srv["name"], srv["port"], proc))
        print(f"  Started {srv['name']} (pid={proc.pid}) on :{srv['port']}")

    print("\nWaiting for servers...")
    results = {}
    for name, port, proc in procs:
        ok = _wait_for_server(port, timeout=20.0)
        results[name] = ok
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}  (:{ port})")

    # Teardown
    for name, port, proc in procs:
        proc.terminate()
    env_path.unlink(missing_ok=True)

    if all(results.values()):
        print("\nAll servers healthy. ✓")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nFailed: {failed}")
        sys.exit(1)
