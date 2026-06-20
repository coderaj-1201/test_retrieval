# Local Run Guide

This branch replaces all Azure dependencies with local equivalents:

| Production | Local replacement |
|---|---|
| Azure OpenAI (chat) | Mistral AI — `mistral-small-latest` |
| Azure OpenAI (embeddings) | Mistral AI — `mistral-embed` |
| Azure AI Search | ChromaDB (persisted to `./local_data/chroma`) |
| Azure Cosmos DB | SQLite (`./local_data/rag.db`) |
| Azure Service Bus / Zendesk | Stdout stub (logs to console) |
| Azure Identity | Not needed |
| Azure Monitor | Disabled |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — the only required value is MISTRAL_API_KEY
```

Get your Mistral API key from https://console.mistral.ai/

### 3. Seed test documents (optional but recommended)

Without documents the retrieval agent returns empty results (low confidence).
Edit `scripts/seed_local_docs.py` to add your own content, then run:

```bash
python scripts/seed_local_docs.py
```

### 4. Start the three servers

Open three terminals (or use a process manager):

```bash
# Terminal 1 — Retrieval agent
uvicorn agents.retrieval_agent:app --port 8002

# Terminal 2 — Orchestrator agent
uvicorn agents.orchestrator_agent:app --port 8001

# Terminal 3 — Main agent
uvicorn agents.main_agent:app --port 8000
```

### 5. Test

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I submit an expense claim?", "user_id": "test-user"}'
```

Health checks:
```bash
curl http://localhost:8000/health/live
curl http://localhost:8001/health/live
curl http://localhost:8002/health/live
```

Run startup smoke test:
```bash
pytest tests/test_server_startup.py -v
```

---

## Architecture notes

- **Mistral embeddings** (`mistral-embed`) produce 1024-dimensional vectors.
  ChromaDB uses cosine distance; we convert to a 0–1 similarity score.
- **ChromaDB** persists to `./local_data/chroma`. Delete this directory to
  reset the index.
- **SQLite** uses a single file (`./local_data/rag.db`) with one table per
  Cosmos container. Delete the file to reset all session/chat/LTM state.
- **CONFIDENCE_THRESHOLD** is set to 0.30 (vs 0.65 in production) because
  local semantic search without BM25 or a reranker scores lower on average.
- The **escalation stub** (`shared/escalation_client.py`) logs escalation
  events to stdout with a `LOCAL_ESCALATION` prefix and returns a fake ref ID.
