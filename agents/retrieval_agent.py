"""
Retrieval Agent
===============
Executes the retrieval tool selected by the Orchestrator, enriches with
parent-chunk context, synthesises an answer and returns a confidence score.

Phase-2 hardening:
  - LLM calls (synthesis, HyDE, decomposition) wrapped with @llm_retry
  - Search calls wrapped with @search_retry (via hybrid_search_tool.py)
  - Parent chunks fetched in parallel with asyncio.gather (was serial)
  - Confidence extracted via json_object response_format (not brittle string parsing)
  - /health/live + /health/ready split for ACA probes
  - InternalAuthMiddleware validates X-Internal-Secret on all non-health paths
  - SIGTERM handler for graceful shutdown
"""
from __future__ import annotations

import asyncio
import json
import re
import signal
import logging
from contextlib import asynccontextmanager

import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from shared.auth_middleware import InternalAuthMiddleware
from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.cosmos_client import probe_cosmos
from shared.logging_config import bind_context, configure_logging, get_logger
from shared.model_router import call_synthesis_llm
from shared.models import (
    Domain, OrchestratorRequest, RetrievalResult, RetrievalStepInput,
    RetrievalTool, SourceDocument, SynthesisInput,
)
from shared.retry import llm_retry
from tools.hybrid_search_tool import SearchDocument, fetch_parent_chunk, hybrid_search
from tools.hyde_tool import generate_hypothetical_document
from tools.query_decomposition_tool import decompose_query
from prompts.synthesis import build_messages as build_synthesis_messages

configure_logging()
logger = get_logger(__name__)


# ── Retrieval steps ────────────────────────────────────────────────────────────

_ENUMERATION_RE = re.compile(
    r"\b(how many|list all|what are all|enumerate|all (the |the policies|documents?|"
    r"sops?|procedures?|runbooks?|policies|guidelines?)|count of|total number)\b",
    re.IGNORECASE,
)

# When an enumeration query is detected, retrieve more chunks so the LLM
# has a broader view of available documents rather than only the top-5.
_ENUMERATION_TOP_K = 20


def _is_enumeration_query(query: str) -> bool:
    return bool(_ENUMERATION_RE.search(query))


@step
async def run_hybrid(inp: RetrievalStepInput) -> list[SearchDocument]:
    try:
        top_k = _ENUMERATION_TOP_K if _is_enumeration_query(inp.query) else None
        docs = await asyncio.to_thread(hybrid_search, inp.query, inp.domain, top_k=top_k)
        logger.info(
            "hybrid_search_complete domain=%s docs=%d enumeration=%s",
            inp.domain, len(docs), top_k is not None,
        )
        return docs
    except Exception as exc:
        logger.error("hybrid_search_error domain=%s: %s", inp.domain, exc, exc_info=True)
        return []


@step
async def run_hyde(inp: RetrievalStepInput) -> list[SearchDocument]:
    try:
        top_k = _ENUMERATION_TOP_K if _is_enumeration_query(inp.query) else None
        hypo = await asyncio.to_thread(generate_hypothetical_document, inp.query)
        logger.debug("hyde_generated length=%d", len(hypo))
        docs = await asyncio.to_thread(hybrid_search, hypo, inp.domain, top_k=top_k)
        logger.info("hyde_search_complete domain=%s docs=%d", inp.domain, len(docs))
        return docs
    except Exception as exc:
        logger.error("hyde_error domain=%s: %s", inp.domain, exc, exc_info=True)
        logger.warning("hyde_fallback_to_hybrid domain=%s", inp.domain)
        try:
            top_k = _ENUMERATION_TOP_K if _is_enumeration_query(inp.query) else None
            return await asyncio.to_thread(hybrid_search, inp.query, inp.domain, top_k=top_k)
        except Exception:
            return []


@step
async def run_decomposition(inp: RetrievalStepInput) -> list[SearchDocument]:
    try:
        sub_queries = await asyncio.to_thread(decompose_query, inp.query)
        logger.info("decomposition_sub_queries count=%d", len(sub_queries))

        # Limit concurrency to avoid bursting Azure OpenAI and Search quotas.
        semaphore = asyncio.Semaphore(2)
        top_k = _ENUMERATION_TOP_K if _is_enumeration_query(inp.query) else None

        async def _bounded_search(sq: str) -> list[SearchDocument]:
            async with semaphore:
                return await asyncio.to_thread(hybrid_search, sq, inp.domain, top_k=top_k)

        result_sets = await asyncio.gather(
            *[_bounded_search(sq) for sq in sub_queries],
            return_exceptions=True,
        )

        seen: dict[str, SearchDocument] = {}
        for i, result in enumerate(result_sets):
            if isinstance(result, Exception):
                logger.error("decomposition_sub_query_failed index=%d: %s", i, result)
                continue
            for doc in result:
                if doc.id not in seen or doc.score > seen[doc.id].score:
                    seen[doc.id] = doc

        cap = top_k or settings.RETRIEVAL_TOP_K
        merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)[:cap]
        logger.info("decomposition_complete domain=%s merged_docs=%d", inp.domain, len(merged))
        return merged
    except Exception as exc:
        logger.error("decomposition_error domain=%s: %s", inp.domain, exc, exc_info=True)
        return []


@step
async def synthesize_answer(inp: SynthesisInput) -> tuple[str, float, list[SourceDocument], bool, list[dict]]:
    query    = inp.query
    all_docs = inp.all_docs

    if not all_docs:
        logger.warning("synthesize_no_docs query_preview=%.60s", query)
        return "I can help you with questions related to Operations. I couldn't find any answers for this query in the available knowledge base.", 0.0, [], False, []

    context_parts = []
    for i, d in enumerate(all_docs):
        heading = getattr(d, "section_heading", "")
        page    = getattr(d, "page_number", 0)
        label   = (
            f"[{i+1}] Source: {d.source}"
            + (f" (p.{page})" if page else "")
            + (f" | {heading}" if heading else "")
        )
        if getattr(d, "chunk_type", "") == "table" and getattr(d, "table_raw", ""):
            context_parts.append(f"{label}\nSummary: {d.content}\nTable:\n{d.table_raw}")
        else:
            context_parts.append(f"{label}\n{d.content}")

    # Apply context budget cap — prevents exceeding the model's context window
    # when a large number of parent + child chunks are assembled.
    max_chars     = settings.SYNTHESIS_MAX_CONTEXT_CHARS
    budget        = max_chars
    capped_parts: list[str] = []
    for part in context_parts:
        if budget <= 0:
            break
        if len(part) > budget:
            capped_parts.append(part[:budget] + "\n[...truncated]")
            budget = 0
        else:
            capped_parts.append(part)
            budget -= len(part)

    if len(capped_parts) < len(context_parts):
        logger.warning(
            "synthesis_context_truncated original_parts=%d included=%d max_chars=%d",
            len(context_parts), len(capped_parts), max_chars,
        )

    context = "\n\n".join(capped_parts)

    session_context = inp.session_context
    ltm_context     = inp.ltm_context
    memory_block    = "\n\n".join(filter(None, [ltm_context, session_context]))

    # Format retrieved docs in the template the few-shot examples expect.
    retrieved_docs = context
    live_query = (
        f"{memory_block}\n\n{query}" if memory_block else query
    )

    synthesis_messages = build_synthesis_messages(live_query, retrieved_docs)

    tool_str = inp.tool.value if isinstance(inp.tool, RetrievalTool) else str(inp.tool)

    try:
        raw_content, model_used = await call_synthesis_llm(
            messages=synthesis_messages,
            query=query,
            tool=tool_str,
            attempt=inp.attempt,
        )
        logger.info("synthesis_model_used=%s", model_used)
    except Exception as exc:
        logger.error("synthesis_llm_error query_preview=%.60s: %s", query, exc, exc_info=True)
        return "Failed to synthesise an answer due to an internal error.", 0.0, [], False, []

    try:
        parsed        = json.loads(raw_content)
        answer        = str(parsed.get("answer", "")).strip()
        confidence    = float(parsed.get("confidence", 0.5))
        confidence    = round(min(max(confidence, 0.0), 1.0), 3)
        llm_citations: list[dict] = parsed.get("citations") or []
        # Extract GAP lines from the thinking scratchpad so the orchestrator
        # can use them to form targeted follow-up queries on retry.
        thinking_text = parsed.get("thinking", "")
        gaps: list[str] = [
            line.strip()
            for line in thinking_text.splitlines()
            if "GAP" in line and line.strip()
        ]
        # show_citations is gated purely on confidence — by the time a query
        # reaches synthesize_answer, the orchestrator has already classified
        # it as a real in-domain question (greetings/out-of-scope never get
        # here), so a second "message_type" re-classification inside this
        # prompt is redundant and was occasionally mislabeling legitimate
        # knowledge questions as "general", incorrectly suppressing citations
        # on otherwise-successful answers.
        show_citations = confidence >= settings.CONFIDENCE_THRESHOLD
        # Keep all citations the LLM returned regardless of show_citations or
        # per-doc confidence — partial/conflicting answers still have real sources
        # and the card renderer will show them under a "Referenced Documents" label.
        # Per-doc confidence scores are preserved for the badge display.
        if not answer:
            raise ValueError("Empty answer field in synthesis response.")
        # Hard cap: truncate at the last complete sentence within the limit
        # so the answer never exceeds the configured character budget.
        max_chars = settings.SYNTHESIS_MAX_ANSWER_CHARS
        if len(answer) > max_chars:
            truncated = answer[:max_chars]
            last_stop = max(truncated.rfind(". "), truncated.rfind(".\n"))
            answer = (truncated[: last_stop + 1] if last_stop > max_chars // 2 else truncated) + "\n\n*[Answer truncated — ask a more specific question for full details.]*"
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "synthesis_parse_error: %s — returning graceful error with default confidence",
            exc,
        )
        answer         = "I was unable to produce a formatted answer. Please try rephrasing your question."
        confidence     = 0.5
        show_citations = False
        llm_citations  = []
        gaps           = []

    # Build SourceDocument list from search results.
    # Deduplicate by title; if we see the same title again with a URL and the
    # previously recorded entry had no URL, upgrade it (parent chunks are fetched
    # after child chunks and tend to carry the doc_url field).
    seen_titles: dict[str, int] = {}   # title -> index in sources list
    sources: list[SourceDocument] = []
    for d in all_docs:
        if len(sources) >= settings.SYNTHESIS_MAX_SOURCES and d.source not in seen_titles:
            break
        url   = getattr(d, "doc_url", "") or ""
        title = d.source
        if title in seen_titles:
            # Upgrade existing entry if it was missing a URL and this doc has one.
            idx = seen_titles[title]
            if url and not sources[idx].url:
                sources[idx] = SourceDocument(
                    title=title,
                    excerpt=sources[idx].excerpt,
                    url=url,
                    relevance=sources[idx].relevance,
                )
            continue
        seen_titles[title] = len(sources)
        sources.append(SourceDocument(
            title=title,
            excerpt=d.content[:200],
            url=url,
            relevance=round(d.score, 3),
        ))

    logger.info(
        "synthesis_complete confidence=%.3f sources=%d show_citations=%s llm_citations=%d gaps=%d",
        confidence, len(sources), show_citations, len(llm_citations), len(gaps),
    )
    return answer, confidence, sources, show_citations, llm_citations, gaps


async def run_retrieval(request: OrchestratorRequest) -> RetrievalResult:
    bind_context(
        agent="retrieval",
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )
    logger.info(
        "retrieval_started attempt=%d domain=%s tool=%s",
        request.attempt, request.domain, request.tool,
    )

    step_inp = RetrievalStepInput(query=request.query, domain=request.domain)
    if request.tool == RetrievalTool.HYDE:
        docs = await run_hyde(step_inp)
    elif request.tool == RetrievalTool.DECOMPOSITION:
        docs = await run_decomposition(step_inp)
    else:
        docs = await run_hybrid(step_inp)

    # Fetch parent chunks in parallel (was serial — each round-trip to AI Search
    # added ~300ms; gathering them concurrently keeps the retrieval tight).
    parent_ids = list({d.parent_id for d in docs if d.parent_id})[:3]
    parent_results = await asyncio.gather(
        *[asyncio.to_thread(fetch_parent_chunk, pid) for pid in parent_ids],
        return_exceptions=True,
    )
    parent_docs: list[SearchDocument] = []
    for pid, result in zip(parent_ids, parent_results):
        if isinstance(result, Exception):
            logger.warning("parent_chunk_fetch_failed parent_id=%s: %s", pid, result)
        elif result is not None:
            parent_docs.append(result)

    child_ids = {d.id for d in docs}
    all_docs  = docs + [p for p in parent_docs if p.id not in child_ids]
    logger.debug(
        "total_docs_for_synthesis count=%d (child=%d parent=%d)",
        len(all_docs), len(docs), len(parent_docs),
    )

    answer, confidence, source_docs, show_citations, llm_citations, gaps = await synthesize_answer(SynthesisInput(
        query=request.query,
        all_docs=all_docs,
        session_context=request.session_context,
        ltm_context=request.ltm_context,
        tool=request.tool,
        attempt=request.attempt,
    ))

    logger.info(
        "retrieval_complete attempt=%d confidence=%.3f passed=%s show_citations=%s",
        request.attempt, confidence, confidence >= settings.CONFIDENCE_THRESHOLD, show_citations,
    )

    return RetrievalResult(
        query=request.query,
        domain=request.domain,
        tool=request.tool,
        attempt=request.attempt,
        answer=answer,
        confidence=confidence,
        sources=[
            {"title": s.title, "excerpt": s.excerpt, "url": s.url, "relevance": s.relevance}
            for s in source_docs
        ],
        show_citations=show_citations,
        citations=llm_citations,
        gaps=gaps,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _register_sigterm()
    await asyncio.to_thread(probe_cosmos)
    logger.info("retrieval_agent_started environment=%s", settings.ENVIRONMENT)
    yield
    logger.info("retrieval_agent_stopped")


retrieval_workflow = workflow(name="retrieval_workflow")(run_retrieval)


def _register_sigterm():
    def _handler(signum, frame):
        logger.info("retrieval_agent_sigterm_received — draining in-flight requests")
    signal.signal(signal.SIGTERM, _handler)


app = FastAPI(title="RAG Retrieval Agent", lifespan=lifespan)
app.add_middleware(InternalAuthMiddleware)


@app.get("/health/live")
async def liveness() -> dict:
    return {"status": "alive", "agent": "retrieval"}


@app.get("/health/ready")
async def readiness() -> Response:
    checks: dict[str, str] = {}
    try:
        from shared.cosmos_client import get_chat_container
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {type(exc).__name__}"

    try:
        await asyncio.to_thread(get_openai_client().models.list)
        checks["openai"] = "ok"
    except Exception as exc:
        checks["openai"] = f"error: {type(exc).__name__}"

    overall_ok = all(v == "ok" for v in checks.values())
    return Response(
        content=json.dumps({
            "status": "ready" if overall_ok else "degraded",
            "agent":  "retrieval",
            "checks": checks,
        }),
        media_type="application/json",
        status_code=200 if overall_ok else 503,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "retrieval"}


@app.post("/retrieve")
async def retrieve(raw: Request) -> Response:
    body = await raw.json()

    domain_val = body.get("domain")
    tool_val   = body.get("tool")

    try:
        domain = Domain(domain_val) if domain_val else Domain.OPS
    except ValueError:
        logger.warning("unknown_domain_in_request value='%s' defaulting=ops", domain_val)
        domain = Domain.OPS

    try:
        tool = RetrievalTool(tool_val) if tool_val else RetrievalTool.HYBRID
    except ValueError:
        logger.warning("unknown_tool_in_request value='%s' defaulting=hybrid", tool_val)
        tool = RetrievalTool.HYBRID

    request = OrchestratorRequest(
        query=body.get("query", ""),
        domain=domain,
        tool=tool,
        attempt=int(body.get("attempt", 1)),
        conversation_id=body.get("conversation_id", ""),
        user_id=body.get("user_id", ""),
        question_id=body.get("question_id", ""),
    )

    bind_context(
        agent="retrieval",
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )

    try:
        result: RetrievalResult = await run_retrieval(request)
    except Exception as exc:
        logger.error("retrieve_endpoint_unhandled_error: %s", exc, exc_info=True)
        result = RetrievalResult(
            query=request.query, domain=request.domain, tool=request.tool,
            attempt=request.attempt, answer="Service error during retrieval.",
            confidence=0.0, sources=[],
            conversation_id=request.conversation_id,
            user_id=request.user_id, question_id=request.question_id,
        )

    return Response(
        content=json.dumps(result.to_dict()),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run(
        "agents.retrieval_agent:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        timeout_graceful_shutdown=60,
    )
