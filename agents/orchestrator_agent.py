"""
Orchestrator Agent
==================
Classifies every incoming query and routes it through one of five paths:

  1. domain=none / offensive      → firm equal-energy decline (no retrieval)
  2. domain=none / greeting       → warm brief reply (streak NOT incremented)
  3. domain=none / general        → brief capability reply
  4. domain=none / decision_making→ acknowledge intent, info-only boundary
  5. domain=none / clarify        → ask to rephrase
  6. domain=none / decline        → polite redirect (streak reminder if >= 3)
  7. in-domain + reformat command → condense LATEST answer only (no retrieval)
  8. in-domain + whole-chat ask   → summarize ALL session turns (explicit count)
  9. in-domain + normal           → rewrite if follow-up → AI Search → synthesize

Phase-2 hardening retained:
  - LLM classification wrapped with @llm_retry
  - Shared httpx.AsyncClient reused across requests
  - /health/live + /health/ready split for ACA probes
  - H-9: classification failure returns error response
  - InternalAuthMiddleware validates incoming requests from Main Agent
  - X-Internal-Secret header on all outbound Retrieval calls
  - CircuitBreaker on Retrieval agent calls
  - SIGTERM handler for graceful shutdown
"""
from __future__ import annotations

import asyncio
import json
import signal
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
try:
    from agent_framework import step, workflow
except Exception:
    from retrieval_pipeline.agent_framework import step, workflow
from fastapi import FastAPI, Request, Response

from shared.auth_middleware import InternalAuthMiddleware
from shared.azure_clients import get_openai_client
from shared.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.config import settings
from shared.cosmos_client import probe_cosmos
from shared.logging_config import bind_context, configure_logging, get_logger
from shared.models import (
    ClassifyInput, DOMAIN_DESCRIPTIONS, Domain, FinalResponse, OrchestratorInput,
    OrchestratorRequest, RetrievalResult, RetrievalTool, SessionMemory, UserQuery,
)
from shared.retry import llm_retry
from shared.telemetry import record_tool
from prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_FALLBACKS,
    STREAK_REMINDER,
    STREAK_REMINDER_FIRM,
    REWRITE_SYSTEM,
    REFORMAT_SYSTEM,
    REFORMAT_VERBS,
    WHOLE_CHAT_SUMMARY_SYSTEM,
    WHOLE_CHAT_PHRASES,
)

configure_logging()
logger = get_logger(__name__)

_TOOL_LADDER = [RetrievalTool.HYBRID, RetrievalTool.HYDE, RetrievalTool.DECOMPOSITION]
_ALL_DOMAINS = list(Domain)

_http: httpx.AsyncClient | None = None
_retrieval_breaker = CircuitBreaker(name="retrieval-agent", fail_max=3, reset_timeout=30)

# Response types that count as out-of-scope and increment the streak.
_STREAK_INCREMENTING_TYPES = {"general", "decision_making", "offensive", "decline"}
# Response types that are fine and do NOT touch the streak.
_STREAK_EXEMPT_TYPES = {"greeting", "clarify"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_reformat_command(text: str) -> bool:
    t = text.strip().lower()
    return any(phrase in t for phrase in REFORMAT_VERBS)


def _is_whole_chat_summary(text: str) -> bool:
    t = text.strip().lower()
    return any(phrase in t for phrase in WHOLE_CHAT_PHRASES)


def _apply_streak_reminder(message: str, streak: int) -> str:
    """Append a purpose reminder to deflection messages when streak is high."""
    if streak >= 6:
        return message + STREAK_REMINDER_FIRM
    if streak >= 3:
        return message + STREAK_REMINDER
    return message


def _internal_headers() -> dict[str, str]:
    secret = (
        settings.INTERNAL_API_SECRET.get_secret_value()
        if settings.INTERNAL_API_SECRET is not None
        else None
    )
    return {"X-Internal-Secret": secret} if secret else {}


# ── Classification result ─────────────────────────────────────────────────────

class ClassifyResult:
    __slots__ = (
        "domain", "domain_confidence", "secondary_domain", "tool", "failed",
        "out_of_scope", "deflection_message", "is_followup", "response_type",
    )

    def __init__(
        self,
        domain: Domain | None,
        domain_confidence: float,
        secondary_domain: Domain | None,
        tool: RetrievalTool,
        failed: bool = False,
        out_of_scope: bool = False,
        deflection_message: str = "",
        is_followup: bool = False,
        response_type: str = "decline",
    ) -> None:
        self.domain              = domain
        self.domain_confidence   = domain_confidence
        self.secondary_domain    = secondary_domain
        self.tool                = tool
        self.failed              = failed
        self.out_of_scope        = out_of_scope
        self.deflection_message  = deflection_message
        self.is_followup         = is_followup
        self.response_type       = response_type


# ── Classify step ─────────────────────────────────────────────────────────────

@step
async def classify_query(inp: ClassifyInput) -> ClassifyResult:
    memory_block = "\n\n".join(filter(None, [inp.ltm_context, inp.session_context]))
    user_content = (
        f"{memory_block}\n\nQuestion: {inp.query}"
        if memory_block else
        f"Question: {inp.query}"
    )

    @llm_retry
    def _call_llm():
        return get_openai_client().chat.completions.create(
            model           = settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages        = [
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            temperature     = 0,
            max_tokens      = 300,
            response_format = {"type": "json_object"},
        )

    try:
        resp = await asyncio.to_thread(_call_llm)
        raw  = json.loads(resp.choices[0].message.content)
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
        response_type      = (raw.get("response_type") or "decline").lower()
        deflection_message = str(raw.get("deflection_message") or "").strip()
        if not deflection_message:
            deflection_message = CLASSIFY_FALLBACKS.get(
                response_type, CLASSIFY_FALLBACKS["decline"]
            )
        logger.info(
            "classify_out_of_scope query_preview=%.60s response_type=%s",
            inp.query, response_type,
        )
        return ClassifyResult(
            None, 0.0, None, RetrievalTool.HYBRID,
            failed        = False,
            out_of_scope  = True,
            deflection_message = deflection_message,
            response_type = response_type,
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
        "classify_complete domain=%s confidence=%.2f secondary=%s tool=%s "
        "is_followup=%s reason='%s'",
        domain, domain_confidence, secondary_domain or "none", tool,
        is_followup, raw.get("reason", ""),
    )
    return ClassifyResult(
        domain, domain_confidence, secondary_domain, tool,
        is_followup   = is_followup,
        response_type = "in_domain",
    )


# ── LLM action helpers ────────────────────────────────────────────────────────

async def _reformat_prior_answer(instruction: str, session_context: str) -> str:
    """
    Condense or reformat ONLY the most recent answer using the user's instruction.
    Bypasses retrieval entirely — the prior answer is already in session_context.
    """
    @llm_retry
    def _call():
        return get_openai_client().chat.completions.create(
            model    = settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages = [
                {"role": "system", "content": REFORMAT_SYSTEM},
                {"role": "user",   "content": f"{session_context}\n\nInstruction: {instruction}"},
            ],
            temperature = 0,
            max_tokens  = 500,
        )

    try:
        resp = await asyncio.to_thread(_call)
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning(
            "reformat_prior_answer_failed instruction=%.60s: %s", instruction, exc
        )
        return ""


async def _summarize_whole_chat(
    session: SessionMemory,
    turn_texts: dict[str, dict[str, str]],
) -> str:
    """
    Summarize ALL turns held in the session context window.
    Returns a message that leads with an explicit count statement so the user
    knows exactly how many questions were covered.

    turn_texts is keyed by question_id and contains {"question": ..., "answer": ...}.
    Fetched by main_agent before calling the orchestrator and passed via OrchestratorInput.
    """

    available_turns = [
        t for t in session.turns
        if t.question_id in turn_texts
    ]
    count = len(available_turns)

    if count == 0:
        return (
            "I don't have any previous questions from this session on record to summarize."
        )

    numbered = "\n\n".join(
        f"{i+1}. Q: {turn_texts[t.question_id]['question']}\n"
        f"   A: {turn_texts[t.question_id]['answer']}"
        for i, t in enumerate(available_turns)
    )

    if count < settings.SESSION_MAX_TURNS:
        preamble = (
            f"Summarizing {count} question(s) from this session "
            f"({count} on record, fewer than the {settings.SESSION_MAX_TURNS} maximum):"
        )
    else:
        preamble = f"Summarizing your last {count} questions from this session:"

    @llm_retry
    def _call():
        return get_openai_client().chat.completions.create(
            model    = settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages = [
                {"role": "system", "content": WHOLE_CHAT_SUMMARY_SYSTEM},
                {"role": "user",   "content": numbered},
            ],
            temperature = 0,
            max_tokens  = 600,
        )

    try:
        resp    = await asyncio.to_thread(_call)
        summary = resp.choices[0].message.content.strip()
        return f"{preamble}\n\n{summary}"
    except Exception as exc:
        logger.warning("whole_chat_summary_failed: %s", exc)
        return (
            f"{preamble}\n\n"
            "I wasn't able to generate a summary right now. Please try again."
        )


async def _rewrite_query_if_needed(
    query: str, session_context: str, is_followup: bool
) -> str:
    """Rewrite a follow-up query into a standalone search string."""
    if not is_followup or not session_context:
        return query

    @llm_retry
    def _call():
        return get_openai_client().chat.completions.create(
            model    = settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages = [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user",   "content": f"{session_context}\n\nFollow-up question: {query}"},
            ],
            temperature = 0,
            max_tokens  = 120,
        )

    try:
        resp      = await asyncio.to_thread(_call)
        rewritten = resp.choices[0].message.content.strip().strip('"')
        if rewritten:
            logger.info(
                "query_rewritten original=%.60s rewritten=%.60s", query, rewritten
            )
            return rewritten
    except Exception as exc:
        logger.warning("query_rewrite_failed query=%.60s: %s", query, exc)
    return query


# ── Retrieval helpers ─────────────────────────────────────────────────────────

async def _call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
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
    client  = _http or httpx.AsyncClient(timeout=60.0)
    headers = {**_internal_headers(), "X-Request-ID": req.question_id}
    resp    = await client.post(
        f"{str(settings.RETRIEVAL_URL).rstrip('/')}/retrieve", json=payload, headers=headers
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
        query           = data.get("query", req.query),
        domain          = domain,
        tool            = tool,
        attempt         = data.get("attempt", req.attempt),
        answer          = data.get("answer", ""),
        confidence      = float(data.get("confidence", 0.0)),
        sources         = data.get("sources", []),
        conversation_id = data.get("conversation_id", req.conversation_id),
        user_id         = data.get("user_id", req.user_id),
        question_id     = data.get("question_id", req.question_id),
        show_citations  = bool(data.get("show_citations", False)),
        citations       = data.get("citations", []),
    )


@step
async def call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
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
    try:
        return await call_retrieval(req)
    except Exception as exc:
        logger.error("retrieval_fanout_failed domain=%s: %s", req.domain, exc)
        return None


def _merge_retrieval_results(
    primary: RetrievalResult, secondary: RetrievalResult | None
) -> RetrievalResult:
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
    import dataclasses
    return dataclasses.replace(
        base,
        sources    = merged[:8],
        confidence = max(primary.confidence, secondary.confidence),
    )


# ── Main workflow ─────────────────────────────────────────────────────────────

@workflow(name="orchestrator_workflow")
async def orchestrator_workflow(inp: OrchestratorInput) -> FinalResponse:
    user_query      = inp.user_query
    session_context = inp.session_context
    ltm_context     = inp.ltm_context
    session         = inp.session          # SessionMemory passed from main_agent
    turn_texts      = inp.turn_texts       # pre-fetched {question_id: {question, answer}}

    bind_context(
        agent           = "orchestrator",
        conversation_id = user_query.conversation_id,
        user_id         = user_query.user_id,
        question_id     = user_query.question_id,
    )
    logger.info("orchestrator_started query_preview=%.80s", user_query.text)

    classification = await classify_query(ClassifyInput(
        query           = user_query.text,
        session_context = session_context,
        ltm_context     = ltm_context,
    ))

    # ── H-9: hard classification failure ─────────────────────────────────────
    if classification.failed:
        logger.error("classify_failed_returning_error query_preview=%.60s", user_query.text)
        return FinalResponse(
            status          = "error",
            answer          = "I wasn't able to process your query. Please rephrase and try again.",
            domain          = None,
            sources         = [],
            confidence      = 0.0,
            attempts_used   = 0,
            conversation_id = user_query.conversation_id,
            user_id         = user_query.user_id,
            question_id     = user_query.question_id,
            tools_used      = [],
            show_citations  = False,
            citations       = [],
        )

    # ── Path A: domain = none (all out-of-scope variants) ────────────────────
    if classification.out_of_scope:
        response_type = classification.response_type
        message       = classification.deflection_message
        streak        = session.off_topic_streak if session else 0

        logger.info(
            "classify_out_of_scope response_type=%s streak=%d query_preview=%.60s",
            response_type, streak, user_query.text,
        )

        # Greetings are fine — no streak increment, no reminder appended.
        # For all other out-of-scope types, append a reminder if streak is high.
        if response_type not in _STREAK_EXEMPT_TYPES:
            message = _apply_streak_reminder(message, streak)

        return FinalResponse(
            status          = "out_of_scope",
            answer          = message,
            domain          = None,
            sources         = [],
            confidence      = 1.0,
            attempts_used   = 0,
            conversation_id = user_query.conversation_id,
            user_id         = user_query.user_id,
            question_id     = user_query.question_id,
            tools_used      = [],
            show_citations  = False,
            citations       = [],
            response_type   = response_type,
        )

    # ── In-domain paths ───────────────────────────────────────────────────────
    domain            = classification.domain
    secondary_domain  = classification.secondary_domain
    domain_confidence = classification.domain_confidence
    is_cross_domain   = (
        domain_confidence < settings.DOMAIN_CONFIDENCE_THRESHOLD
        and secondary_domain is not None
    )

    if is_cross_domain:
        logger.info(
            "cross_domain_fanout primary=%s secondary=%s confidence=%.2f",
            domain, secondary_domain, domain_confidence,
        )

    # ── Path B: reformat latest answer (no retrieval) ─────────────────────────
    # Checked BEFORE whole-chat summary because "summarize" alone (without
    # "our chat" / "this session" etc.) targets only the latest answer.
    if session_context and _is_reformat_command(user_query.text) and not _is_whole_chat_summary(user_query.text):
        logger.info("reformat_shortcut_activated query=%.60s", user_query.text)
        reformatted = await _reformat_prior_answer(user_query.text, session_context)
        if reformatted:
            return FinalResponse(
                status          = "success",
                answer          = reformatted,
                domain          = domain,
                sources         = [],
                confidence      = 1.0,
                attempts_used   = 0,
                conversation_id = user_query.conversation_id,
                user_id         = user_query.user_id,
                question_id     = user_query.question_id,
                tools_used      = [],
                show_citations  = False,
                citations       = [],
            )
        logger.warning("reformat_shortcut_fallthrough — proceeding with retrieval")

    # ── Path C: whole-chat summary ────────────────────────────────────────────
    if _is_whole_chat_summary(user_query.text):
        logger.info("whole_chat_summary_activated query=%.60s", user_query.text)
        if session and turn_texts is not None:
            summary_answer = await _summarize_whole_chat(session, turn_texts)
        else:
            summary_answer = (
                "I don't have session context available to summarize this conversation."
            )
        return FinalResponse(
            status          = "success",
            answer          = summary_answer,
            domain          = domain,
            sources         = [],
            confidence      = 1.0,
            attempts_used   = 0,
            conversation_id = user_query.conversation_id,
            user_id         = user_query.user_id,
            question_id     = user_query.question_id,
            tools_used      = [],
            show_citations  = False,
            citations       = [],
        )

    # ── Path D: normal retrieval ──────────────────────────────────────────────
    # Rewrite once before the retry loop.
    search_query = await _rewrite_query_if_needed(
        user_query.text, session_context, classification.is_followup
    )

    last_result: RetrievalResult | None = None
    tools_tried: list[str] = []

    for attempt_idx in range(settings.MAX_RETRIEVAL_ATTEMPTS):
        idx     = min(attempt_idx, len(_TOOL_LADDER) - 1)
        tool    = _TOOL_LADDER[idx]
        attempt = attempt_idx + 1
        tools_tried.append(tool.value)

        logger.info(
            "retrieval_attempt attempt=%d/%d domain=%s tool=%s cross_domain=%s",
            attempt, settings.MAX_RETRIEVAL_ATTEMPTS, domain, tool, is_cross_domain,
        )
        record_tool(tool=tool.value, domain=domain.value)

        primary_req = OrchestratorRequest(
            query           = search_query,
            domain          = domain,
            tool            = tool,
            attempt         = attempt,
            conversation_id = user_query.conversation_id,
            user_id         = user_query.user_id,
            question_id     = user_query.question_id,
            session_context = session_context,
            ltm_context     = ltm_context,
        )

        if is_cross_domain and secondary_domain:
            secondary_req = OrchestratorRequest(
                query           = search_query,
                domain          = secondary_domain,
                tool            = tool,
                attempt         = attempt,
                conversation_id = user_query.conversation_id,
                user_id         = user_query.user_id,
                question_id     = user_query.question_id,
                session_context = session_context,
                ltm_context     = ltm_context,
            )
            primary_result, secondary_result = await asyncio.gather(
                _call_retrieval_safe(primary_req),
                _call_retrieval_safe(secondary_req),
            )
            if primary_result is None and secondary_result is None:
                logger.error("retrieval_fanout_both_failed attempt=%d", attempt)
                continue
            result = (
                secondary_result if primary_result is None
                else primary_result if secondary_result is None
                else _merge_retrieval_results(primary_result, secondary_result)
            )
        else:
            try:
                result = await call_retrieval(primary_req)
            except CircuitOpenError:
                return FinalResponse(
                    status          = "error",
                    answer          = "Retrieval service is temporarily unavailable. Please try again shortly.",
                    domain          = domain,
                    sources         = [],
                    confidence      = 0.0,
                    attempts_used   = attempt,
                    conversation_id = user_query.conversation_id,
                    user_id         = user_query.user_id,
                    question_id     = user_query.question_id,
                    tools_used      = tools_tried,
                )
            except Exception as exc:
                logger.error("retrieval_failed attempt=%d: %s", attempt, exc)
                continue

        last_result = result
        logger.info(
            "retrieval_result attempt=%d confidence=%.3f passed=%s",
            attempt, result.confidence, result.passed,
        )

        if result.passed:
            logger.info(
                "orchestrator_success attempt=%d confidence=%.3f",
                attempt, result.confidence,
            )
            return FinalResponse(
                status          = "success",
                answer          = result.answer,
                domain          = domain,
                sources         = result.sources,
                confidence      = result.confidence,
                attempts_used   = attempt,
                conversation_id = user_query.conversation_id,
                user_id         = user_query.user_id,
                question_id     = user_query.question_id,
                tools_used      = tools_tried,
                show_citations  = result.show_citations,
                citations       = result.citations,
            )

        logger.warning(
            "confidence_below_threshold attempt=%d confidence=%.3f threshold=%.2f",
            attempt, result.confidence, settings.CONFIDENCE_THRESHOLD,
        )

    logger.error(
        "orchestrator_failed all_attempts=%d exhausted", settings.MAX_RETRIEVAL_ATTEMPTS
    )
    return FinalResponse(
        status          = "failure",
        answer          = last_result.answer if last_result else "",
        domain          = domain,
        sources         = last_result.sources if last_result else [],
        confidence      = last_result.confidence if last_result else 0.0,
        attempts_used   = settings.MAX_RETRIEVAL_ATTEMPTS,
        conversation_id = user_query.conversation_id,
        user_id         = user_query.user_id,
        question_id     = user_query.question_id,
        show_citations  = last_result.show_citations if last_result else False,
        citations       = last_result.citations if last_result else [],
        tools_used      = tools_tried,
    )


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _http
    _register_sigterm()
    _http = httpx.AsyncClient(
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
        limits  = httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
    await asyncio.to_thread(probe_cosmos)
    logger.info("orchestrator_agent_started environment=%s", settings.ENVIRONMENT)
    yield
    await _http.aclose()
    logger.info("orchestrator_agent_stopped")


def _register_sigterm():
    def _handler(signum, frame):
        logger.info("orchestrator_agent_sigterm_received — draining in-flight requests")
    signal.signal(signal.SIGTERM, _handler)


app = FastAPI(title="RAG Orchestrator Agent", lifespan=lifespan)
app.add_middleware(InternalAuthMiddleware)


@app.get("/health/live")
async def liveness() -> dict:
    return {"status": "alive", "agent": "orchestrator"}


@app.get("/health/ready")
async def readiness() -> Response:
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        from shared.cosmos_client import get_chat_container
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {type(exc).__name__}"
        overall_ok = False

    cb_state = _retrieval_breaker.to_dict()
    checks["retrieval_circuit"] = cb_state["state"]
    if cb_state["state"] == "open":
        overall_ok = False

    return Response(
        content    = json.dumps({
            "status": "ready" if overall_ok else "degraded",
            "agent":  "orchestrator",
            "checks": checks,
        }),
        media_type = "application/json",
        status_code = 200 if overall_ok else 503,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "orchestrator"}


@app.post("/orchestrate")
async def orchestrate(raw: Request) -> Response:
    body        = await raw.json()
    session_ctx = body.pop("session_context", "")
    ltm_ctx     = body.pop("ltm_context", "")

    user_query = UserQuery(
        text            = body.get("text", ""),
        conversation_id = body.get("conversation_id", ""),
        user_id         = body.get("user_id", ""),
        question_id     = body.get("question_id", ""),
    )
    bind_context(
        agent           = "orchestrator",
        conversation_id = user_query.conversation_id,
        user_id         = user_query.user_id,
        question_id     = user_query.question_id,
    )

    try:
        result_obj = await orchestrator_workflow.run(OrchestratorInput(
            user_query      = user_query,
            session_context = session_ctx,
            ltm_context     = ltm_ctx,
            session         = None,
            turn_texts      = None,
        ))
        outputs = result_obj.get_outputs()
        final: FinalResponse = outputs[0] if outputs else FinalResponse(
            status          = "failure",
            answer          = "",
            domain          = None,
            conversation_id = user_query.conversation_id,
            user_id         = user_query.user_id,
            question_id     = user_query.question_id,
        )
    except Exception as exc:
        logger.error("orchestrate_endpoint_error: %s", exc, exc_info=True)
        final = FinalResponse(
            status          = "error",
            answer          = "",
            domain          = None,
            conversation_id = user_query.conversation_id,
            user_id         = user_query.user_id,
            question_id     = user_query.question_id,
        )

    return Response(
        content    = json.dumps(final.to_dict()),
        media_type = "application/json",
    )


if __name__ == "__main__":
    uvicorn.run(
        "agents.orchestrator_agent:app",
        host    = "0.0.0.0",
        port    = 8001,
        reload  = False,
        timeout_graceful_shutdown = 60,
    )
