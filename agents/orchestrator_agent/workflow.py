"""
Core orchestration workflow.

Routing tree (in priority order):
  H-9  classification.failed       → error response
  A    out_of_scope                → deflection (with streak reminder if streak >= 3)
  B    in-domain + reformat cmd    → condense latest answer only (no retrieval)
  C    in-domain + whole-chat ask  → summarize all session turns
  D    in-domain + normal          → rewrite query → retrieval loop

Cross-domain fanout: when domain_confidence < threshold AND a secondary
domain exists, both domains are queried in parallel and results are merged.
"""
from __future__ import annotations

import asyncio

from agent_framework import workflow

from shared.config import settings
from shared.logging_config import bind_context, get_logger
from shared.models import (
    ClassifyInput, Domain, FinalResponse, OrchestratorInput,
    OrchestratorRequest, RetrievalResult, RetrievalTool,
)
from shared.telemetry import record_tool

from agents.orchestrator_agent.classify import classify_query
from agents.orchestrator_agent.retrieval import (
    _call_retrieval_safe, _merge_retrieval_results, call_retrieval,
)
from agents.orchestrator_agent.shortcuts import (
    _apply_streak_reminder,
    _generate_personality_response,
    _is_reformat_command,
    _is_whole_chat_summary,
    _reformat_prior_answer,
    _rewrite_query_if_needed,
    _summarize_whole_chat,
    _STREAK_EXEMPT_TYPES,
)

logger = get_logger(__name__)

_TOOL_LADDER = [RetrievalTool.HYBRID, RetrievalTool.HYDE, RetrievalTool.DECOMPOSITION]


@workflow(name="orchestrator_workflow")
async def orchestrator_workflow(inp: OrchestratorInput) -> FinalResponse:
    """
    Route a user query to the appropriate handler and return a FinalResponse.

    Args:
        inp: OrchestratorInput containing the query, session/LTM context,
             session memory, and pre-fetched turn texts.
    """
    user_query      = inp.user_query
    session_context = inp.session_context
    ltm_context     = inp.ltm_context
    session         = inp.session
    turn_texts      = inp.turn_texts

    bind_context(
        agent="orchestrator",
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
    )
    logger.info("orchestrator_started query_preview=%.80s", user_query.text)

    classification = await classify_query(ClassifyInput(
        query=user_query.text,
        session_context=session_context,
        ltm_context=ltm_context,
    ))

    # ── H-9: hard classification failure ──────────────────────────────────────
    if classification.failed:
        logger.error("classify_failed query_preview=%.60s", user_query.text)
        return FinalResponse(
            status="error",
            answer="I wasn't able to process your query. Please rephrase and try again.",
            domain=None, sources=[], confidence=0.0, attempts_used=0,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
            tools_used=[], show_citations=False, citations=[],
        )

    # ── Path A: out-of-scope (greetings, general, offensive, etc.) ────────────
    if classification.out_of_scope:
        # Reformat verb with no prior in-domain context — give a friendly nudge.
        if _is_reformat_command(user_query.text) and not inp.last_answer:
            return FinalResponse(
                status="out_of_scope",
                answer="There's nothing to condense yet! Ask me something first — I can help with HR, IT, Legal, or Operations policies.",
                domain=None, sources=[], confidence=1.0, attempts_used=0,
                conversation_id=user_query.conversation_id,
                user_id=user_query.user_id,
                question_id=user_query.question_id,
                tools_used=[], show_citations=False, citations=[],
                response_type="clarify",
            )

        response_type = classification.response_type

        # ── "decline" queries go to retrieval — the doc store may have the answer ──
        # Only pure greetings, general capability, and offensive messages skip retrieval.
        _HARD_OOS = {"greeting", "general", "offensive"}
        if response_type not in _HARD_OOS:
            # Use secondary_domain if available, otherwise default to OPS.
            fallback_domain = (
                classification.secondary_domain
                or Domain.OPS
            )
            logger.info(
                "decline_routed_to_retrieval response_type=%s fallback_domain=%s query=%.60s",
                response_type, fallback_domain, user_query.text,
            )
            # Fall through to retrieval with fallback domain.
            domain            = fallback_domain
            secondary_domain  = None
            domain_confidence = 0.5
            is_cross_domain   = False
            search_query      = user_query.text
            last_result: RetrievalResult | None = None
            tools_tried: list[str] = []
            for attempt_idx in range(settings.MAX_RETRIEVAL_ATTEMPTS):
                idx     = min(attempt_idx, len(_TOOL_LADDER) - 1)
                tool    = _TOOL_LADDER[idx]
                attempt = attempt_idx + 1
                tools_tried.append(tool.value)
                record_tool(tool=tool.value, domain=domain.value)
                primary_req = OrchestratorRequest(
                    query=search_query, domain=domain, tool=tool, attempt=attempt,
                    conversation_id=user_query.conversation_id,
                    user_id=user_query.user_id,
                    question_id=user_query.question_id,
                    session_context=session_context,
                    ltm_context=ltm_context,
                )
                try:
                    result = await call_retrieval(primary_req)
                except Exception as exc:
                    logger.error("decline_retrieval_failed attempt=%d: %s", attempt, exc)
                    continue
                last_result = result
                if result.passed:
                    return FinalResponse(
                        status="success", answer=result.answer,
                        domain=domain, sources=result.sources,
                        confidence=result.confidence, attempts_used=attempt,
                        conversation_id=user_query.conversation_id,
                        user_id=user_query.user_id,
                        question_id=user_query.question_id,
                        tools_used=tools_tried,
                        show_citations=result.show_citations,
                        citations=result.citations,
                    )
            # Retrieval found nothing — generate a personality-driven deflection
            personality_msg = await _generate_personality_response(
                user_query.text, response_type, session_context
            )
            return FinalResponse(
                status="out_of_scope",
                answer=personality_msg,
                domain=None, sources=[], confidence=0.0, attempts_used=settings.MAX_RETRIEVAL_ATTEMPTS,
                conversation_id=user_query.conversation_id,
                user_id=user_query.user_id,
                question_id=user_query.question_id,
                tools_used=tools_tried, show_citations=False, citations=[],
                response_type=response_type,
            )

        message = await _generate_personality_response(
            user_query.text, response_type, session_context
        )
        streak  = session.off_topic_streak if session else 0

        logger.info(
            "classify_out_of_scope response_type=%s streak=%d query_preview=%.60s",
            response_type, streak, user_query.text,
        )

        if response_type not in _STREAK_EXEMPT_TYPES:
            message = _apply_streak_reminder(message, streak)

        return FinalResponse(
            status="out_of_scope",
            answer=message,
            domain=None, sources=[], confidence=1.0, attempts_used=0,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
            tools_used=[], show_citations=False, citations=[],
            response_type=response_type,
        )

    # ── In-domain setup ────────────────────────────────────────────────────────
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

    # ── Path B: reformat latest answer ────────────────────────────────────────
    # "Summarize" alone → reformat.  "Summarize our chat" → whole-chat (Path C).
    # last_answer is extracted by question_id in main_agent and sent in the HTTP
    # payload — no string parsing needed, works correctly with multiple turns.
    _last_answer = inp.last_answer or ""

    if _last_answer and _is_reformat_command(user_query.text) and not _is_whole_chat_summary(user_query.text):
        logger.info("reformat_shortcut_activated query=%.60s", user_query.text)
        reformatted = await _reformat_prior_answer(user_query.text, _last_answer)
        if reformatted:
            return FinalResponse(
                status="success", answer=reformatted,
                domain=domain, sources=[], confidence=1.0, attempts_used=0,
                conversation_id=user_query.conversation_id,
                user_id=user_query.user_id,
                question_id=user_query.question_id,
                tools_used=[], show_citations=False, citations=[],
            )
        logger.warning("reformat_shortcut_fallthrough — proceeding to retrieval")

    # ── Path C: whole-chat summary ─────────────────────────────────────────────
    if _is_whole_chat_summary(user_query.text):
        logger.info("whole_chat_summary_activated query=%.60s", user_query.text)
        summary_answer = (
            await _summarize_whole_chat(session, turn_texts)
            if session and turn_texts is not None
            else "I don't have session context available to summarize this conversation."
        )
        return FinalResponse(
            status="success", answer=summary_answer,
            domain=domain, sources=[], confidence=1.0, attempts_used=0,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
            tools_used=[], show_citations=False, citations=[],
        )

    # ── Path D: normal retrieval loop ──────────────────────────────────────────
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
            query=search_query, domain=domain, tool=tool, attempt=attempt,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
            session_context=session_context,
            ltm_context=ltm_context,
        )

        if is_cross_domain and secondary_domain:
            secondary_req = OrchestratorRequest(
                query=search_query, domain=secondary_domain, tool=tool, attempt=attempt,
                conversation_id=user_query.conversation_id,
                user_id=user_query.user_id,
                question_id=user_query.question_id,
                session_context=session_context,
                ltm_context=ltm_context,
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
            except Exception as exc:
                from shared.circuit_breaker import CircuitOpenError
                if isinstance(exc, CircuitOpenError):
                    return FinalResponse(
                        status="error",
                        answer="Retrieval service is temporarily unavailable. Please try again shortly.",
                        domain=domain, sources=[], confidence=0.0, attempts_used=attempt,
                        conversation_id=user_query.conversation_id,
                        user_id=user_query.user_id,
                        question_id=user_query.question_id,
                        tools_used=tools_tried,
                    )
                logger.error("retrieval_failed attempt=%d: %s", attempt, exc)
                continue

        last_result = result
        logger.info(
            "retrieval_result attempt=%d confidence=%.3f passed=%s",
            attempt, result.confidence, result.passed,
        )

        if result.passed:
            logger.info("orchestrator_success attempt=%d confidence=%.3f", attempt, result.confidence)
            return FinalResponse(
                status="success", answer=result.answer,
                domain=domain, sources=result.sources,
                confidence=result.confidence, attempts_used=attempt,
                conversation_id=user_query.conversation_id,
                user_id=user_query.user_id,
                question_id=user_query.question_id,
                tools_used=tools_tried,
                show_citations=result.show_citations,
                citations=result.citations,
            )

        logger.warning(
            "confidence_below_threshold attempt=%d confidence=%.3f threshold=%.2f",
            attempt, result.confidence, settings.CONFIDENCE_THRESHOLD,
        )

    logger.error("orchestrator_failed all_attempts=%d exhausted", settings.MAX_RETRIEVAL_ATTEMPTS)
    return FinalResponse(
        status="failure",
        answer=last_result.answer if last_result else "",
        domain=domain,
        sources=last_result.sources if last_result else [],
        confidence=last_result.confidence if last_result else 0.0,
        attempts_used=settings.MAX_RETRIEVAL_ATTEMPTS,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
        show_citations=last_result.show_citations if last_result else False,
        citations=last_result.citations if last_result else [],
        tools_used=tools_tried,
    )
