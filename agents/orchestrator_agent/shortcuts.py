"""
LLM shortcut handlers for the Orchestrator Agent.

These paths bypass retrieval entirely when the query intent is clear from
the session context alone:

  _reformat_prior_answer   — condense/reformat the latest answer only
  _summarize_whole_chat    — summarize ALL turns in the session window
  _rewrite_query_if_needed — expand a follow-up into a self-contained search query

Helper predicates:
  _is_reformat_command  — detects "summarize", "bullet points", "shorter" etc.
  _is_whole_chat_summary — detects "summarize our chat", "what did we discuss" etc.
  _apply_streak_reminder — appends a purpose reminder when streak >= 3
"""
from __future__ import annotations

import asyncio

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.logging_config import get_logger
from shared.models import SessionMemory
from shared.retry import llm_retry
from prompts import (
    REFORMAT_SYSTEM,
    REFORMAT_VERBS,
    REWRITE_SYSTEM,
    STREAK_REMINDER,
    STREAK_REMINDER_FIRM,
    WHOLE_CHAT_PHRASES,
    WHOLE_CHAT_SUMMARY_SYSTEM,
)

logger = get_logger(__name__)

# Response types that increment the off-topic streak counter.
_STREAK_INCREMENTING_TYPES = {"general", "decision_making", "offensive", "decline"}
# Response types that are fine and exempt from streak tracking.
_STREAK_EXEMPT_TYPES = {"greeting", "clarify"}


def _is_reformat_command(text: str) -> bool:
    """Return True when the query looks like a reformat/condense instruction."""
    t = text.strip().lower()
    return any(phrase in t for phrase in REFORMAT_VERBS)


def _is_whole_chat_summary(text: str) -> bool:
    """Return True when the query asks to summarize the full conversation."""
    t = text.strip().lower()
    return any(phrase in t for phrase in WHOLE_CHAT_PHRASES)


def _apply_streak_reminder(message: str, streak: int) -> str:
    """Append a purpose reminder to deflection messages when streak is high."""
    if streak >= 6:
        return message + STREAK_REMINDER_FIRM
    if streak >= 3:
        return message + STREAK_REMINDER
    return message


async def _reformat_prior_answer(instruction: str, session_context: str) -> str:
    """
    Reformat the most recent answer using the user's instruction.

    Args:
        instruction:     The user's reformat request (e.g. "make it shorter").
        session_context: Formatted session context containing the prior answer.

    Returns:
        Reformatted text, or empty string on failure (caller falls through to retrieval).
    """
    @llm_retry
    def _call():
        return get_openai_client().chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": REFORMAT_SYSTEM},
                {"role": "user",   "content": f"{session_context}\n\nInstruction: {instruction}"},
            ],
            temperature=0,
            max_tokens=500,
        )

    try:
        resp = await asyncio.to_thread(_call)
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("reformat_prior_answer_failed instruction=%.60s: %s", instruction, exc)
        return ""


async def _summarize_whole_chat(
    session: SessionMemory,
    turn_texts: dict[str, dict[str, str]],
) -> str:
    """
    Summarize ALL turns stored in the session context window.

    Leads with an explicit count statement so the user knows how many
    questions are being covered.

    Args:
        session:    Current SessionMemory (used for turn ordering).
        turn_texts: Pre-fetched {question_id: {"question": …, "answer": …}}.

    Returns:
        A summary string prefixed with a count statement.
    """
    available_turns = [t for t in session.turns if t.question_id in turn_texts]
    count = len(available_turns)

    if count == 0:
        return "I don't have any previous questions from this session on record to summarize."

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
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": WHOLE_CHAT_SUMMARY_SYSTEM},
                {"role": "user",   "content": numbered},
            ],
            temperature=0,
            max_tokens=600,
        )

    try:
        resp = await asyncio.to_thread(_call)
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
    """
    Expand a follow-up query into a self-contained search string.

    Only triggers when is_followup=True AND session context is non-empty.
    Falls back to the original query on any LLM error.
    """
    if not is_followup or not session_context:
        return query

    @llm_retry
    def _call():
        return get_openai_client().chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user",   "content": f"{session_context}\n\nFollow-up question: {query}"},
            ],
            temperature=0,
            max_tokens=120,
        )

    try:
        resp = await asyncio.to_thread(_call)
        rewritten = resp.choices[0].message.content.strip().strip('"')
        if rewritten:
            logger.info("query_rewritten original=%.60s rewritten=%.60s", query, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("query_rewrite_failed query=%.60s: %s", query, exc)
    return query
