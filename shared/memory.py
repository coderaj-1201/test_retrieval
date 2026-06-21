"""
Memory manager — short-term (session) + long-term (per-user summary).

Short-term  : last SESSION_MAX_TURNS turn *pointers* per conversation_id.
              Each pointer holds question_id, answer_id, domain, confidence,
              tools_used, and timestamp — NO full text.
              Full question/answer text lives in the chat-history container and
              is fetched on demand via fetch_turn_texts().
              Stored in Cosmos `sessions` container (TTL = 7 days).
              Also kept in process-local LRU cache for low-latency reads.

Long-term   : rolling LLM-generated summary + extracted key_facts per user_id.
              Stored in Cosmos `long-term-memory` container (no TTL).
              Updated every LTM_SUMMARY_EVERY_N turns as a background task.

off_topic_streak:
              Counts consecutive out-of-scope / declined responses in the session.
              Reset to 0 on any successful in-domain answer.
              Greeted-but-not-declined turns do NOT increment the streak.
              The orchestrator uses this to append a bot-purpose reminder.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from shared.config import settings
from shared.cosmos_client import (
    get_chat_container, get_ltm_container, get_sessions_container,
    get_document, upsert_document, query_documents,
)
from shared.models import ConversationTurn, LongTermMemoryRecord, SessionMemory

logger = logging.getLogger(__name__)


# ── Short-term memory ─────────────────────────────────────────────────────────

async def load_session(conversation_id: str, user_id: str) -> SessionMemory:
    """Load session from Cosmos, or create a new one if not found."""
    try:
        doc = await asyncio.to_thread(
            get_document, get_sessions_container(), conversation_id, conversation_id
        )
    except Exception as exc:
        logger.error(
            "session_load_failed conversation_id=%s — starting fresh: %s",
            conversation_id, exc, exc_info=True,
        )
        doc = None

    if doc:
        turns = [
            ConversationTurn(**{
                k: v for k, v in t.items()
                # Accept docs written before the text-strip migration;
                # ignore question/answer fields if present.
                if k in ("question_id", "answer_id", "domain", "confidence",
                         "tools_used", "timestamp")
            })
            for t in doc.get("turns", [])
        ]
        session = SessionMemory(
            conversation_id  = conversation_id,
            user_id          = user_id,
            turns            = turns,
            off_topic_streak = int(doc.get("off_topic_streak", 0)),
            last_answer      = doc.get("last_answer", ""),
            created_at       = doc.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at       = doc.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )
    else:
        session = SessionMemory(conversation_id=conversation_id, user_id=user_id)

    return session


async def append_turn(
    session: SessionMemory,
    turn: ConversationTurn,
    *,
    is_in_domain: bool,
    is_greeting: bool = False,
    last_answer: str = "",
) -> None:
    """
    Append a turn pointer, update the off_topic_streak, trim to window, persist.

    streak rules:
      - Successful in-domain answer  → reset streak to 0
      - Greeting response            → streak unchanged (greetings are fine)
      - Any other out-of-scope       → increment streak

    last_answer is stored on the session so the NEXT request can read the
    previous answer reliably — works across replicas and after restarts.
    """
    session.turns.append(turn)
    if len(session.turns) > settings.SESSION_MAX_TURNS:
        session.turns = session.turns[-settings.SESSION_MAX_TURNS:]

    if is_in_domain:
        session.off_topic_streak = 0
    elif not is_greeting:
        session.off_topic_streak += 1

    if last_answer:
        session.last_answer = last_answer

    session.updated_at = datetime.now(timezone.utc).isoformat()
    try:
        await asyncio.to_thread(upsert_document, get_sessions_container(), session.to_dict())
    except Exception as exc:
        logger.error(
            "session_persist_failed conversation_id=%s: %s",
            session.conversation_id, exc, exc_info=True,
        )
    logger.debug(
        "session_updated conversation_id=%s turns=%d streak=%d",
        session.conversation_id, len(session.turns), session.off_topic_streak,
    )


async def fetch_turn_texts(
    conversation_id: str,
    question_ids: list[str],
) -> dict[str, dict[str, str]]:
    """
    Batch-fetch question and answer text for a list of question_ids from
    the chat-history container (partitioned by conversation_id).

    Returns a dict keyed by question_id:
      { question_id: {"question": "...", "answer": "..."} }

    Missing documents are silently omitted from the result.
    """
    if not question_ids:
        return {}

    placeholders = ", ".join(f"@id{i}" for i in range(len(question_ids)))
    params = [{"name": f"@id{i}", "value": qid} for i, qid in enumerate(question_ids)]
    cosmos_query = (
        f"SELECT c.question_id, c.question, c.answer "
        f"FROM c WHERE c.question_id IN ({placeholders})"
    )

    try:
        docs = await asyncio.to_thread(
            query_documents,
            get_chat_container(),
            cosmos_query,
            params,
            partition_key=conversation_id,
        )
        return {
            d["question_id"]: {
                "question": d.get("question", ""),
                "answer":   d.get("answer", ""),
            }
            for d in docs
            if "question_id" in d
        }
    except Exception as exc:
        logger.error(
            "fetch_turn_texts_failed conversation_id=%s: %s",
            conversation_id, exc, exc_info=True,
        )
        return {}


def format_recent_context(
    records: list[dict[str, str]],
    *,
    max_answer_chars: int = 800,
) -> str:
    """
    Build a formatted context block from a list of recent chat records
    (as returned by fetch_recent_chat_records, newest-first).

    Returns an empty string when records is empty.
    Injected into classifier and rewrite prompts so the LLM can resolve
    follow-ups and detect topic continuations.
    """
    if not records:
        return ""

    # Reverse so context reads oldest → newest (natural conversation order).
    ordered = list(reversed(records))
    lines = [f"## Recent conversation (last {len(ordered)} turn(s))"]
    for r in ordered:
        lines.append(f"Q: {r['question']}")
        answer = r["answer"]
        excerpt = answer[:max_answer_chars] + ("…" if len(answer) > max_answer_chars else "")
        lines.append(f"A: {excerpt}")
    return "\n".join(lines)


# ── Long-term memory ──────────────────────────────────────────────────────────

async def load_ltm(user_id: str) -> LongTermMemoryRecord | None:
    try:
        doc = await asyncio.to_thread(
            get_document, get_ltm_container(), f"ltm-{user_id}", user_id
        )
    except Exception as exc:
        logger.error("ltm_load_failed user_id=%s: %s", user_id, exc, exc_info=True)
        return None
    if doc:
        return LongTermMemoryRecord(
            id                      = doc["id"],
            user_id                 = doc["user_id"],
            summary                 = doc.get("summary", ""),
            key_facts               = doc.get("key_facts", []),
            last_updated            = doc.get("last_updated", ""),
            source_conversation_ids = doc.get("source_conversation_ids", []),
        )
    return None


async def update_ltm(user_id: str, session: SessionMemory) -> None:
    """
    Called every LTM_SUMMARY_EVERY_N turns. Uses LLM to produce a rolling
    summary + key facts list from the full session history.
    Fetches turn texts from chat-history before summarising.
    """
    from shared.azure_clients import get_openai_client
    from prompts import LTM_UPDATE_SYSTEM

    existing      = await load_ltm(user_id)
    prior_summary = existing.summary   if existing else ""
    prior_facts   = existing.key_facts if existing else []

    prior_summary_bounded = prior_summary[:settings.LTM_MAX_SUMMARY_CHARS]
    prior_facts_bounded   = prior_facts[:settings.LTM_MAX_FACTS]

    if len(prior_summary) > settings.LTM_MAX_SUMMARY_CHARS:
        logger.warning(
            "ltm_summary_truncated user_id=%s original_len=%d bounded_len=%d",
            user_id, len(prior_summary), settings.LTM_MAX_SUMMARY_CHARS,
        )

    # Fetch full text for all turns in this session for the LTM summary.
    question_ids = [t.question_id for t in session.turns]
    turn_texts   = await fetch_turn_texts(session.conversation_id, question_ids)
    all_text = "\n".join(
        f"Q: {turn_texts[t.question_id]['question']}\nA: {turn_texts[t.question_id]['answer']}"
        for t in session.turns
        if t.question_id in turn_texts
    )

    if not all_text:
        logger.warning("ltm_update_skipped user_id=%s — no turn texts available", user_id)
        return

    user_msg = (
        f"Prior summary:\n{prior_summary_bounded}\n\n"
        f"Prior key facts:\n{json.dumps(prior_facts_bounded)}\n\n"
        f"New turns:\n{all_text}"
    )

    try:
        resp = await asyncio.to_thread(
            get_openai_client().chat.completions.create,
            model    = settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages = [
                {"role": "system", "content": LTM_UPDATE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature     = 0,
            max_tokens      = 600,
            response_format = {"type": "json_object"},
        )
        raw       = json.loads(resp.choices[0].message.content)
        summary   = raw.get("summary",   prior_summary_bounded)
        key_facts = raw.get("key_facts", prior_facts_bounded)
    except Exception as exc:
        logger.error("ltm_update_llm_failed user_id=%s: %s", user_id, exc, exc_info=True)
        return

    src_ids = list({
        *(existing.source_conversation_ids if existing else []),
        session.conversation_id,
    })
    record = LongTermMemoryRecord(
        id       = f"ltm-{user_id}",
        user_id  = user_id,
        summary  = summary,
        key_facts = key_facts,
        source_conversation_ids = src_ids,
    )
    await asyncio.to_thread(upsert_document, get_ltm_container(), record.to_dict())
    logger.info("ltm_updated user_id=%s facts=%d", user_id, len(key_facts))


async def fetch_recent_chat_records(
    conversation_id: str,
    n: int,
) -> list[dict[str, str]]:
    """
    Return the most recent `n` chat-history records for a conversation,
    newest first. Each entry has "question" and "answer" keys.

    Queried by _ts DESC so it is always consistent with what the user last saw —
    no session cache, no replica skew.
    """
    query = f"SELECT TOP {n} c.question, c.answer FROM c ORDER BY c._ts DESC"
    try:
        docs = await asyncio.to_thread(
            query_documents,
            get_chat_container(),
            query,
            [],
            partition_key=conversation_id,
        )
        return [
            {"question": d.get("question", ""), "answer": d.get("answer", "")}
            for d in docs
        ]
    except Exception as exc:
        logger.warning(
            "fetch_recent_chat_records_failed conversation_id=%s n=%d: %s",
            conversation_id, n, exc,
        )
    return []


def format_ltm_context(ltm: LongTermMemoryRecord | None) -> str:
    """Render LTM as a compact string for prompt injection."""
    if not ltm or not ltm.summary:
        return ""
    lines = ["## Long-term user context", ltm.summary]
    if ltm.key_facts:
        lines.append("Key facts:")
        lines.extend(f"- {f}" for f in ltm.key_facts[:10])
    return "\n".join(lines)
