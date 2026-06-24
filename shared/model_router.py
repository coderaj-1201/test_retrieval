"""
Model router — selects GPT-4.1 (fast, simple) or Claude Sonnet (complex).

Complexity signals (any one triggers Claude):
  - Tool is "decomposition" (orchestrator already decided it's multi-hop)
  - Attempt > 1 (GPT failed/low confidence — retry with stronger model)
  - Query word count > 50
  - Multiple questions in one query (>1 question mark)
  - Two or more complexity keywords present

Routing only applies to synthesis in retrieval_agent.
Classification, shortcuts, HyDE, and decomposition always use GPT (fast,
structured output, cheap).

If CLAUDE_ENDPOINT is not set, all queries use GPT regardless of complexity.
"""
from __future__ import annotations

import asyncio
import re
from typing import Callable

from shared.config import settings
from shared.logging_config import get_logger

logger = get_logger(__name__)

_COMPLEXITY_KEYWORDS = re.compile(
    r"\b("
    r"compare|comparison|difference between|pros and cons|advantages|disadvantages|"
    r"explain|elaborate|walk me through|step by step|in detail|"
    r"why|how does|how do|what would happen|impact of|implication|"
    r"multiple|several|various|all the|list all|summarise|summarize|"
    r"conflict|contradict|exception|unless|however|but what if|edge case"
    r")\b",
    re.IGNORECASE,
)


def is_complex(query: str, tool: str = "hybrid", attempt: int = 1) -> bool:
    if attempt > 1:
        return True
    if tool == "decomposition":
        return True
    if len(query.split()) > 50:
        return True
    if query.count("?") > 1:
        return True
    if len(_COMPLEXITY_KEYWORDS.findall(query)) >= 2:
        return True
    return False


def _call_gpt(messages: list[dict]) -> str:
    """Synchronous GPT call — run via asyncio.to_thread."""
    from shared.azure_clients import get_openai_client
    from shared.retry import llm_retry

    @llm_retry
    def _inner():
        return get_openai_client().chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=settings.SYNTHESIS_TEMPERATURE,
            max_tokens=settings.SYNTHESIS_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

    resp = _inner()
    return resp.choices[0].message.content.strip()


def _call_claude(messages: list[dict]) -> str:
    """Synchronous Claude call via AnthropicFoundry — run via asyncio.to_thread."""
    from shared.azure_clients import get_claude_client
    from shared.retry import llm_retry

    client = get_claude_client()
    if client is None:
        raise RuntimeError("Claude client not available")

    # Claude API takes system as a top-level param; extract it from messages list.
    system_content = ""
    non_system = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            non_system.append(msg)

    @llm_retry
    def _inner():
        return client.messages.create(
            model=settings.CLAUDE_CHAT_DEPLOYMENT,
            system=system_content,
            messages=non_system,
            max_tokens=settings.SYNTHESIS_MAX_TOKENS,
        )

    resp = _inner()
    return resp.content[0].text.strip()


async def call_synthesis_llm(
    messages: list[dict],
    query: str,
    tool: str = "hybrid",
    attempt: int = 1,
) -> tuple[str, str]:
    """
    Route and call the appropriate LLM for synthesis.
    Returns (raw_content_str, model_used).
    Falls back to GPT if Claude fails.
    """
    use_claude = is_complex(query, tool, attempt) and settings.CLAUDE_ENDPOINT

    if use_claude:
        logger.info(
            "model_router=claude tool=%s attempt=%d query_preview=%.60s",
            tool, attempt, query,
        )
        try:
            content = await asyncio.to_thread(_call_claude, messages)
            return content, settings.CLAUDE_CHAT_DEPLOYMENT
        except Exception as exc:
            logger.warning(
                "claude_synthesis_failed tool=%s attempt=%d — falling back to GPT: %s",
                tool, attempt, exc,
            )

    content = await asyncio.to_thread(_call_gpt, messages)
    return content, settings.AZURE_OPENAI_CHAT_DEPLOYMENT
