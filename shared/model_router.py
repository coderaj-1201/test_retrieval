"""
Model router — selects GPT-4.1 (fast, simple) or Claude Sonnet (complex).

Complexity signals (any one triggers Claude):
  - Tool is "decomposition" (orchestrator already decided it's multi-hop)
  - Attempt > 1 (GPT failed/low confidence — retry with stronger model)
  - Query word count > 50
  - Multiple questions in one query (>1 question mark)
  - Two or more complexity keywords present

Routing only applies to the synthesis LLM call in retrieval_agent.
Classification, shortcuts, HyDE, and decomposition always use GPT (fast,
structured output, cheap) — Claude is reserved for answer generation.

If CLAUDE_ENDPOINT is not set, all queries use GPT regardless of complexity.
"""
from __future__ import annotations

import re

from shared.azure_clients import get_claude_client, get_openai_client
from shared.config import settings
from shared.logging_config import get_logger
from openai import OpenAI

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
    words = query.split()
    if len(words) > 50:
        return True
    if query.count("?") > 1:
        return True
    if len(_COMPLEXITY_KEYWORDS.findall(query)) >= 2:
        return True
    return False


def get_model_for_query(
    query: str,
    tool: str = "hybrid",
    attempt: int = 1,
) -> tuple[OpenAI, str]:
    """
    Returns (llm_client, deployment_name) for the given query context.
    Falls back to GPT if Claude is not configured or client init fails.
    """
    if is_complex(query, tool, attempt):
        claude = get_claude_client()
        if claude is not None:
            logger.info(
                "model_router=claude tool=%s attempt=%d query_preview=%.60s",
                tool, attempt, query,
            )
            return claude, settings.CLAUDE_CHAT_DEPLOYMENT
        logger.warning("model_router=gpt_fallback claude_endpoint_not_configured")

    return get_openai_client(), settings.AZURE_OPENAI_CHAT_DEPLOYMENT
