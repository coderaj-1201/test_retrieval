"""
Query Decomposition tool.
Splits a complex, multi-part question into 2-4 atomic sub-questions,
retrieves against each, then merges results before synthesis.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from shared.azure_clients import get_openai_client
from shared.config import settings
from prompts import DECOMPOSE_SYSTEM

logger = logging.getLogger(__name__)


def decompose_query(query: str) -> list[str]:
    """Returns a list of sub-questions. Falls back to [query] on parse error."""
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {"role": "user", "content": f"Question: {query}"},
            ],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw: Any = json.loads(response.choices[0].message.content)
        # Model may return {"questions": [...]} or a bare list
        if isinstance(raw, list):
            sub_questions: list[str] = raw
        elif isinstance(raw, dict):
            sub_questions = next(
                (v for v in raw.values() if isinstance(v, list)), [query]
            )
        else:
            sub_questions = [query]

        if not sub_questions:
            sub_questions = [query]

        logger.debug("Decomposed '%s' → %s", query[:60], sub_questions)
        return sub_questions

    except (json.JSONDecodeError, KeyError, StopIteration) as exc:
        logger.warning("Query decomposition failed (%s); falling back to original query.", exc)
        return [query]
