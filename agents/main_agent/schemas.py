"""
Request/response Pydantic models for the Main Agent HTTP API.

Used by:
  - routes/query.py    (QueryBody)
  - routes/feedback.py (FeedbackBody)
"""
from __future__ import annotations

import html
import re

from pydantic import BaseModel, Field, field_validator

from shared.config import settings
from shared.models import FeedbackRating


class QueryBody(BaseModel):
    """Incoming POST /query payload."""

    text: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    conversation_id: str | None = None
    user_id: str = "anonymous"
    idempotency_key: str | None = None

    @field_validator("text")
    @classmethod
    def sanitise_text(cls, v: str) -> str:
        # Strip control characters (null bytes, BEL, BS, etc.) but keep
        # printable unicode, tabs, and newlines.
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v).strip()
        if not v:
            raise ValueError("Query text is empty after sanitisation.")
        return v


class FeedbackBody(BaseModel):
    """Incoming POST /feedback payload."""

    question_id: str = Field(min_length=1, max_length=128)
    answer_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=256)
    user_id: str = Field(default="anonymous", max_length=256)
    rating: FeedbackRating
    comment: str = Field(default="", max_length=2000)

    @field_validator("comment")
    @classmethod
    def escape_html(cls, v: str) -> str:
        """HTML-escape comment before storage to prevent XSS in dashboards."""
        return html.escape(v, quote=True)
