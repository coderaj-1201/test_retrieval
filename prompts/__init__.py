"""
prompts/__init__.py
────────────────────
Central re-export point for all LLM prompt constants.

Each module documents: which agent uses it, when it fires, and its purpose.
Import from here rather than from individual submodules to keep call sites clean.

  from prompts import (
      CLASSIFY_SYSTEM, CLASSIFY_FALLBACKS, STREAK_REMINDER, STREAK_REMINDER_FIRM,
      REWRITE_SYSTEM,
      REFORMAT_SYSTEM, REFORMAT_VERBS,
      WHOLE_CHAT_SUMMARY_SYSTEM, WHOLE_CHAT_PHRASES,
      SYNTHESIS_SYSTEM,
      HYDE_SYSTEM,
      DECOMPOSE_SYSTEM,
      LTM_UPDATE_SYSTEM,
      build_classify_system,
  )
"""

from prompts.classify import (
    CLASSIFY_FALLBACKS,
    STREAK_REMINDER,
    STREAK_REMINDER_FIRM,
    build_classify_system,
)
from prompts.decompose import DECOMPOSE_SYSTEM
from prompts.hyde import HYDE_SYSTEM
from prompts.ltm_update import LTM_UPDATE_SYSTEM
from prompts.reformat import REFORMAT_SYSTEM, REFORMAT_VERBS
from prompts.synthesis import SYNTHESIS_SYSTEM
from prompts.rewrite import REWRITE_SYSTEM
from prompts.whole_chat_summary import WHOLE_CHAT_SUMMARY_SYSTEM, WHOLE_CHAT_PHRASES

# Built once at import time from the live Domain registry.
CLASSIFY_SYSTEM: str = build_classify_system()

__all__ = [
    "CLASSIFY_SYSTEM",
    "CLASSIFY_FALLBACKS",
    "STREAK_REMINDER",
    "STREAK_REMINDER_FIRM",
    "build_classify_system",
    "REWRITE_SYSTEM",
    "REFORMAT_SYSTEM",
    "REFORMAT_VERBS",
    "WHOLE_CHAT_SUMMARY_SYSTEM",
    "WHOLE_CHAT_PHRASES",
    "SYNTHESIS_SYSTEM",
    "HYDE_SYSTEM",
    "DECOMPOSE_SYSTEM",
    "LTM_UPDATE_SYSTEM",
]
