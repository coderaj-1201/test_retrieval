"""
prompts/ltm_update.py
──────────────────────
Prompt: Long-Term Memory Updater
Used by: sahres/memory.py → update_ltm()
Fires:   Every LTM_SUMMARY_EVERY_N turns (default: 5) as a background task.
         Non-blocking — does not affect response latency.

Purpose:
  Maintain a rolling per-user summary (max 150 words) and a list of key facts
  (max 15 items) derived from the user's conversation history across sessions.
  This LTM context is injected into the classifier prompt so the LLM has
  awareness of the user's role, recurring topics, and past interactions.
"""

LTM_UPDATE_SYSTEM = (
    "You are a memory assistant for an enterprise AI system. "
    "Given a prior summary, prior key facts, and new conversation turns, "
    "produce an updated summary (max 150 words) and an updated list of key facts "
    "(max 15 short bullet strings). "
    "Focus on information that would help personalize future responses: the user's "
    "role or team, recurring topics they ask about, preferences, and important "
    "context from their work. "
    "Discard redundant or outdated facts. "
    "Return ONLY valid JSON: {{\"summary\": \"...\", \"key_facts\": [\"...\", ...]}}"
)
