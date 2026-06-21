"""
prompts/reformat.py
───────────────────
Prompt: Latest-Answer Reformat / Condenser
Used by: agents/orchestrator_agent.py → _reformat_prior_answer()
Fires:   When the user sends a reformat instruction ("summarize", "bullet points",
         "shorter", "one sentence", "tl;dr", etc.) AND the session has at least
         one prior in-domain turn.

         This path bypasses AI Search entirely — the prior answer already exists
         in session context; we only condense/reshape it per the user's instruction.

         IMPORTANT: This applies ONLY to the most recent answer in the session.
         If the user wants the whole chat summarized, that is handled separately
         by _summarize_whole_chat() via the whole-chat summary path.

Purpose:
  Reformat or condense the most recent assistant answer from conversation history
  according to the user's explicit instruction. No new information is introduced.
"""

REFORMAT_SYSTEM = (
    "You are a helpful enterprise assistant. "
    "You will be given the exact text of the most recent answer the user received, "
    "followed by their reformatting instruction. "
    "Apply the instruction exactly to that answer — reformat, condense, or reshape it as asked. "
    "Do NOT introduce new information, do NOT search for anything, and do NOT "
    "repeat the original answer verbatim unless specifically asked. "
    "Return only the reformatted content, clean and ready to send."
)

# Phrases that signal a reformat-of-latest-answer intent.
# When any of these appear in the query (AND is_followup=True), retrieval is skipped.
REFORMAT_VERBS: frozenset[str] = frozenset({
    "summarize", "summary", "shorter", "briefly", "simplify", "rephrase",
    "bullet point", "bullet points", "in 10 words", "in 5 words", "in one line",
    "one sentence", "in points", "give me a summary", "make it shorter",
    "tl;dr", "tldr", "condense", "shorten", "explain in simple",
    "concise", "short answer", "brief answer", "short summary", "quick summary",
    "in brief", "in short", "quick answer", "one liner", "one-liner",
})
