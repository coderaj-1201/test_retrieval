"""
prompts/hyde.py
───────────────
Prompt: HyDE — Hypothetical Document Embedder
Used by: tools/hyde_tool.py → generate_hypothetical_document()
Fires:   When the orchestrator selects tool="hyde" for vague, conceptual, or
         exploratory questions where a hypothetical document improves recall.

Purpose:
  Generate a short, plausible, factual passage that looks like it would appear
  in an internal enterprise policy or procedure document. This synthetic passage
  is then embedded and used as the search vector instead of the raw query —
  dramatically improving retrieval for abstract or open-ended questions.
"""

HYDE_SYSTEM = (
    "You are a knowledgeable enterprise assistant. "
    "Write a concise, factual passage (3–5 sentences) that directly answers "
    "the question below as if it appeared in an internal company policy or "
    "procedure document. "
    "Do not add caveats, disclaimers, or first-person language ('I', 'we'). "
    "Write as authoritative internal documentation — present tense, specific, "
    "and grounded in enterprise operations context."
)
