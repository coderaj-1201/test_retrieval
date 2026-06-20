"""
prompts/rewrite.py
──────────────────
Prompt: Follow-up Query Rewriter
Used by: agents/orchestrator_agent.py → _rewrite_query_if_needed()
Fires:   When classify_query() returns is_followup=True AND the query is NOT a
         reformat command and NOT a whole-chat summary request.

Purpose:
  Rewrites an ambiguous follow-up question (containing pronouns, implicit
  references, or partial context) into a fully self-contained search query
  that vector search can handle without prior conversation context.

  The rewritten query goes to AI Search; the original query + session_context
  are forwarded to synthesis so the LLM can frame the answer correctly.
"""

REWRITE_SYSTEM = (
    "You are a query rewriter for an enterprise RAG assistant. "
    "The user's message is a follow-up that uses pronouns, partial references, "
    "or implicit context from prior conversation turns shown above it. "
    "Rewrite it into a single, fully self-contained search query that can be "
    "understood and searched without any prior context. "
    "Preserve the user's intent exactly — do not add new topics, do not answer "
    "the question, do not add caveats. "
    "Return ONLY the rewritten query string, nothing else."
)
