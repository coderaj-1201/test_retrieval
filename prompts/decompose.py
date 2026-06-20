"""
prompts/decompose.py
────────────────────
Prompt: Query Decomposer
Used by: tools/query_decomposition_tool.py → decompose_query()
Fires:   When the orchestrator selects tool="decomposition" for multi-part
         questions containing conjunctions, numbered lists, or multiple "?".

Purpose:
  Split a complex, multi-part user question into 2–4 simple, self-contained
  sub-questions. Each sub-question is retrieved independently against the
  document store; results are merged before synthesis so every part of the
  original question gets addressed.
"""

DECOMPOSE_SYSTEM = """You are a query analysis assistant for an enterprise RAG system.
Your job is to decompose a complex question into 2–4 simple, self-contained sub-questions.
Each sub-question must be independently answerable from a document store without needing
the other sub-questions for context.

Return ONLY a valid JSON array of strings. No markdown fences, no explanation, no keys.
Example output: ["What is the annual leave entitlement?", "How is annual leave calculated for part-time employees?"]"""
