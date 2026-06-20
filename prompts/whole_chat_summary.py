"""
prompts/whole_chat_summary.py
──────────────────────────────
Prompt: Whole-Chat Summariser
Used by: agents/orchestrator_agent.py → _summarize_whole_chat()
Fires:   When the user explicitly asks to summarize the entire conversation,
         e.g. "summarize our chat", "what did we discuss?", "recap everything",
         "give me a summary of this session".

         This is DISTINCT from the latest-answer reformat path (prompts/reformat.py).
         That path condenses only the most recent answer.
         This path summarizes ALL turns held in the session context window (up to 10).

Purpose:
  Fetch the text of all N turns from chat-history, summarize the full conversation,
  and prepend an explicit count statement so the user knows how many turns were
  included. If fewer than 10 turns exist, clearly state the actual count.

Response always begins with a count statement, e.g.:
  "Here is a summary of your last 7 questions from this session:"
  or
  "Here is a summary of all 3 questions from this session (fewer than 10 on record):"
"""

WHOLE_CHAT_SUMMARY_SYSTEM = (
    "You are a helpful enterprise assistant summarising a conversation. "
    "You will receive a numbered list of question-answer pairs from the session. "
    "Write a concise, professional summary of the topics discussed and key points "
    "covered. Group related topics if there are multiple. "
    "Do not repeat every answer verbatim — synthesise the key takeaways. "
    "Write in second person (e.g. 'You asked about...', 'The session covered...'). "
    "Return only the summary text, clean and ready to send."
)

# Phrases that signal whole-chat summary intent (checked BEFORE reformat verbs).
WHOLE_CHAT_PHRASES: frozenset[str] = frozenset({
    "summarize our chat", "summarize the chat", "summarize our conversation",
    "summarize this conversation", "summarize our session", "summarize the session",
    "summarize everything", "summarize all", "summary of our chat",
    "summary of this chat", "summary of the conversation", "summary of our conversation",
    "what did we discuss", "what did we talk about", "what have we covered",
    "recap our chat", "recap the chat", "recap everything", "recap this session",
    "recap our conversation", "give me a recap", "session summary",
    "chat summary", "conversation summary", "what were my questions",
    "list my questions", "what have i asked",
})
