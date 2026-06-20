"""
50-Question RAG evaluation script.

Tests the local ChromaDB + Mistral pipeline across 6 cognitive categories:
  1. Direct Retrieval      (Q1–Q10)   – factual lookups from the docs
  2. Reasoning             (Q11–Q20)  – inference across multiple policy facts
  3. Maths / Calculation   (Q21–Q28)  – arithmetic using numbers in the docs
  4. Memory (multi-turn)   (Q29–Q36)  – questions that reference earlier answers
  5. Deep Thinking         (Q37–Q43)  – synthesis, edge-cases, conflict resolution
  6. Memory + Reasoning    (Q44–Q50)  – prior context + new logical inference

Results are saved to: results/qa_results_<timestamp>.json

Usage:
    cd <repo-root>
    python scripts/test_50_questions.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from mistralai import Mistral
from tools.local_search import search

# ── Config ────────────────────────────────────────────────────────────────────
CHAT_MODEL   = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
N_DOCS       = int(os.getenv("TEST_N_DOCS", "5"))
OUTPUT_DIR   = Path(__file__).parent.parent / "results"
DELAY_S      = float(os.getenv("TEST_DELAY_S", "0.5"))   # polite rate-limit pause

# ── Questions ─────────────────────────────────────────────────────────────────
# Format: (question_id, category, question_text, domain_hint_or_None)
QUESTIONS: list[tuple[str, str, str, str | None]] = [

    # ── 1. Direct Retrieval ──────────────────────────────────────────────────
    ("Q01", "direct_retrieval",
     "How many phases does the employee onboarding process consist of, and what are they?",
     None),

    ("Q02", "direct_retrieval",
     "What is the deadline for submitting expense reimbursement requests after incurring the expense?",
     None),

    ("Q03", "direct_retrieval",
     "How far in advance must annual leave be requested for periods longer than 3 days?",
     None),

    ("Q04", "direct_retrieval",
     "What is the yearly wellness allowance amount provided by the company?",
     None),

    ("Q05", "direct_retrieval",
     "How often are emergency evacuation drills conducted at corporate offices?",
     None),

    ("Q06", "direct_retrieval",
     "What is the MFA account suspension period for accounts without MFA enabled?",
     None),

    ("Q07", "direct_retrieval",
     "What is the hotel booking cap per night in North America?",
     None),

    ("Q08", "direct_retrieval",
     "How long must financial records be retained?",
     None),

    ("Q09", "direct_retrieval",
     "What is the maximum number of unused annual leave days that can be carried forward?",
     None),

    ("Q10", "direct_retrieval",
     "By what date must employees complete annual security awareness training each year?",
     None),

    # ── 2. Reasoning ─────────────────────────────────────────────────────────
    ("Q11", "reasoning",
     "An employee wants to request 5 days of leave starting next Monday. Is that enough advance notice? Explain the rule.",
     None),

    ("Q12", "reasoning",
     "A contractor asks about bonus eligibility and wellness allowance. What can you tell them?",
     None),

    ("Q13", "reasoning",
     "If a purchase request is $30,000, whose approval is required and why?",
     None),

    ("Q14", "reasoning",
     "An employee booked a hotel in Germany at $200/night. Does this comply with policy?",
     None),

    ("Q15", "reasoning",
     "A new employee joined 4 months ago. Are they eligible for the annual performance bonus?",
     None),

    ("Q16", "reasoning",
     "If an office access card has been inactive since January 1st and today is April 15th, what has happened to it?",
     None),

    ("Q17", "reasoning",
     "An employee's project is currently 25 days behind schedule. What governance action is required?",
     None),

    ("Q18", "reasoning",
     "A vendor will store company customer data. What additional step is required before they can be approved?",
     None),

    ("Q19", "reasoning",
     "An employee attended a conference in May. What must they do to get the expense reimbursed?",
     None),

    ("Q20", "reasoning",
     "A department's approved budget is $100,000 but they have spent $112,000. What must they do?",
     None),

    # ── 3. Maths / Calculation ────────────────────────────────────────────────
    ("Q21", "maths",
     "If an employee submits two expense claims: one for $300 (no receipts) and one for $20 (no receipts), which ones require receipts according to policy?",
     None),

    ("Q22", "maths",
     "An employee has 8 unused leave days at year-end. How many can be carried forward and how many are forfeited?",
     None),

    ("Q23", "maths",
     "If the meal reimbursement cap is $75/day, how much can an employee claim for a 5-day business trip?",
     None),

    ("Q24", "maths",
     "An employee spent $1,800 on two certification exams this year. How much remains in their annual certification reimbursement budget?",
     None),

    ("Q25", "maths",
     "A project has a baseline schedule of 50 days. At what number of days elapsed would the project be classified as Red status?",
     None),

    ("Q26", "maths",
     "If hotel stays in Asia cost $160/night for 4 nights, what is the total cost and does it stay within policy limits?",
     None),

    ("Q27", "maths",
     "An employee uses $450 of their $600 wellness allowance on gym membership. How much is left, and does unused balance roll over?",
     None),

    ("Q28", "maths",
     "A department's budget is $90,000 and they spend $100,000. By what dollar amount and percentage have they exceeded the budget? Does this trigger a variance report?",
     None),

    # ── 4. Memory (multi-turn) ────────────────────────────────────────────────
    ("Q29", "memory",
     "Recall the onboarding phases you described earlier. How many business days does an employee have to complete all of them?",
     None),

    ("Q30", "memory",
     "You mentioned the expense reimbursement deadline earlier. If an employee incurred an expense on March 1st, what is the absolute latest submission date?",
     None),

    ("Q31", "memory",
     "Based on the wellness allowance figure you provided, if a company has 200 employees all claiming the full amount, what is the total annual cost to the company?",
     None),

    ("Q32", "memory",
     "Earlier you stated the hotel cap for North America. What is the difference between that cap and the cap for other regions?",
     None),

    ("Q33", "memory",
     "You told me about the MFA suspension rule. If an account was created on June 1st and the user never enabled MFA, on what date would the account be suspended?",
     None),

    ("Q34", "memory",
     "Based on the laptop replacement policy you described, if an employee got their current laptop in January 2021, when is the earliest they can request a replacement under standard policy?",
     None),

    ("Q35", "memory",
     "You said carried-forward leave expires on March 31. If an employee carries forward 5 days and uses 2 before March 31, what happens to the remaining 3 days on April 1?",
     None),

    ("Q36", "memory",
     "Recall the financial records retention period. If a record was created on January 1, 2020, until what year must it be retained?",
     None),

    # ── 5. Deep Thinking ──────────────────────────────────────────────────────
    ("Q37", "deep_thinking",
     "An employee works remotely 4 days a week and needs a new laptop. Their current device is 3 years old and passes hardware diagnostics. What equipment are they entitled to and can they get a new laptop now?",
     None),

    ("Q38", "deep_thinking",
     "A project is 35 days behind schedule and has exceeded its $110,000 budget by 12%. List all the governance and finance actions that must be triggered, referencing the specific policies.",
     None),

    ("Q39", "deep_thinking",
     "An employee wants to maximise their learning benefits this year. They plan to take a $1,500 certification exam and attend a $2,000 conference. Are both reimbursable, and what must they do to claim both?",
     None),

    ("Q40", "deep_thinking",
     "Compare the approval requirements for a $6,000 purchase versus a $26,000 purchase versus a $260,000 customer contract. What escalation chain applies to each?",
     None),

    ("Q41", "deep_thinking",
     "A security incident is classified as 'high severity'. Walk through the complete response timeline: when must it be reported, and what cybersecurity standards apply?",
     None),

    ("Q42", "deep_thinking",
     "An employee's PII deletion request arrives today. Are there any scenarios where the company could lawfully delay deletion beyond 30 days? What does the policy say?",
     None),

    ("Q43", "deep_thinking",
     "If a visitor arrives at a restricted office area without a scheduled escort, what policy applies, and what should staff do?",
     None),

    # ── 6. Memory + Reasoning ────────────────────────────────────────────────
    ("Q44", "memory_reasoning",
     "You described the bonus eligibility rule earlier (6 months continuous employment, March payout). An employee joined on September 15th. Will they receive the March bonus payout that follows their joining date, and why?",
     None),

    ("Q45", "memory_reasoning",
     "Given what you said about the procurement approval matrix and the software subscription policy, would a $12,000 annual SaaS tool require CFO approval? Walk through the reasoning.",
     None),

    ("Q46", "memory_reasoning",
     "Combining the meal reimbursement cap and the travel policy for business class, what is the maximum a travelling executive could claim per day in meals for a 10-hour intercontinental flight trip lasting 3 days?",
     None),

    ("Q47", "memory_reasoning",
     "Based on the project status framework and the executive escalation policy you recalled, at what exact schedule variance percentage AND number of calendar days does a project simultaneously trigger both Red status and an executive review?",
     None),

    ("Q48", "memory_reasoning",
     "You mentioned the certification reimbursement limit ($2,000) and conference policy. If an employee spent $1,800 on certs and $800 on a conference this year, what is the total company reimbursement, and does the combined spend exceed any single cap?",
     None),

    ("Q49", "memory_reasoning",
     "Recall the data retention rules for financial records and PII. If a document contains both financial data and PII, which retention rule governs and why? What happens to the PII when a deletion request is received?",
     None),

    ("Q50", "memory_reasoning",
     "Synthesise everything you know from our conversation: an employee (not a contractor, employed for 8 months, working remotely 4 days/week) submits an expense on day 29 after a business trip that included a $900 hotel stay in Europe for 5 nights. List every policy that applies, flag any violations, and state what they are eligible for.",
     None),
]


# ── RAG helpers ───────────────────────────────────────────────────────────────

def build_context(question: str, domain: str | None) -> tuple[str, list[dict]]:
    """Retrieve relevant docs and format them as a context block."""
    docs = search(question, domain=domain, n_results=N_DOCS)
    if not docs:
        return "No relevant documents found.", []

    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(
            f"[{i}] Title: {d.get('title', 'N/A')}\n"
            f"    Source: {d.get('doc_name', 'N/A')} (p.{d.get('page_number', '?')})\n"
            f"    Content: {d['content']}"
        )
    return "\n\n".join(lines), docs


def ask(
    client: Mistral,
    question: str,
    domain: str | None,
    history: list[dict],
) -> tuple[str, list[dict], list[dict]]:
    """
    Run a single RAG turn.
    Returns (answer, updated_history, retrieved_docs).
    """
    context, docs = build_context(question, domain)

    system_msg = (
        "You are a precise corporate policy assistant. "
        "Answer the user's question using ONLY the provided policy context. "
        "When a question references previous answers, draw on the conversation history. "
        "Be specific: cite numbers, dates, and thresholds directly from the documents. "
        "If the answer cannot be found in the context, say so clearly."
    )

    user_content = (
        f"### Policy Context\n{context}\n\n"
        f"### Question\n{question}"
    )

    messages = [{"role": "system", "content": system_msg}] + history + [
        {"role": "user", "content": user_content}
    ]

    response = client.chat.complete(model=CHAT_MODEL, messages=messages)
    answer   = response.choices[0].message.content.strip()

    # Append to history (keep only the plain question, not the full context block,
    # so history stays compact for multi-turn memory tests).
    new_history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return answer, new_history, docs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY environment variable is not set.")
        sys.exit(1)

    client  = Mistral(api_key=api_key)
    history: list[dict] = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"qa_results_{timestamp}.json"

    results: list[dict] = []

    print(f"Running {len(QUESTIONS)} questions with model '{CHAT_MODEL}' …\n")
    print(f"Results will be saved to: {output_path}\n")
    print("=" * 70)

    for idx, (qid, category, question, domain) in enumerate(QUESTIONS, 1):
        print(f"[{idx:02d}/50] {qid} [{category}]")
        print(f"  Q: {question[:100]}{'…' if len(question) > 100 else ''}")

        t0 = time.perf_counter()
        try:
            answer, history, docs = ask(client, question, domain, history)
            elapsed = time.perf_counter() - t0
            status  = "ok"
        except Exception as exc:
            answer  = f"ERROR: {exc}"
            elapsed = time.perf_counter() - t0
            status  = "error"
            docs    = []

        print(f"  A: {answer[:120]}{'…' if len(answer) > 120 else ''}")
        print(f"  [{status}] {elapsed:.1f}s | docs retrieved: {len(docs)}\n")

        results.append({
            "question_id": qid,
            "category":    category,
            "question":    question,
            "domain_hint": domain,
            "answer":      answer,
            "status":      status,
            "elapsed_s":   round(elapsed, 3),
            "docs_retrieved": [
                {
                    "id":       d["id"],
                    "title":    d.get("title", ""),
                    "doc_name": d.get("doc_name", ""),
                    "score":    round(d.get("score", 0), 4),
                }
                for d in docs
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Save incrementally so partial results survive early termination.
        _save(results, output_path, timestamp)

        if idx < len(QUESTIONS):
            time.sleep(DELAY_S)

    print("=" * 70)
    print(f"\nAll done. {len(results)} results saved to:\n  {output_path}")

    # Summary by category
    from collections import Counter
    cats = Counter(r["category"] for r in results)
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\nCategory breakdown: {dict(cats)}")
    print(f"Errors: {errors}/{len(results)}")


def _save(results: list[dict], path: Path, timestamp: str) -> None:
    payload = {
        "metadata": {
            "generated_at":  timestamp,
            "model":         CHAT_MODEL,
            "total_questions": len(results),
            "categories": list({r["category"] for r in results}),
        },
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
