"""
Short-term memory evaluation — 5 scenarios with follow-up questions.

Each scenario establishes context in the first question, then asks 3-4
follow-ups that can only be answered correctly if the bot recalls what
was said earlier in the same conversation.

Memory behaviours tested per scenario:
  S1 — Expense claim journey  : recall amounts, apply 30-day rule, spot the gap
  S2 — New hire onboarding    : recall phases, chain leave → bonus → wellness
  S3 — Project governance     : recall budget + delay, apply two thresholds
  S4 — Security & IT assets   : recall incident timing, apply laptop + MFA rules
  S5 — Travel planning        : recall trip details, correct an over-limit claim

Usage:
    python scripts/run_memory_eval.py

Output written to ./eval_results/:
    memory_eval.json   — full structured results
    memory_eval.txt    — human-readable Q&A report

The 'session_key' field groups questions into conversations.
All questions sharing a key are sent with the same conversation_id.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ── Config ─────────────────────────────────────────────────────────────────────

MAIN_AGENT_URL = os.getenv("MAIN_AGENT_URL", "http://localhost:8000")
OUTPUT_DIR     = Path("eval_results")
REQUEST_DELAY  = 1.5    # seconds between requests
TIMEOUT        = 60.0   # seconds per request

# ── Test scenarios ──────────────────────────────────────────────────────────────
# Each scenario has a unique session_key so all its turns share one conversation.
# 'turn' is just a label (e.g. "opener", "follow-up-1") for reporting clarity.
# 'note' is what the human evaluator should verify.

TESTS: list[dict[str, Any]] = [

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 1 — EXPENSE CLAIM JOURNEY
    # Sarah just got back from a 5-day trip to Toronto and Tokyo.
    # She wants to know what she can claim, then discovers a problem.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "scenario":    "S1 – Expense claim journey",
        "turn":        "opener",
        "session_key": "s1-expense",
        "question":    (
            "I just returned from a business trip — 3 nights in Toronto at $220/night "
            "and 2 nights in Tokyo at $210/night. What are the hotel reimbursement rules "
            "for each city?"
        ),
        "note": (
            "Toronto (North America): cap $250 — $220 fully covered. "
            "Tokyo (outside NA): cap $180 — $210 exceeds cap by $30/night. "
            "Bot should state both caps and flag the Tokyo overage."
        ),
    },
    {
        "scenario":    "S1 – Expense claim journey",
        "turn":        "follow-up-1",
        "session_key": "s1-expense",
        "question":    "So how much in total will I get reimbursed for hotels, and how much comes out of my own pocket?",
        "note": (
            "Should recall the trip from the opener: "
            "Toronto: 3×$220=$660 (fully reimbursed). "
            "Tokyo: 2×$180=$360 reimbursed, 2×$30=$60 OOP. "
            "Total reimbursed: $1,020. Total OOP: $60."
        ),
    },
    {
        "scenario":    "S1 – Expense claim journey",
        "turn":        "follow-up-2",
        "session_key": "s1-expense",
        "question":    "I also spent $80 on meals one day in Tokyo and $60 another day. What can I claim for those two days?",
        "note": (
            "Meal cap is $75/day. "
            "Day 1: $80 → claim $75, OOP $5. "
            "Day 2: $60 → claim $60 (under cap). "
            "Total meal claim: $135."
        ),
    },
    {
        "scenario":    "S1 – Expense claim journey",
        "turn":        "follow-up-3",
        "session_key": "s1-expense",
        "question":    (
            "The trip ended 32 days ago. I haven't submitted my expense claim yet. "
            "Am I still eligible for reimbursement?"
        ),
        "note": (
            "Policy: 30-day deadline. 32 days = past the deadline. "
            "Bot should say the claim is technically ineligible and suggest "
            "contacting the finance team for an exception."
        ),
    },
    {
        "scenario":    "S1 – Expense claim journey",
        "turn":        "follow-up-4",
        "session_key": "s1-expense",
        "question":    "If I do get an exception approved, summarise the total I'd be claiming.",
        "note": (
            "Bot should recall the full session: "
            "Hotels: $1,020 reimbursed ($60 OOP). "
            "Meals: $135 for the two Tokyo days. "
            "Grand total claim: $1,155. Should present a clean summary."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 2 — NEW HIRE ONBOARDING JOURNEY
    # Marcus starts next Monday and wants to understand his first month.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "scenario":    "S2 – New hire onboarding",
        "turn":        "opener",
        "session_key": "s2-newhire",
        "question":    "I'm starting a new job on Monday. What exactly happens during my first week?",
        "note": (
            "Should describe all 3 onboarding phases: "
            "Day 1 documentation, Days 2-3 system access, Week 1 orientation. "
            "All must be done within 5 business days."
        ),
    },
    {
        "scenario":    "S2 – New hire onboarding",
        "turn":        "follow-up-1",
        "session_key": "s2-newhire",
        "question":    "Great. And after my first week — when can I take annual leave?",
        "note": (
            "Leave policy: no tenure requirement. Can request immediately. "
            "Must give 2 weeks' notice for periods longer than 3 days. "
            "Bot should NOT say 6 months (that's the bonus rule, a common mix-up)."
        ),
    },
    {
        "scenario":    "S2 – New hire onboarding",
        "turn":        "follow-up-2",
        "session_key": "s2-newhire",
        "question":    "What about the performance bonus — when do I become eligible, and when is it paid?",
        "note": (
            "Bonus: 6 months continuous employment to be eligible. "
            "Payout in March, calculated on prior year's performance score. "
            "Should correctly distinguish this from the leave rule."
        ),
    },
    {
        "scenario":    "S2 – New hire onboarding",
        "turn":        "follow-up-3",
        "session_key": "s2-newhire",
        "question":    "I start Monday. If I want the bonus that pays out this coming March, can I get it?",
        "note": (
            "Today is 2026-06-20. March 2027 payout requires 6 months by that March. "
            "Starting in June 2026 → 6 months = December 2026 → eligible for March 2027. "
            "Bot should reason through the timeline and say yes."
        ),
    },
    {
        "scenario":    "S2 – New hire onboarding",
        "turn":        "follow-up-4",
        "session_key": "s2-newhire",
        "question":    (
            "Give me a one-paragraph summary of everything you've told me about my first year — "
            "onboarding, leave, and bonus timeline."
        ),
        "note": (
            "Should recall all prior turns: 5-day onboarding, immediate leave eligibility "
            "(2-week notice for >3 days), bonus after 6 months (eligible March 2027). "
            "Tests full-session memory summarisation."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 3 — PROJECT GOVERNANCE CHAIN
    # Priya is running a $130,000 project that's in trouble.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "scenario":    "S3 – Project governance",
        "turn":        "opener",
        "session_key": "s3-project",
        "question":    (
            "I'm managing a project with a $130,000 budget. "
            "What governance requirements apply to a project of this size?"
        ),
        "note": (
            "Budget > $100k → monthly steering committee reviews required. "
            "Should state this clearly."
        ),
    },
    {
        "scenario":    "S3 – Project governance",
        "turn":        "follow-up-1",
        "session_key": "s3-project",
        "question":    "The project is now 18 days behind schedule and has a 15% budget overrun. What needs to happen?",
        "note": (
            "18 days delay: under 30-day executive review threshold — no exec review yet. "
            "15% overrun: > 10% → must submit variance explanation to Finance. "
            "Still needs monthly steering committee (already established in opener). "
            "Bot should recall the $130k budget from previous turn."
        ),
    },
    {
        "scenario":    "S3 – Project governance",
        "turn":        "follow-up-2",
        "session_key": "s3-project",
        "question":    "It's now been 33 days since the last milestone. Does anything new get triggered?",
        "note": (
            "33 days > 30-day threshold → executive review now required. "
            "Bot should recall prior turns and add this new trigger to the existing list."
        ),
    },
    {
        "scenario":    "S3 – Project governance",
        "turn":        "follow-up-3",
        "session_key": "s3-project",
        "question":    "Give me a complete list of every governance action currently required for this project.",
        "note": (
            "Should recall the full session and list all three: "
            "1. Monthly steering committee (>$100k). "
            "2. Variance explanation to Finance (>10% overrun). "
            "3. Executive review (>30 days delay). "
            "Tests accumulation of facts across turns."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 4 — SECURITY & IT ASSETS
    # Ravi reports a security incident and then asks about his equipment.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "scenario":    "S4 – Security & IT assets",
        "turn":        "opener",
        "session_key": "s4-security",
        "question":    (
            "We detected a security incident at 9:15am this morning. "
            "It's been classified as high severity. When does it need to be reported by?"
        ),
        "note": (
            "High severity: within 4 hours of detection. "
            "9:15am + 4h = 1:15pm. Bot should state the deadline clearly."
        ),
    },
    {
        "scenario":    "S4 – Security & IT assets",
        "turn":        "follow-up-1",
        "session_key": "s4-security",
        "question":    "Actually we've just reclassified it as critical. How does that change things?",
        "note": (
            "Critical: within 1 hour. 9:15am + 1h = 10:15am. "
            "Bot should recall the 9:15am detection time from the opener "
            "and give the updated deadline."
        ),
    },
    {
        "scenario":    "S4 – Security & IT assets",
        "turn":        "follow-up-2",
        "session_key": "s4-security",
        "question":    (
            "Separately — my laptop is 3 years old and I want to replace it. "
            "Am I eligible?"
        ),
        "note": (
            "Laptop replacement: every 4 years unless hardware fails diagnostics "
            "or executive approval. 3 years = not yet eligible under standard policy. "
            "Bot should say not yet eligible."
        ),
    },
    {
        "scenario":    "S4 – Security & IT assets",
        "turn":        "follow-up-3",
        "session_key": "s4-security",
        "question":    (
            "One of my team members hasn't set up MFA yet — their account was created "
            "5 days ago. Is there a problem?"
        ),
        "note": (
            "MFA policy: accounts without MFA suspended after 7 days. "
            "5 days in = still within grace period, but should flag the urgency. "
            "Bot should recall this is a separate IT question (not related to the incident)."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 5 — TRAVEL PLANNING NEGOTIATION
    # Elena is planning a conference trip and negotiating what's claimable.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "scenario":    "S5 – Travel planning",
        "turn":        "opener",
        "session_key": "s5-travel",
        "question":    (
            "I'm attending a 3-day conference in Sydney next month. "
            "I'll need 2 nights in a hotel and 3 days of meals. "
            "What are my reimbursement limits?"
        ),
        "note": (
            "Sydney (outside NA): hotel cap $180/night → 2×$180=$360 max. "
            "Meals: $75/day → 3×$75=$225 max. "
            "Total max: $585."
        ),
    },
    {
        "scenario":    "S5 – Travel planning",
        "turn":        "follow-up-1",
        "session_key": "s5-travel",
        "question":    "The hotel I want to book costs $220/night. Can I book it?",
        "note": (
            "Sydney cap is $180 (outside NA). $220 > $180 → needs VP approval. "
            "Bot should recall Sydney was established in the opener "
            "and flag VP approval requirement."
        ),
    },
    {
        "scenario":    "S5 – Travel planning",
        "turn":        "follow-up-2",
        "session_key": "s5-travel",
        "question":    "My VP approved the $220 hotel. What's my revised total reimbursement for the whole trip?",
        "note": (
            "Hotel now approved at $220: 2×$220=$440. "
            "Meals unchanged: $225. "
            "Total: $665. Bot should recall the 2 nights and 3 days from the opener."
        ),
    },
    {
        "scenario":    "S5 – Travel planning",
        "turn":        "follow-up-3",
        "session_key": "s5-travel",
        "question":    (
            "The conference registration is $1,500. Since this is a learning event, "
            "can I combine that with my certification budget for the year? "
            "I've already used $800 of my certification allowance."
        ),
        "note": (
            "Cert budget: $2,000/year, $800 used → $1,200 remaining. "
            "Conference: reimbursable separately (not from cert budget) if learning summary submitted. "
            "Can combine both in same year. Bot should distinguish the two pots "
            "and confirm the $1,500 conference comes from a separate conference budget."
        ),
    },
    {
        "scenario":    "S5 – Travel planning",
        "turn":        "follow-up-4",
        "session_key": "s5-travel",
        "question":    (
            "Perfect. Give me a final breakdown of everything I can claim for this Sydney trip "
            "including the conference fee."
        ),
        "note": (
            "Should recall the full session: "
            "Hotel: $440 (2×$220, VP approved). "
            "Meals: $225 (3×$75). "
            "Conference fee: $1,500 (separate pot, pending learning summary). "
            "Grand total: $2,165. "
            "Should also mention the remaining cert budget ($1,200) is separate."
        ),
    },
]


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def post_query(client: httpx.Client, question: str, conversation_id: str) -> dict:
    resp = client.post(
        f"{MAIN_AGENT_URL}/query",
        json={
            "text":            question,
            "user_id":         "memory-eval-user",
            "conversation_id": conversation_id,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def check_health(client: httpx.Client) -> bool:
    try:
        r = client.get(f"{MAIN_AGENT_URL}/health/live", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


# ── Report writers ─────────────────────────────────────────────────────────────

def write_json(results: list[dict], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def write_txt(results: list[dict], path: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("SHORT-TERM MEMORY EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Total turns: {len(results)}")
    passed  = sum(1 for r in results if r["status"] == "success")
    failed  = sum(1 for r in results if r["status"] == "failure")
    errored = sum(1 for r in results if r["status"] == "error")
    oos     = sum(1 for r in results if r["status"] == "out_of_scope")
    lines.append(f"success={passed}  failure={failed}  error={errored}  out_of_scope={oos}")
    lines.append("=" * 80)

    current_scenario = ""
    for i, r in enumerate(results, 1):
        if r["scenario"] != current_scenario:
            current_scenario = r["scenario"]
            lines.append("")
            lines.append("─" * 80)
            lines.append(f"  {current_scenario.upper()}")
            lines.append(f"  session_id: {r.get('conversation_id', '?')}")
            lines.append("─" * 80)

        turn_label = r.get("turn", f"turn-{i}")
        lines.append("")
        lines.append(
            f"  [{turn_label.upper()}]  [{r['status'].upper()}]  "
            f"confidence={r.get('confidence', 0):.2f}  "
            f"tools={r.get('tools_used', [])}"
        )
        lines.append(f"  NOTE: {r.get('note', '')}")
        lines.append("")
        lines.append(f"  Q: {r['question']}")
        lines.append("")
        lines.append("  A:")
        for line in r.get("answer", "(no answer)").splitlines():
            lines.append(f"     {line}")
        if r.get("sources"):
            lines.append("")
            lines.append("  SOURCES:")
            for s in r["sources"]:
                title = s.get("title") or s.get("doc_name") or s.get("source", "?")
                lines.append(f"    • {title}")
        if r.get("error"):
            lines.append(f"  ERROR: {r['error']}")
        lines.append("")
        lines.append("  " + "·" * 76)

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main runner ────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    n = len(TESTS)
    scenarios = len({t["session_key"] for t in TESTS})

    print(f"\n{'='*60}")
    print("  Short-Term Memory Evaluation")
    print(f"  {scenarios} scenarios  |  {n} total turns")
    print(f"  Server: {MAIN_AGENT_URL}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    with httpx.Client() as c:
        if not check_health(c):
            print(f"ERROR: Main agent not reachable at {MAIN_AGENT_URL}/health/live")
            print("       Start all three servers before running this script.")
            sys.exit(1)
    print("✓ Server is alive\n")

    # Assign one stable conversation_id per session_key.
    session_map: dict[str, str] = {}
    results: list[dict] = []

    with httpx.Client() as client:
        for idx, test in enumerate(TESTS, 1):
            session_key = test["session_key"]
            if session_key not in session_map:
                session_map[session_key] = str(uuid.uuid4())
            conversation_id = session_map[session_key]

            scenario = test["scenario"]
            turn     = test["turn"]
            question = test["question"]

            print(f"[{idx:02d}/{n}] {scenario:<30} | {turn:<12} | {question[:45]}…")

            result: dict = {
                "index":           idx,
                "scenario":        scenario,
                "turn":            turn,
                "session_key":     session_key,
                "conversation_id": conversation_id,
                "question":        question,
                "note":            test.get("note", ""),
                "answer":          "",
                "status":          "error",
                "confidence":      0.0,
                "attempts_used":   0,
                "tools_used":      [],
                "sources":         [],
                "error":           None,
            }

            try:
                data = post_query(client, question, conversation_id)
                result["answer"]        = data.get("answer", "")
                result["status"]        = data.get("status", "error")
                result["confidence"]    = round(float(data.get("confidence", 0.0)), 4)
                result["attempts_used"] = int(data.get("attempts_used", 0))
                result["tools_used"]    = data.get("tools_used", [])
                result["sources"]       = data.get("sources", [])

                icon = {"success": "✓", "failure": "⚠", "error": "✗", "out_of_scope": "○"}.get(
                    result["status"], "?"
                )
                print(f"         {icon} {result['status']}  conf={result['confidence']:.2f}")

            except Exception as exc:
                result["error"] = str(exc)
                print(f"         ✗ FAILED: {exc}")

            results.append(result)

            if idx < n:
                time.sleep(REQUEST_DELAY)

    json_path = OUTPUT_DIR / "memory_eval.json"
    txt_path  = OUTPUT_DIR / "memory_eval.txt"
    write_json(results, json_path)
    write_txt(results, txt_path)

    passed  = sum(1 for r in results if r["status"] == "success")
    failed  = sum(1 for r in results if r["status"] == "failure")
    oos     = sum(1 for r in results if r["status"] == "out_of_scope")

    print(f"\n{'='*60}")
    print(f"  Done — {n} turns across {scenarios} scenarios")
    print(f"  success={passed}  failure={failed}  out_of_scope={oos}")
    print(f"  Results: {txt_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
