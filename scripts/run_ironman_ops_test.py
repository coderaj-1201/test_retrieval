"""
scripts/run_ironman_ops_test.py
────────────────────────────────
Full IRONMAN / OPS end-to-end test covering every question type:

  ✓ Greeting & farewell
  ✓ Single direct OPS questions
  ✓ Multi-question in one prompt  (decomposition tool)
  ✓ Mixed-domain batch  (some answerable, some OOS)
  ✓ Long paragraph / scenario questions
  ✓ Multi-language questions (Japanese + Tulu)
  ✓ Complex calculation question (race timeline)
  ✓ Escalation request
  ✓ Out-of-scope (weather forecast)
  ✓ Gibberish / numeric-only input
  ✓ Follow-up + reformat turns woven throughout
  ✓ Cross-session recall at the end

Usage:
    python scripts/run_ironman_ops_test.py
    python scripts/run_ironman_ops_test.py --url http://localhost:8000 --user raj-1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_URL  = "http://localhost:8000"
DEFAULT_USER = f"opstest-{uuid.uuid4().hex[:6]}"
CONV_ID      = f"conv-ops-{uuid.uuid4().hex[:8]}"

# ── Colours ────────────────────────────────────────────────────────────────────
G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
B   = "\033[94m"
CY  = "\033[96m"
W   = "\033[97m"
DIM = "\033[2m"
RST = "\033[0m"


# ── Turn definition ────────────────────────────────────────────────────────────
@dataclass
class Turn:
    label:       str
    question:    str
    expect_oos:  bool       = False   # True  → expect domain=none / status≠success
    expect_in:   bool       = True    # False → don't check domain
    checks:      list[str]  = field(default_factory=list)   # must appear in answer
    bad:         list[str]  = field(default_factory=list)   # must NOT appear
    min_conf:    float      = 0.0
    note:        str        = ""      # printed as context for the tester


# ── All turns ──────────────────────────────────────────────────────────────────
TURNS: list[Turn] = [

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 1 — Greeting
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label      = "B1-T1 | Greeting",
        question   = "Hi",
        expect_oos = True,
        note       = "Should greet back warmly, NOT say out-of-scope",
        bad        = ["out of scope", "cannot help", "sorry"],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 2 — Multi-question in single prompt (decomposition)
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label    = "B2-T1 | Multi-Q: SOP + signage list + signage rules",
        question = (
            "What is SOP?\n"
            "Can you provide me the list of signage who should assist athletes?\n"
            "What are the Rules and Guidance for Venue Signage?"
        ),
        checks   = ["SOP"],
        min_conf = 0.50,
        note     = "3 questions → should trigger decomposition tool",
    ),
    Turn(
        label    = "B2-T2 | Follow-up: summarize signage rules in one line",
        question = "Can you give me a one-line summary of the venue signage rules?",
        checks   = [],
        min_conf = 0.0,
        note     = "Reformat follow-up — must not go out-of-scope",
    ),
    Turn(
        label    = "B2-T3 | Follow-up: who is responsible for signage installation?",
        question = "And who is specifically responsible for installing that signage?",
        checks   = [],
        min_conf = 0.40,
        note     = "Implicit 'that signage' reference from previous turn",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 3 — Mixed-domain batch (some OPS/HR, some OOS)
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label    = "B3-T1 | Mixed batch: 3 OPS + 1 OOS",
        question = (
            "Can you give me the details of the below questions:\n\n"
            "1. Planning and Food Voucher collection\n"
            "2. Guidelines about sanitary installations\n"
            "3. How much will be the salary range for Ironman Managers\n"
            "4. Who is the President of US"
        ),
        checks   = [],
        min_conf = 0.0,
        note     = (
            "Q1/Q2 → OPS docs. Q3 → likely no doc (salary). Q4 → OOS. "
            "Decomposition should fire. Bot must answer what it can."
        ),
    ),
    Turn(
        label    = "B3-T2 | Follow-up: which questions could you answer?",
        question = "Which of those four questions were you able to answer fully?",
        checks   = [],
        min_conf = 0.0,
        note     = "Cross-turn recall of what was and wasn't answered",
    ),
    Turn(
        label    = "B3-T3 | Follow-up: condense food voucher info",
        question = "Condense the food voucher collection information into bullet points.",
        checks   = [],
        min_conf = 0.0,
        note     = "Reformat — must route as follow-up, not new retrieval",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 4 — Long paragraph / scenario questions
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label    = "B4-T1 | Scenario: Venue Operations setup",
        question = (
            "I am part of the event operations team and would like to understand "
            "the venue setup process for race day. Can you explain when the venue "
            "setup activities should begin, who is responsible for installing "
            "directional signage, participant flow barriers, registration counters, "
            "and medical support stations, and what checks must be completed before "
            "the venue is approved for participant access?"
        ),
        checks   = [],
        min_conf = 0.45,
        note     = "Multi-part paragraph — should use decomposition or hyde",
    ),
    Turn(
        label    = "B4-T2 | Follow-up: summarize venue setup in 3 steps",
        question = "Summarize the venue setup process in 3 key steps.",
        checks   = ["1.", "2.", "3."],
        min_conf = 0.0,
        note     = "Reformat to numbered list",
    ),
    Turn(
        label    = "B4-T3 | Scenario: Athlete Registration Operations",
        question = (
            "We are preparing for athlete check-in and packet distribution. "
            "Could you explain the complete process for athlete registration, "
            "including required identification documents, waiver verification, "
            "timing chip allocation, escalation procedures for missing registrations, "
            "and the cut-off times for athlete check-in?"
        ),
        checks   = [],
        min_conf = 0.45,
        note     = "Long scenario — decomposition expected",
    ),
    Turn(
        label    = "B4-T4 | Follow-up: most critical step in registration",
        question = "What is the most critical step in the athlete registration process?",
        checks   = [],
        min_conf = 0.0,
        note     = "Implicit reference to prior answer",
    ),
    Turn(
        label    = "B4-T5 | Follow-up: compare venue setup vs registration",
        question = "How does the venue setup timeline compare to the athlete registration cut-off times?",
        checks   = [],
        min_conf = 0.0,
        note     = "Cross-turn synthesis spanning B4-T1 and B4-T3",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 5 — Multi-language questions
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label    = "B5-T1 | Japanese: What is SOP?",
        question = "SOPって何？",
        checks   = [],
        min_conf = 0.0,
        note     = "Japanese query — classifier must still route to OPS or ask to clarify",
    ),
    Turn(
        label    = "B5-T2 | Follow-up: answer in English",
        question = "Can you answer that in English?",
        checks   = ["SOP"],
        min_conf = 0.0,
        note     = "Reformat follow-up in English",
    ),
    Turn(
        label    = "B5-T3 | Tulu: venue signage guidelines",
        question = "ಕಾರ್ಯಕ್ರಮದ ಸ್ಥಳದ ಸೂಚನಾ ಫಲಕಗಳ ಮಾರ್ಗಸೂಚಿಗಳು ಏನು?",
        checks   = [],
        min_conf = 0.0,
        note     = "Tulu script — likely classified as clarify or OPS",
    ),
    Turn(
        label    = "B5-T4 | Follow-up: English please",
        question = "Please answer in English — what are the venue signage guidelines?",
        checks   = [],
        min_conf = 0.40,
        note     = "Should now retrieve signage guidelines in English",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 6 — Complex / Keats race timeline question
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label    = "B6-T1 | Keats: full race timeline calculation",
        question = (
            "Build out a full race timeline for an IRONMAN event based on "
            "67 Male Pros, 34 Female Pros, 2089 age group athletes checked in. "
            "Sunrise is at 6:17 AM. It is a two loop swim course, 2 loop bike "
            "course, and three loop run course. I need to know the first and last "
            "athlete timeline for each segment of the race. "
            "Base this on the standard IRONMAN cut offs."
        ),
        checks   = [],
        min_conf = 0.0,
        note     = (
            "Very complex — expects cut-off times from OPS docs. "
            "Good answer uses document cut-offs (swim 2h20m, bike 10h30m, run 17h). "
            "May be low confidence if docs don't cover this fully."
        ),
    ),
    Turn(
        label    = "B6-T2 | Follow-up: condense timeline to key times only",
        question = "Can you condense that to just the key start and cut-off times in a simple format?",
        checks   = [],
        min_conf = 0.0,
        note     = "Reformat — must not trigger new retrieval",
    ),
    Turn(
        label    = "B6-T3 | Follow-up: what happens to athletes who miss the swim cut-off?",
        question = "What happens to athletes who miss the swim cut-off time?",
        checks   = [],
        min_conf = 0.40,
        note     = "Follow-up from race timeline context",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 7 — Escalation
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label      = "B7-T1 | Escalation: Can I escalate to Andrew?",
        question   = "Can I escalate to Andrew?",
        expect_oos = False,
        note       = (
            "Named escalation request — bot may offer escalation options "
            "or explain the escalation process. Should NOT go out-of-scope."
        ),
    ),
    Turn(
        label    = "B7-T2 | Follow-up: what is the escalation SLA?",
        question = "What is the SLA once I raise an escalation?",
        checks   = [],
        min_conf = 0.0,
        note     = "Follow-up on escalation process",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 8 — Out-of-scope questions
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label      = "B8-T1 | OOS: weather forecast",
        question   = "What will be the weather during next year's race?",
        expect_oos = True,
        bad        = [],
        note       = "Weather → OOS, should decline politely",
    ),
    Turn(
        label      = "B8-T2 | OOS: gibberish / numeric only",
        question   = "54748",
        expect_oos = True,
        note       = "Random number — should ask to clarify or decline",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 9 — Farewell + cross-session recall
    # ══════════════════════════════════════════════════════════════════════════
    Turn(
        label      = "B9-T1 | Farewell",
        question   = "Thank you!",
        expect_oos = True,
        bad        = ["out of scope", "cannot help", "sorry"],
        note       = "Farewell — warm response expected, no apology",
    ),
    Turn(
        label    = "B9-T2 | Cross-session recall",
        question = "What enterprise topics did we cover in this conversation?",
        checks   = [],
        min_conf = 0.0,
        note     = "Tests whether the bot can summarize across all prior turns",
    ),
]


# ── HTTP helper ────────────────────────────────────────────────────────────────

def send(url: str, user_id: str, conv_id: str, text: str) -> dict[str, Any]:
    payload = {"text": text, "conversation_id": conv_id, "user_id": user_id}
    r = requests.post(f"{url}/query", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(base_url: str, user_id: str) -> None:
    conv_id = CONV_ID
    passed = failed = warned = 0
    results: list[dict] = []

    print(f"\n{B}{'═'*72}{RST}")
    print(f"{W}  IRONMAN OPS MEMORY TEST  |  {len(TURNS)} turns  |  conv={conv_id}{RST}")
    print(f"{B}{'═'*72}{RST}\n")

    for i, turn in enumerate(TURNS, 1):
        scenario_break = turn.label.startswith("B") and turn.label.endswith("T1")
        if scenario_break:
            block = turn.label.split("-")[0]
            print(f"{CY}── {block} {'─'*60}{RST}")

        print(f"{DIM}[{i:02d}/{len(TURNS)}] {turn.label}{RST}")
        if turn.note:
            print(f"  {DIM}Note: {turn.note}{RST}")
        print(f"  {Y}Q:{RST} {turn.question[:200].replace(chr(10), ' ↵ ')}")

        try:
            resp   = send(base_url, user_id, conv_id, turn.question)
            status = resp.get("status", "?")
            answer = resp.get("answer", "")
            conf   = resp.get("confidence", 0.0)
            domain = resp.get("domain", "?")
            tools  = resp.get("tools_used", [])

            issues: list[str] = []

            # OOS expectation check
            if turn.expect_oos and status == "success" and domain not in ("", None):
                issues.append(f"expected OOS but got domain={domain}")

            # Substring checks
            for needle in turn.checks:
                if needle.lower() not in answer.lower():
                    issues.append(f"missing '{needle}'")
            for needle in turn.bad:
                if needle.lower() in answer.lower():
                    issues.append(f"banned word '{needle}' found")

            # Confidence floor
            if turn.min_conf and conf < turn.min_conf:
                issues.append(f"conf {conf:.2f} < floor {turn.min_conf}")

            if issues:
                tag = f"{R}FAIL{RST}"
                failed += 1
            elif status in ("failure", "error"):
                tag = f"{Y}WARN{RST}"
                warned += 1
            else:
                tag = f"{G}PASS{RST}"
                passed += 1

            tool_str = ",".join(tools) if tools else "—"
            print(f"  {tag}  status={status}  conf={conf:.2f}  domain={domain}  tools=[{tool_str}]")
            print(f"  {DIM}A: {answer[:250].replace(chr(10), ' ')}{RST}")
            if issues:
                for iss in issues:
                    print(f"  {R}✗ {iss}{RST}")

            results.append({
                "turn":   turn.label,
                "status": status,
                "conf":   round(conf, 3),
                "domain": domain,
                "tools":  tools,
                "issues": issues,
                "q":      turn.question[:300],
                "a":      answer[:500],
            })

        except Exception as exc:
            print(f"  {R}EXCEPTION: {exc}{RST}")
            failed += 1
            results.append({
                "turn":   turn.label,
                "status": "exception",
                "issues": [str(exc)],
                "q":      turn.question[:300],
                "a":      "",
            })

        print()
        time.sleep(1)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(TURNS)
    pct   = round(passed / total * 100) if total else 0
    print(f"\n{B}{'═'*72}{RST}")
    print(
        f"  RESULTS  {G}{passed} passed{RST}  {Y}{warned} warned{RST}  "
        f"{R}{failed} failed{RST}  / {total} total  ({pct}%)"
    )
    print(f"{B}{'═'*72}{RST}\n")

    failures = [r for r in results if r.get("issues")]
    if failures:
        print(f"{R}Failures:{RST}")
        for r in failures:
            print(f"  {r['turn']}: {', '.join(r['issues'])}")
        print()

    here = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.join(here, "ironman_ops_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"conv_id": conv_id, "user_id": user_id, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {out}\n")

    sys.exit(0 if failed == 0 else 1)


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default=DEFAULT_URL,  help="Main agent base URL")
    parser.add_argument("--user", default=DEFAULT_USER, help="User ID")
    args = parser.parse_args()
    run(args.url, args.user)
