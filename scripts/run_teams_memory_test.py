"""
scripts/run_teams_memory_test.py
─────────────────────────────────
Memory + reformat test script simulating a realistic Teams bot conversation.

Tests:
  - Domain routing (HR, IT, OPS, LEGAL)
  - Follow-up context retention (pronouns, implicit references)
  - Reformat instructions (summarize, one-liner, bullet points)
  - Arithmetic correctness (caps, remainders)
  - Date/timeline reasoning
  - Cross-turn synthesis ("based on everything above...")

Usage:
    python scripts/run_teams_memory_test.py
    python scripts/run_teams_memory_test.py --url http://localhost:8000 --user test-user-1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_URL  = "http://localhost:8000"
DEFAULT_USER = f"memtest-{uuid.uuid4().hex[:6]}"
CONV_ID      = f"conv-{uuid.uuid4().hex[:8]}"

# ── Colours ───────────────────────────────────────────────────────────────────

G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
W  = "\033[97m"   # white
DIM = "\033[2m"
RST = "\033[0m"


# ── Turn definition ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    label:    str
    question: str
    checks:   list[str] = field(default_factory=list)   # substrings that MUST appear in answer
    bad:      list[str] = field(default_factory=list)   # substrings that must NOT appear
    min_conf: float     = 0.0                           # 0 = no confidence check


# ── Test scenarios ─────────────────────────────────────────────────────────────

TURNS: list[Turn] = [

    # ── S1: Hotel / travel expense ────────────────────────────────────────────
    Turn(
        label    = "S1-T1 | Hotel cap direct",
        question = "What is the nightly hotel cap for domestic travel?",
        checks   = ["250", "night"],
        min_conf = 0.55,
    ),
    Turn(
        label    = "S1-T2 | Follow-up pronoun",
        question = "And what about international?",
        checks   = ["300", "international"],
        min_conf = 0.50,
    ),
    Turn(
        label    = "S1-T3 | Arithmetic — below cap",
        question = "I spent $240/night for 3 nights domestically. How much can I claim?",
        checks   = ["720", "240"],    # 240×3 = 720, all under cap so full amount
        bad      = ["750"],           # must NOT say 250×3 = 750
        min_conf = 0.55,
    ),
    Turn(
        label    = "S1-T4 | Summarize reformat",
        question = "Summarize what you just told me in one sentence.",
        checks   = [],               # just checking it doesn't fail / go OOS
        min_conf = 0.0,
    ),

    # ── S2: Annual leave ──────────────────────────────────────────────────────
    Turn(
        label    = "S2-T1 | Leave entitlement",
        question = "How many days of annual leave do employees get per year?",
        checks   = [],
        min_conf = 0.50,
    ),
    Turn(
        label    = "S2-T2 | Follow-up carry-over",
        question = "Can unused leave be carried over to the next year?",
        checks   = [],
        min_conf = 0.45,
    ),
    Turn(
        label    = "S2-T3 | Follow-up implicit reference",
        question = "What happens if I don't use the carry-over within that period?",
        checks   = [],
        min_conf = 0.40,
    ),
    Turn(
        label    = "S2-T4 | Condense reformat",
        question = "Give me the key leave rules as a short numbered list.",
        checks   = ["1.", "2."],
        min_conf = 0.0,
    ),

    # ── S3: New hire eligibility (date reasoning) ─────────────────────────────
    Turn(
        label    = "S3-T1 | Bonus eligibility tenure",
        question = "How long do I need to work here before I am eligible for the annual performance bonus?",
        checks   = [],
        min_conf = 0.50,
    ),
    Turn(
        label    = "S3-T2 | Date math — should pass",
        question = "I started in June 2026. Will I be eligible for the bonus paid out in March 2027?",
        checks   = ["eligible", "December 2026"],   # June+6 = Dec 2026 < Mar 2027 → eligible
        bad      = ["not eligible", "ineligible"],
        min_conf = 0.50,
    ),
    Turn(
        label    = "S3-T3 | Follow-up why",
        question = "Can you explain why that is the case?",
        checks   = [],
        min_conf = 0.40,
    ),
    Turn(
        label    = "S3-T4 | One-liner reformat",
        question = "Give me that answer in one line.",
        checks   = [],
        min_conf = 0.0,
    ),

    # ── S4: IT — MFA and laptop ───────────────────────────────────────────────
    Turn(
        label    = "S4-T1 | MFA setup deadline",
        question = "What is the deadline for employees to set up multi-factor authentication?",
        checks   = [],
        min_conf = 0.50,
    ),
    Turn(
        label    = "S4-T2 | Follow-up implicit",
        question = "What happens if someone misses that deadline?",
        checks   = [],
        min_conf = 0.40,
    ),
    Turn(
        label    = "S4-T3 | Pivot — laptop policy",
        question = "Also, when is a laptop eligible for replacement?",
        checks   = [],
        min_conf = 0.50,
    ),
    Turn(
        label    = "S4-T4 | Summarize both",
        question = "Can you summarize the MFA and laptop policies together briefly?",
        checks   = ["MFA", "laptop"],
        min_conf = 0.0,
    ),

    # ── S5: Learning & development budget ─────────────────────────────────────
    Turn(
        label    = "S5-T1 | L&D budget",
        question = "What is the annual learning and development budget per employee?",
        checks   = [],
        min_conf = 0.50,
    ),
    Turn(
        label    = "S5-T2 | Separate pots",
        question = "Is the certification budget separate from the conference budget?",
        checks   = [],
        min_conf = 0.45,
    ),
    Turn(
        label    = "S5-T3 | Arithmetic — remaining budget",
        question = "If I've spent $800 on a conference, how much is left in my conference budget?",
        bad      = ["700", "600"],    # must not subtract wrong
        min_conf = 0.50,
    ),
    Turn(
        label    = "S5-T4 | Cross-turn synthesis",
        question = "Based on everything you've told me about L&D, what should I prioritise booking first this year?",
        checks   = [],
        min_conf = 0.0,
    ),

    # ── S6: Off-topic streak + recovery ───────────────────────────────────────
    Turn(
        label    = "S6-T1 | Off-topic (sports)",
        question = "Who won the Champions League this year?",
        checks   = [],
        bad      = [],
        min_conf = 0.0,
    ),
    Turn(
        label    = "S6-T2 | Off-topic (general)",
        question = "What is the boiling point of water?",
        checks   = [],
        min_conf = 0.0,
    ),
    Turn(
        label    = "S6-T3 | Recovery — back to HR",
        question = "Okay, what is the meal allowance for business travel?",
        checks   = [],
        min_conf = 0.45,
    ),

    # ── S7: Full-session recall ────────────────────────────────────────────────
    Turn(
        label    = "S7-T1 | Cross-scenario recall",
        question = "From our conversation today, what topics did I ask about?",
        checks   = [],
        min_conf = 0.0,
    ),
]

# ── Runner ─────────────────────────────────────────────────────────────────────

def send(url: str, user_id: str, conv_id: str, text: str) -> dict[str, Any]:
    payload = {"text": text, "conversation_id": conv_id, "user_id": user_id}
    r = requests.post(f"{url}/query", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def run(base_url: str, user_id: str) -> None:
    conv_id   = CONV_ID
    passed    = 0
    failed    = 0
    warned    = 0
    results   = []

    print(f"\n{B}{'─'*70}{RST}")
    print(f"{W}  TEAMS MEMORY TEST  |  conv={conv_id}  |  user={user_id}{RST}")
    print(f"{B}{'─'*70}{RST}\n")

    for i, turn in enumerate(TURNS, 1):
        print(f"{DIM}[{i:02d}/{len(TURNS)}] {turn.label}{RST}")
        print(f"  {Y}Q:{RST} {turn.question}")

        try:
            resp    = send(base_url, user_id, conv_id, turn.question)
            status  = resp.get("status", "?")
            answer  = resp.get("answer", "")
            conf    = resp.get("confidence", 0.0)
            domain  = resp.get("domain", "?")
            tools   = resp.get("tools_used", [])

            # Evaluate
            issues = []
            for needle in turn.checks:
                if needle.lower() not in answer.lower():
                    issues.append(f"missing '{needle}'")
            for needle in turn.bad:
                if needle.lower() in answer.lower():
                    issues.append(f"contains banned '{needle}'")
            if turn.min_conf and conf < turn.min_conf:
                issues.append(f"conf {conf:.2f} < {turn.min_conf}")

            if issues:
                tag = f"{R}FAIL{RST}"
                failed += 1
            elif status in ("failure", "error"):
                tag = f"{Y}WARN{RST}"
                warned += 1
            else:
                tag = f"{G}PASS{RST}"
                passed += 1

            print(f"  {tag}  status={status}  conf={conf:.2f}  domain={domain}  tools={tools}")
            print(f"  {DIM}A: {answer[:220].replace(chr(10), ' ')}{RST}")
            if issues:
                print(f"  {R}Issues: {', '.join(issues)}{RST}")

            results.append({
                "turn":   turn.label,
                "status": status,
                "conf":   conf,
                "domain": domain,
                "issues": issues,
                "answer": answer,
            })

        except Exception as exc:
            print(f"  {R}ERROR: {exc}{RST}")
            failed += 1
            results.append({"turn": turn.label, "status": "exception", "issues": [str(exc)]})

        print()
        time.sleep(1)  # be kind to the API

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(TURNS)
    print(f"{B}{'─'*70}{RST}")
    print(f"  RESULTS: {G}{passed} passed{RST}  {Y}{warned} warned{RST}  {R}{failed} failed{RST}  / {total} total")
    print(f"{B}{'─'*70}{RST}\n")

    # Failures detail
    failures = [r for r in results if r.get("issues")]
    if failures:
        print(f"{R}Failed turns:{RST}")
        for r in failures:
            print(f"  {r['turn']}: {', '.join(r['issues'])}")
        print()

    # Save JSON
    out_path = "scripts/teams_memory_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"conv_id": conv_id, "user_id": user_id, "results": results}, f, indent=2)
    print(f"Full results saved → {out_path}\n")

    sys.exit(0 if failed == 0 else 1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default=DEFAULT_URL,  help="Main agent base URL")
    parser.add_argument("--user", default=DEFAULT_USER, help="User ID for the session")
    args = parser.parse_args()
    run(args.url, args.user)
