"""
Seed the local ChromaDB collection with sample documents for testing.

Usage:
    python scripts/seed_local_docs.py

This script adds a small set of sample documents across the domains defined
in shared/models.py. Edit SAMPLE_DOCS below to add your own test content.

After running this, the three servers will return real answers (not just
"no documents found") when you POST /query.

Domain assignment guide:
  hr    — onboarding, leave, payroll, bonuses, travel expenses, wellness, L&D
  it    — laptops, remote equipment, MFA, cybersecurity, software procurement
  ops   — procurement approval, PMO/project governance, facilities, finance ops
  legal — contracts, data retention, GDPR/PII, compliance obligations
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from shared.models import Domain
from tools.local_search import add_documents, _get_collection

# ── Sample documents ──────────────────────────────────────────────────────────
# Domains are set per-document to match what the LLM classifier will route to.
# All 33 docs span hr / it / ops / legal so cross-domain queries succeed.

SAMPLE_DOCS = [
    # ── HR: onboarding, leave, payroll, benefits, travel, L&D ────────────────
    {
        "id":          "doc-001",
        "content":     (
            "The onboarding process for new employees consists of three phases: "
            "documentation submission (day 1), system access setup (days 2-3), "
            "and department orientation (week 1). All steps must be completed "
            "within the first five business days."
        ),
        "domain":      Domain.HR.value,
        "title":       "Employee Onboarding Guide",
        "doc_name":    "HR Policy Manual v2.4",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 12,
    },
    {
        "id":          "doc-002",
        "content":     (
            "Expense reimbursement requests must be submitted within 30 days of "
            "incurring the expense. Receipts are required for any item over $25. "
            "Submit via the Concur portal and select your cost centre code. "
            "Approval from your line manager is required for amounts over $500."
        ),
        "domain":      Domain.HR.value,
        "title":       "Expense Reimbursement Policy",
        "doc_name":    "Finance Policy Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 8,
    },
    {
        "id":          "doc-003",
        "content":     (
            "To request annual leave, log into the HR portal and submit a leave "
            "request at least 2 weeks in advance for periods longer than 3 days. "
            "Public holidays are automatically excluded. Your manager will receive "
            "an approval notification within 24 hours."
        ),
        "domain":      Domain.HR.value,
        "title":       "Annual Leave Policy",
        "doc_name":    "HR Policy Manual v2.4",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 34,
    },
    {
        "id":          "doc-004",
        "content":     (
            "Employees become eligible for annual performance bonuses only after "
            "completing 6 months of continuous employment. Bonus payouts occur in "
            "March and are calculated using the previous calendar year's performance score."
        ),
        "domain":      Domain.HR.value,
        "title":       "Bonus Eligibility Policy",
        "doc_name":    "Compensation Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 15,
    },
    {
        "id":          "doc-005",
        "content":     (
            "Contractors are not eligible for annual bonuses, wellness allowances, "
            "or employee stock purchase programmes."
        ),
        "domain":      Domain.HR.value,
        "title":       "Contractor Benefits Restrictions",
        "doc_name":    "Contractor Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 7,
    },
    {
        "id":          "doc-006",
        "content":     (
            "Employees may carry forward a maximum of 5 unused annual leave days "
            "into the next calendar year. Carried-forward leave expires after March 31."
        ),
        "domain":      Domain.HR.value,
        "title":       "Leave Carry Forward Rules",
        "doc_name":    "HR Policy Manual v2.4",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 39,
    },
    {
        "id":          "doc-007",
        "content":     (
            "The company provides a yearly wellness allowance of $600. Claims may "
            "include gym memberships, fitness classes, or ergonomic equipment. "
            "Unused allowance does not roll over."
        ),
        "domain":      Domain.HR.value,
        "title":       "Wellness Allowance Policy",
        "doc_name":    "Benefits Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 22,
    },
    {
        "id":          "doc-012",
        "content":     (
            "Business-class travel is permitted only for flights longer than 8 hours "
            "or for executive-level employees."
        ),
        "domain":      Domain.HR.value,
        "title":       "Air Travel Policy",
        "doc_name":    "Travel Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 13,
    },
    {
        "id":          "doc-013",
        "content":     (
            "Hotel bookings are capped at $250 per night in North America and "
            "$180 per night in all other regions unless approved by a Vice President."
        ),
        "domain":      Domain.HR.value,
        "title":       "Hotel Accommodation Policy",
        "doc_name":    "Travel Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 18,
    },
    {
        "id":          "doc-014",
        "content":     (
            "Meals during business travel may be reimbursed up to $75 per day. "
            "Alcoholic beverages are not reimbursable."
        ),
        "domain":      Domain.HR.value,
        "title":       "Meal Reimbursement Policy",
        "doc_name":    "Travel Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 21,
    },
    {
        "id":          "doc-018",
        "content":     (
            "Conference attendance expenses may be reimbursed if the employee "
            "submits a learning summary within 10 business days after returning."
        ),
        "domain":      Domain.HR.value,
        "title":       "Conference Attendance Policy",
        "doc_name":    "Learning & Development Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 19,
    },
    {
        "id":          "doc-019",
        "content":     (
            "The company reimburses up to $2,000 per employee annually for approved "
            "professional certification exams and training."
        ),
        "domain":      Domain.HR.value,
        "title":       "Certification Reimbursement Policy",
        "doc_name":    "Learning & Development Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 11,
    },
    {
        "id":          "doc-033",
        "content":     (
            "Employees may combine certification reimbursement and conference "
            "reimbursement benefits within the same year, subject to their respective limits."
        ),
        "domain":      Domain.HR.value,
        "title":       "Learning Benefit Combination Rules",
        "doc_name":    "Learning & Development Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 27,
    },
    # ── IT: laptops, remote equipment, MFA, cybersecurity, software ───────────
    {
        "id":          "doc-008",
        "content":     (
            "Laptop replacement is permitted every 4 years unless the device fails "
            "hardware diagnostics or receives executive approval."
        ),
        "domain":      Domain.IT.value,
        "title":       "Laptop Lifecycle Policy",
        "doc_name":    "IT Asset Standards",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 11,
    },
    {
        "id":          "doc-009",
        "content":     (
            "Employees working remotely more than 3 days per week are eligible for "
            "one additional monitor and a docking station."
        ),
        "domain":      Domain.IT.value,
        "title":       "Remote Work Equipment Policy",
        "doc_name":    "IT Asset Standards",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 14,
    },
    {
        "id":          "doc-010",
        "content":     (
            "Multi-factor authentication is mandatory for all corporate systems. "
            "Accounts without MFA enabled are automatically suspended after 7 days."
        ),
        "domain":      Domain.IT.value,
        "title":       "MFA Enforcement Policy",
        "doc_name":    "Cybersecurity Standard",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 5,
    },
    {
        "id":          "doc-011",
        "content":     (
            "Critical security incidents must be reported to the security team within "
            "1 hour of detection. High severity incidents must be reported within 4 hours."
        ),
        "domain":      Domain.IT.value,
        "title":       "Incident Reporting Standard",
        "doc_name":    "Cybersecurity Standard",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 27,
    },
    {
        "id":          "doc-017",
        "content":     (
            "New software subscriptions exceeding $10,000 annually require both "
            "IT architecture review and procurement approval."
        ),
        "domain":      Domain.IT.value,
        "title":       "Software Procurement Policy",
        "doc_name":    "Procurement Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 15,
    },
    {
        "id":          "doc-020",
        "content":     (
            "Employees must complete annual security awareness training before "
            "October 31 each year."
        ),
        "domain":      Domain.IT.value,
        "title":       "Mandatory Security Training",
        "doc_name":    "Compliance Requirements",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 4,
    },
    # ── OPS: procurement, PMO/project governance, facilities, finance ops ─────
    {
        "id":          "doc-015",
        "content":     (
            "Purchase requests above $5,000 require department head approval. "
            "Requests above $25,000 require CFO approval."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Procurement Approval Matrix",
        "doc_name":    "Procurement Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 6,
    },
    {
        "id":          "doc-016",
        "content":     (
            "Approved vendors must complete a security assessment if they will "
            "store, process, or transmit company data."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Vendor Security Assessment Policy",
        "doc_name":    "Vendor Management Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 12,
    },
    {
        "id":          "doc-021",
        "content":     (
            "Projects exceeding $100,000 budget require monthly steering committee reviews."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Project Governance Standard",
        "doc_name":    "PMO Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 9,
    },
    {
        "id":          "doc-022",
        "content":     (
            "Project status is classified as Green, Amber, or Red. A project is "
            "considered Red if schedule variance exceeds 20 percent."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Project Status Framework",
        "doc_name":    "PMO Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 12,
    },
    {
        "id":          "doc-023",
        "content":     (
            "Any project delayed by more than 30 calendar days requires an executive review."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Executive Escalation Policy",
        "doc_name":    "PMO Handbook",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 14,
    },
    {
        "id":          "doc-024",
        "content":     (
            "Office access cards inactive for 90 consecutive days are automatically disabled."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Access Card Management",
        "doc_name":    "Facilities Manual",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 6,
    },
    {
        "id":          "doc-025",
        "content":     (
            "Emergency evacuation drills are conducted twice per year at all corporate offices."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Emergency Preparedness Policy",
        "doc_name":    "Facilities Manual",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 17,
    },
    {
        "id":          "doc-026",
        "content":     (
            "Visitors must be escorted at all times while in restricted office areas."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Visitor Access Policy",
        "doc_name":    "Facilities Manual",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 10,
    },
    {
        "id":          "doc-030",
        "content":     (
            "Quarterly revenue targets are distributed equally across all three months "
            "within a quarter unless adjusted by Finance."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Revenue Planning Standard",
        "doc_name":    "Finance Operations Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 8,
    },
    {
        "id":          "doc-031",
        "content":     (
            "Departments exceeding their approved budget by more than 10 percent must "
            "submit a variance explanation to Finance."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Budget Variance Policy",
        "doc_name":    "Finance Operations Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 12,
    },
    {
        "id":          "doc-032",
        "content":     (
            "Monthly financial reports are due on the fifth business day of the following month."
        ),
        "domain":      Domain.OPS.value,
        "title":       "Financial Reporting Schedule",
        "doc_name":    "Finance Operations Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 17,
    },
    # ── LEGAL: contracts, data retention, GDPR/PII ───────────────────────────
    {
        "id":          "doc-027",
        "content":     (
            "Customer contracts valued above $250,000 require legal review before signature."
        ),
        "domain":      Domain.LEGAL.value,
        "title":       "Contract Approval Policy",
        "doc_name":    "Legal Operations Guide",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 5,
    },
    {
        "id":          "doc-028",
        "content":     (
            "Data retention for financial records is 7 years from the date of creation."
        ),
        "domain":      Domain.LEGAL.value,
        "title":       "Financial Records Retention",
        "doc_name":    "Compliance Requirements",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 23,
    },
    {
        "id":          "doc-029",
        "content":     (
            "Personally identifiable information must be deleted within 30 days of "
            "receiving a verified deletion request unless legal retention obligations apply."
        ),
        "domain":      Domain.LEGAL.value,
        "title":       "PII Deletion Policy",
        "doc_name":    "Privacy Compliance Standard",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 16,
    },
]

# Domain breakdown:
#   hr    — 13 docs (onboarding, leave, bonuses, travel, wellness, L&D)
#   it    — 6 docs  (laptops, MFA, incidents, software procurement, security training)
#   ops   — 11 docs (procurement approval, PMO, facilities, finance operations)
#   legal — 3 docs  (contracts, data retention, PII)

if __name__ == "__main__":
    print(f"Seeding {len(SAMPLE_DOCS)} documents into ChromaDB...")
    added = add_documents(SAMPLE_DOCS)
    collection = _get_collection()
    print(f"Done. Added {added}/{len(SAMPLE_DOCS)} documents.")
    print(f"Collection '{collection.name}' now has {collection.count()} total documents.")
    print("\nDomains seeded:")
    for d in sorted(set(doc["domain"] for doc in SAMPLE_DOCS)):
        count = sum(1 for doc in SAMPLE_DOCS if doc["domain"] == d)
        print(f"  {d}: {count} docs")
