"""
Seed the local ChromaDB collection with sample documents for testing.

Usage:
    python scripts/seed_local_docs.py

This script adds a small set of sample documents across the domains defined
in shared/models.py. Edit SAMPLE_DOCS below to add your own test content.

After running this, the three servers will return real answers (not just
"no documents found") when you POST /query.
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
# Add more entries here to improve answer quality.
# Fields:
#   id          (str, unique)
#   content     (str, the text to embed and retrieve)
#   domain      (str, must match a Domain enum value)
#   title       (str, shown in citations)
#   doc_name    (str, source attribution)
#   doc_url     (str, link shown in citations — can be empty)
#   chunk_type  (str, "paragraph" | "table" | "heading")
#   page_number (int)

SAMPLE_DOCS = [
    # ── Add domain-specific docs here to match your Domain enum values ────────
    # Example using the first two domains from the enum for illustration.
    {
        "id":          "doc-001",
        "content":     (
            "The onboarding process for new employees consists of three phases: "
            "documentation submission (day 1), system access setup (days 2-3), "
            "and department orientation (week 1). All steps must be completed "
            "within the first five business days."
        ),
        "domain":      list(Domain)[0].value,
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
        "domain":      list(Domain)[0].value,
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
        "domain":      list(Domain)[0].value,
        "title":       "Annual Leave Policy",
        "doc_name":    "HR Policy Manual v2.4",
        "doc_url":     "",
        "chunk_type":  "paragraph",
        "page_number": 34,
    },
]

# If you have more than one domain, duplicate the block above for each domain.
# All Domain enum values can be found in shared/models.py.

if __name__ == "__main__":
    print(f"Seeding {len(SAMPLE_DOCS)} documents into ChromaDB...")
    added = add_documents(SAMPLE_DOCS)
    collection = _get_collection()
    print(f"Done. Added {added}/{len(SAMPLE_DOCS)} documents.")
    print(f"Collection '{collection.name}' now has {collection.count()} total documents.")
    print("\nDomains seeded:")
    for d in set(doc["domain"] for doc in SAMPLE_DOCS):
        print(f"  {d}")
