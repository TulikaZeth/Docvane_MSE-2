"""End-to-End DocVane Demonstration Script.
Tests sample PDF and TXT documents across all 4 categories, extracts fields,
and showcases monitoring & drift detection.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import DocumentClassifier
from src.monitoring.drift import run_drift_check
from src.monitoring.logger import get_monitoring_statistics


def run_demo():
    print("=" * 70)
    print("🚀 DOCVANE: INTELLIGENT DOCUMENT CLASSIFICATION & MONITORING PIPELINE")
    print("=" * 70)

    classifier = DocumentClassifier()

    sample_tests = [
        {
            "category": "invoice",
            "file": PROJECT_ROOT / "data" / "raw" / "invoice" / "invoice_001.pdf",
            "fallback_text": "TAX INVOICE\nInvoice #: INV-1001\nBilling Date: 12/10/2026\nDue Date: 24/11/2026\nClient: Nexus Health Inc\nTotal Amount Due: $3,250.00",
        },
        {
            "category": "resume",
            "file": PROJECT_ROOT / "data" / "raw" / "resume" / "resume_001.pdf",
            "fallback_text": "David Miller\nEmail: davidm@gmail.com | Phone: +1 (555) 304-2910\nSkills: Python, Scikit-Learn, Docker, Kubernetes, FastAPI, SQL\nExperience: Senior Machine Learning Engineer at Innovatech Labs",
        },
        {
            "category": "certificate",
            "file": PROJECT_ROOT / "data" / "raw" / "certificate" / "certificate_001.pdf",
            "fallback_text": "CERTIFICATE OF ACHIEVEMENT\nThis certificate is proudly presented to Sophia Patel\nfor successfully completing: Machine Learning & MLOps Engineering\nCredential ID: CERT-84920001\nIssue Date: 15/09/2026",
        },
        {
            "category": "bank_statement",
            "file": PROJECT_ROOT / "data" / "raw" / "bank_statement" / "bank_statement_001.pdf",
            "fallback_text": "NATIONAL RESERVE BANK\nMonthly Account Statement\nAccount Holder: Emily Davis\nAccount Number: 492019380001\nStatement Period: October 01, 2026 to October 31, 2026\nOpening Balance: $14,200.00\nClosing Balance: $18,950.00",
        },
    ]

    print("\n--- 1. BATCH PREDICTIONS & FIELD EXTRACTION ---")
    for test in sample_tests:
        expected = test["category"]
        file_path = test["file"]
        if file_path.exists():
            print(f"\n📂 Processing PDF File: {file_path.name} (Expected: {expected})")
            res = classifier.predict_file(file_path, filename=file_path.name)
        else:
            print(f"\n📝 Processing Raw Text Snippet (Expected: {expected})")
            res = classifier.predict_text(test["fallback_text"], filename=f"sample_{expected}.txt")

        print(f"  👉 Predicted Category : {res['category'].upper()}")
        print(f"  📊 Calibrated Confidence: {res['confidence'] * 100:.2f}%")
        print(f"  🔍 Extracted Fields    : {json.dumps(res['extracted_fields'], indent=4)}")

    print("\n--- 2. OPERATIONAL MONITORING & METRICS ---")
    stats = get_monitoring_statistics()
    print(f"  Total Predictions Logged: {stats['total_predictions']}")
    print(f"  Category Distribution  : {stats['category_distribution']}")
    print(f"  Average Confidence      : {stats['average_confidence'] * 100:.2f}%")
    print(f"  Low Confidence Rate     : {stats['low_confidence_percentage']}%")

    print("\n--- 3. DATA DRIFT DETECTION (PSI) ---")
    drift_rep = run_drift_check()
    print(f"  Status        : {drift_rep.get('status')}")
    print(f"  PSI Score     : {drift_rep.get('psi_score')}")
    print(f"  Drift Level   : {drift_rep.get('drift_level')}")
    print(f"  Drift Detected: {drift_rep.get('drift_detected')}")

    print("\n" + "=" * 70)
    print("✅ All DocVane components operating successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
