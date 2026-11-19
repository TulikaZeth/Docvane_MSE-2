"""Tests for data ingestion, extraction, and validation."""

from pathlib import Path
import pytest
import pandas as pd

from src.data.extraction import (
    extract_regex_fields,
    extract_text,
    extract_text_from_txt,
)
from src.data.validation import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_validate_dataset_success():
    """Verify that the generated dataset passes validation."""
    dataset_csv = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
    assert dataset_csv.exists(), "Dataset CSV must exist before running tests"
    res = validate_dataset(dataset_csv)
    assert res["valid"] is True
    assert res["total_rows"] >= 40
    for cat in ["invoice", "resume", "certificate", "bank_statement"]:
        assert cat in res["class_distribution"]


def test_validate_dataset_missing_column(tmp_path):
    """Test validation failure when required column is missing."""
    invalid_csv = tmp_path / "bad.csv"
    df = pd.DataFrame({"text": ["sample"], "label": ["invoice"]})
    df.to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(invalid_csv)


def test_validate_dataset_missing_category(tmp_path):
    """Test validation failure when a required category is missing."""
    invalid_csv = tmp_path / "missing_cat.csv"
    df = pd.DataFrame({
        "document_text": ["text1", "text2"],
        "label": ["invoice", "resume"]
    })
    df.to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="missing required document categories"):
        validate_dataset(invalid_csv)


def test_extract_text_from_txt(tmp_path):
    """Test text extraction from TXT file."""
    txt_file = tmp_path / "test.txt"
    content = "Hello DocVane document test."
    txt_file.write_text(content, encoding="utf-8")
    extracted = extract_text_from_txt(txt_file)
    assert extracted == content


def test_extract_text_from_pdf():
    """Test PDF text extraction on a sample PDF from data/raw/."""
    pdf_sample = PROJECT_ROOT / "data" / "raw" / "invoice" / "invoice_001.pdf"
    if pdf_sample.exists():
        extracted = extract_text(pdf_sample, filename="invoice_001.pdf")
        assert len(extracted) > 50
        assert "invoice" in extracted.lower() or "inv" in extracted.lower()


def test_regex_extraction_invoice():
    """Test regex field extraction on an invoice snippet."""
    sample = "Tax Invoice: INV-9902 | Date: 12/10/2026 | Total Amount: $4,500.00"
    fields = extract_regex_fields(sample, category="invoice")
    assert fields.get("invoice_number") == "INV-9902"
    assert fields.get("total_amount") == "4,500.00"


def test_regex_extraction_resume():
    """Test regex field extraction on a resume snippet."""
    sample = "John Doe\nEmail: john.doe@example.com\nPhone: +1 (555) 123-4567\nSkills: Python, Docker, Pytest"
    fields = extract_regex_fields(sample, category="resume")
    assert fields.get("email") == "john.doe@example.com"
    assert fields.get("phone") == "+1 (555) 123-4567"
    assert "Python" in fields.get("skills", [])


def test_regex_extraction_bank_statement():
    """Test regex field extraction on a bank statement snippet."""
    sample = "First National Bank\nAccount Number: 987654321012\nClosing Balance: $12,450.75"
    fields = extract_regex_fields(sample, category="bank_statement")
    assert fields.get("account_number") == "987654321012"
    assert fields.get("closing_balance") == "12,450.75"


def test_regex_extraction_certificate():
    """Test regex field extraction on a certificate snippet."""
    sample = "Certificate of Completion\nProudly presented to Alice Walker\nfor completion of: Machine Learning Specialization\nCertificate ID: CERT-8821"
    fields = extract_regex_fields(sample, category="certificate")
    assert fields.get("recipient_name") == "Alice Walker"
    assert "Machine Learning" in fields.get("course_name", "")
    assert fields.get("certificate_id") == "CERT-8821"
