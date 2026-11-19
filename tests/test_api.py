"""Integration tests for FastAPI endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns status healthy and loaded classes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["classes"]) == 4


def test_predict_text_endpoint():
    """Verify POST /predict/text with valid document snippet."""
    payload = {
        "text": (
            "TAX INVOICE\n"
            "Invoice No: INV-7721\n"
            "Total Amount: $2,800.00\n"
            "Due Date: 12/12/2026\n"
            "Bill to: Starlight Logistics"
        )
    }
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["data"]["category"] == "invoice"
    assert "confidence" in res["data"]
    assert "extracted_fields" in res["data"]
    assert res["data"]["extracted_fields"].get("invoice_number") == "INV-7721"


def test_predict_file_upload_pdf():
    """Verify POST /predict with uploaded PDF file."""
    pdf_path = PROJECT_ROOT / "data" / "raw" / "invoice" / "invoice_001.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("invoice_001.pdf", f, "application/pdf")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["category"] == "invoice"


def test_predict_invalid_file_extension():
    """Verify that uploading an unsupported file format returns 400."""
    response = client.post(
        "/predict",
        files={"file": ("unsupported.docx", b"dummy content", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_empty_document():
    """Verify empty text document returns 400 error."""
    payload = {"text": "   "}
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 400


def test_extract_endpoint():
    """Verify POST /extract returns regex parsed fields."""
    payload = {
        "text": "Curriculum Vitae: John Doe\nEmail: john@acme.com\nPhone: +1 555 019 2834\nSkills: Python, FastAPI",
        "category": "resume",
    }
    response = client.post("/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["extracted_fields"].get("email") == "john@acme.com"


def test_model_info_endpoint():
    """Verify GET /model-info returns metadata and evaluation report."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "DocVane"
    assert "TF-IDF + Calibrated Linear SVM" in data["algorithm"]
    assert len(data["classes"]) == 4


def test_monitoring_endpoint():
    """Verify GET /monitoring returns operational stats and drift report."""
    response = client.get("/monitoring")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "statistics" in data
    assert "drift_report" in data


def test_predict_scanned_pdf_file_upload(tmp_path):
    """Verify POST /predict extracts text and classifies a scanned/image PDF via OCR fallback."""
    import io
    import pymupdf as fitz
    from PIL import Image, ImageDraw
    from src.data.extraction import find_tesseract_binary

    if not find_tesseract_binary():
        pytest.skip("Tesseract OCR binary not found")

    img = Image.new("RGB", (800, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 80), "Tax Invoice: INV-9902 | Total Amount: $4,500.00 | Date: 12/10/2026", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=800, height=300)
    page.insert_image(page.rect, stream=buf.getvalue())
    pdf_path = tmp_path / "scanned_invoice.pdf"
    doc.save(str(pdf_path))
    doc.close()

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/predict?ocr_fallback=true",
            files={"file": ("scanned_invoice.pdf", f, "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "category" in data["data"]
    assert "confidence" in data["data"]
