"""FastAPI service for DocVane Intelligent Document Classification & Monitoring."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.extraction import extract_regex_fields, extract_text
from src.models.predict import get_classifier
from src.monitoring.drift import run_drift_check
from src.monitoring.logger import get_monitoring_statistics, get_prediction_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocVane API",
    description="Intelligent Document Classification & Monitoring Pipeline (MLOps)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPredictRequest(BaseModel):
    text: str = Field(..., description="Raw text of the document to classify", min_length=1)


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Raw document text")
    category: Optional[str] = Field(None, description="Optional category hint (invoice, resume, certificate, bank_statement)")


@app.get("/health", summary="Health Check", tags=["General"])
def health():
    """Verify API service health and model availability."""
    classifier = None
    model_loaded = False
    try:
        classifier = get_classifier()
        model_loaded = True
    except Exception as e:
        logger.warning(f"Model not loaded: {e}")

    return {
        "status": "healthy" if model_loaded else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_loaded": model_loaded,
        "classes": list(classifier.classes) if classifier and model_loaded else [],
    }


@app.post("/predict", summary="Classify Document", tags=["Inference"])
async def predict_document(
    file: Optional[UploadFile] = File(None, description="PDF or TXT document file"),
    raw_text: Optional[str] = Query(None, description="Alternative: pass document text as query param"),
):
    """
    Classify an uploaded document (PDF or TXT) or raw text.
    Returns predicted category, calibrated confidence score, and extracted fields.
    """
    classifier = get_classifier()
    text = ""
    filename = ""

    if file is not None:
        filename = file.filename or "unknown"
        ext = Path(filename).suffix.lower()
        if ext not in [".pdf", ".txt"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Only .pdf and .txt are allowed.",
            )
        try:
            content_bytes = await file.read()
            if not content_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty.",
                )
            text = extract_text(content_bytes, filename=filename)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract text from file: {str(e)}",
            )
    elif raw_text:
        text = raw_text.strip()
        filename = "query_text"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a file upload or raw_text parameter must be provided.",
        )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Extracted document text is empty.",
        )

    try:
        result = classifier.predict_text(text, filename=filename)
        return {
            "success": True,
            "filename": filename,
            "data": result,
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/text", summary="Classify Document via JSON Text", tags=["Inference"])
def predict_raw_text(payload: TextPredictRequest):
    """Classify raw document text sent as a JSON body."""
    classifier = get_classifier()
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    try:
        result = classifier.predict_text(text, filename="json_body")
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", summary="Extract Regex Fields", tags=["Information Extraction"])
def extract_fields(payload: ExtractRequest):
    """Extract metadata/fields from document text using regex patterns."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    extracted = extract_regex_fields(text, category=payload.category)
    return {
        "success": True,
        "category_hint": payload.category,
        "extracted_fields": extracted,
        "field_count": len(extracted),
    }


@app.post("/extract/file", summary="Extract Regex Fields from File Upload", tags=["Information Extraction"])
async def extract_fields_from_file(
    file: UploadFile = File(...),
    category: Optional[str] = Query(None, description="Optional category hint"),
):
    """Extract metadata/fields from an uploaded PDF or TXT file using regex."""
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    text = extract_text(content_bytes, filename=file.filename or "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from file.")

    extracted = extract_regex_fields(text, category=category)
    return {
        "success": True,
        "filename": file.filename,
        "category_hint": category,
        "extracted_fields": extracted,
        "field_count": len(extracted),
    }


@app.get("/model-info", summary="Model Metadata", tags=["Model"])
def model_info():
    """Retrieve metadata, parameters, and evaluation metrics of the deployed model."""
    classifier = get_classifier()
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "project": "DocVane",
        "algorithm": "TF-IDF + Calibrated Linear SVM (CalibratedClassifierCV)",
        "classes": list(classifier.classes),
        "total_classes": len(classifier.classes),
        "low_confidence_threshold": classifier.low_conf_threshold,
        "vectorizer": {
            "max_features": classifier.vectorizer.max_features,
            "ngram_range": classifier.vectorizer.ngram_range,
            "sublinear_tf": classifier.vectorizer.sublinear_tf,
            "stop_words": "english",
        },
        "evaluation_metrics": metrics,
    }


@app.get("/monitoring", summary="Monitoring Statistics and Drift Report", tags=["Monitoring"])
def get_monitoring_data():
    """Return operational prediction metrics, class breakdown, and drift analysis."""
    stats = get_monitoring_statistics()
    drift = run_drift_check()

    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "statistics": stats,
        "drift_report": drift,
    }


@app.get("/monitoring/logs", summary="Recent Prediction Logs", tags=["Monitoring"])
def get_logs(limit: int = Query(50, ge=1, le=500)):
    """Fetch recent prediction logs from the JSONL storage."""
    logs = get_prediction_history(limit=limit)
    return {
        "count": len(logs),
        "logs": logs,
    }
