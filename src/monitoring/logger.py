"""Inference logging and metrics tracking for DocVane."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
LOG_FILE = PREDICTIONS_DIR / "prediction_logs.jsonl"

logger = logging.getLogger(__name__)


def log_prediction(
    predicted_category: str,
    confidence: float,
    text_length: int,
    filename: Optional[str] = None,
    extracted_fields: Optional[Dict[str, Any]] = None,
    low_confidence: bool = False,
    log_path: Path = LOG_FILE,
) -> Dict[str, Any]:
    """Append a prediction record to the JSONL log file for monitoring."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "predicted_category": predicted_category,
        "confidence": round(float(confidence), 4),
        "text_length": int(text_length),
        "filename": filename or "raw_text_input",
        "extracted_fields": extracted_fields or {},
        "low_confidence": bool(low_confidence),
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to write prediction log: {e}")

    return record


def get_prediction_history(log_path: Path = LOG_FILE, limit: int = 1000) -> List[Dict[str, Any]]:
    """Read recent prediction records from jsonl."""
    if not log_path.exists():
        return []

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return records[-limit:]


def get_monitoring_statistics(
    log_path: Path = LOG_FILE, low_conf_threshold: float = 0.65
) -> Dict[str, Any]:
    """Calculate operational monitoring statistics from prediction logs."""
    records = get_prediction_history(log_path)
    total_preds = len(records)
    if total_preds == 0:
        return {
            "total_predictions": 0,
            "category_distribution": {},
            "average_confidence": 0.0,
            "low_confidence_count": 0,
            "low_confidence_percentage": 0.0,
        }

    cat_counts: Dict[str, int] = {}
    confidences: List[float] = []
    low_conf_count = 0

    for rec in records:
        cat = rec.get("predicted_category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        conf = rec.get("confidence", 0.0)
        confidences.append(conf)
        if conf < low_conf_threshold or rec.get("low_confidence", False):
            low_conf_count += 1

    return {
        "total_predictions": total_preds,
        "category_distribution": cat_counts,
        "average_confidence": round(sum(confidences) / total_preds, 4),
        "low_confidence_count": low_conf_count,
        "low_confidence_percentage": round((low_conf_count / total_preds) * 100, 2),
    }
