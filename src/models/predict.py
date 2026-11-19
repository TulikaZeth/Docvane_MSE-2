"""Inference pipeline for document classification and field extraction."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from src.data.extraction import extract_regex_fields, extract_text
from src.monitoring.logger import log_prediction

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Document classifier using Calibrated Linear SVM and TF-IDF."""

    def __init__(
        self,
        model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib",
        vectorizer_path: Path = PROJECT_ROOT / "artifacts" / "vectorizer.joblib",
        low_conf_threshold: float = 0.65,
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.low_conf_threshold = low_conf_threshold
        self._load_artifacts()

    def _load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer artifact not found at {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.classes = getattr(self.model, "classes_", [])

    def predict_text(self, text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Classify raw text, get calibrated probability confidence, extract fields, and log."""
        if not text or not text.strip():
            raise ValueError("Input document text is empty.")

        # Vectorize
        features = self.vectorizer.transform([text])

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            top_idx = int(np.argmax(probs))
            category = str(self.classes[top_idx])
            confidence = float(probs[top_idx])
            all_probs = {str(cls): round(float(p), 4) for cls, p in zip(self.classes, probs)}
        else:
            # Fallback if raw SVM without calibration
            pred_class = self.model.predict(features)[0]
            category = str(pred_class)
            confidence = 1.0
            all_probs = {category: 1.0}

        low_confidence = bool(confidence < self.low_conf_threshold)

        # Extract structured fields with regex based on predicted category
        extracted = extract_regex_fields(text, category=category)

        # Log to monitoring JSONL
        log_prediction(
            predicted_category=category,
            confidence=confidence,
            text_length=len(text),
            filename=filename or "text_input",
            extracted_fields=extracted,
            low_confidence=low_confidence,
        )

        return {
            "category": category,
            "confidence": round(confidence, 4),
            "low_confidence": low_confidence,
            "probabilities": all_probs,
            "extracted_fields": extracted,
            "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
        }

    def predict_file(
        self, file_source: Union[str, Path, bytes], filename: str = ""
    ) -> Dict[str, Any]:
        """Extract text from PDF or TXT and classify."""
        extracted_text = extract_text(file_source, filename=filename)
        return self.predict_text(extracted_text, filename=filename)


# Default singleton
_classifier_instance: Optional[DocumentClassifier] = None


def get_classifier() -> DocumentClassifier:
    """Get or create singleton instance of DocumentClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DocumentClassifier()
    return _classifier_instance


if __name__ == "__main__":
    classifier = get_classifier()
    sample = "Invoice number INV-5544, Total Due: $1,450.00 for cloud computing services."
    res = classifier.predict_text(sample)
    print("Inference Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
