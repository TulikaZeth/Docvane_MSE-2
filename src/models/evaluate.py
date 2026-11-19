"""Model evaluation and threshold validation script."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_saved_model(
    model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib",
    vectorizer_path: Path = PROJECT_ROOT / "artifacts" / "vectorizer.joblib",
    dataset_path: Path = PROJECT_ROOT / "data" / "processed" / "dataset.csv",
    params_path: Path = PROJECT_ROOT / "params.yaml",
) -> Dict[str, Any]:
    """Evaluate saved model artifact on test split and return metrics."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found at {vectorizer_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load parameters
    params = {}
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

    test_size = params.get("data", {}).get("test_size", 0.25)
    random_state = params.get("base", {}).get("random_state", 42)
    min_macro_f1 = params.get("train", {}).get("min_macro_f1_threshold", 0.80)

    # Load artifacts
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    df = pd.read_csv(dataset_path)

    # Perform stratified split to match test set
    # Convert series to standard numpy object arrays to avoid PyArrow chunked indexing issues
    docs = np.array(df["document_text"].astype(str).tolist(), dtype=object)
    labels = np.array(df["label"].astype(str).tolist(), dtype=object)

    _, X_test_raw, _, y_test = train_test_split(
        docs,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    X_test_vec = vectorizer.transform(X_test_raw)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    results = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "macro_f1": float(macro_f1),
        "per_class": {
            k: v for k, v in report_dict.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "test_sample_count": len(y_test),
        "threshold_met": bool(macro_f1 >= min_macro_f1),
    }

    # Save reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(reports_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Evaluation finished:\n{report_text}")
    logger.info(f"Macro F1: {macro_f1:.4f} (Threshold: {min_macro_f1})")

    if not results["threshold_met"]:
        raise ValueError(
            f"Model macro F1-score {macro_f1:.4f} did not meet required threshold of {min_macro_f1}!"
        )

    return results


if __name__ == "__main__":
    evaluate_saved_model()
