"""Data and feature drift detection using Population Stability Index (PSI)."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.monitoring.logger import get_prediction_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_psi(
    expected: np.ndarray, actual: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4
) -> float:
    """
    Calculate Population Stability Index (PSI) between expected (baseline/training)
    and actual (production/inference) distributions.
    
    Interpretation:
    - PSI < 0.1: No significant change (stable)
    - 0.1 <= PSI < 0.2: Moderate drift (monitor)
    - PSI >= 0.2: Significant drift (retrain / alert)
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, quantiles)
    # Ensure distinct bin edges
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate frequency counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Convert to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Apply epsilon smoothing to prevent division by zero or log of zero
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    # Re-normalize to 1.0 after smoothing
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    # Calculate PSI
    psi_vector = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_value = float(np.sum(psi_vector))

    return round(psi_value, 4)


def run_drift_check(
    baseline_dataset_path: Path = PROJECT_ROOT / "data" / "processed" / "dataset.csv",
    prediction_logs_path: Path = PROJECT_ROOT / "data" / "predictions" / "prediction_logs.jsonl",
    output_report_path: Path = PROJECT_ROOT / "reports" / "drift_report.json",
    psi_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Compare document lengths from training dataset against inference prediction logs
    and generate a drift monitoring report.
    """
    if not baseline_dataset_path.exists():
        logger.warning(f"Baseline dataset {baseline_dataset_path} does not exist.")
        return {"error": "Baseline dataset not found"}

    baseline_df = pd.read_csv(baseline_dataset_path)
    baseline_lengths = baseline_df["document_text"].astype(str).str.len().values

    # Load inference logs
    records = get_prediction_history(prediction_logs_path)
    if len(records) < 5:
        # Not enough production samples to compute robust PSI
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "insufficient_data",
            "message": f"Only {len(records)} prediction logs found. At least 5 required for drift calculation.",
            "psi_score": 0.0,
            "drift_detected": False,
            "baseline_samples": len(baseline_lengths),
            "production_samples": len(records),
        }
    else:
        actual_lengths = np.array([r.get("text_length", 0) for r in records if r.get("text_length", 0) > 0])
        psi_score = calculate_psi(baseline_lengths, actual_lengths)

        drift_level = "None (Stable)"
        if psi_score >= psi_threshold:
            drift_level = "Significant Drift"
        elif psi_score >= 0.1:
            drift_level = "Moderate Drift"

        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success",
            "psi_score": psi_score,
            "psi_threshold": psi_threshold,
            "drift_detected": bool(psi_score >= psi_threshold),
            "drift_level": drift_level,
            "baseline_samples": int(len(baseline_lengths)),
            "production_samples": int(len(actual_lengths)),
            "baseline_mean_length": float(np.mean(baseline_lengths)),
            "production_mean_length": float(np.mean(actual_lengths)),
        }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Drift analysis complete. PSI: {report.get('psi_score')}, Detected: {report.get('drift_detected')}")
    return report


if __name__ == "__main__":
    rep = run_drift_check()
    print("Drift Check Result:")
    print(json.dumps(rep, indent=2))
