"""Dataset validation and quality check utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["document_text", "label"]
DEFAULT_CATEGORIES = ["invoice", "resume", "certificate", "bank_statement"]


def validate_dataset(
    file_path: Union[str, Path],
    expected_categories: Optional[List[str]] = None,
    min_samples_per_class: int = 5,
) -> Dict[str, Union[int, float, dict, bool]]:
    """
    Validate dataset structure, required columns, missing values,
    and class balance. Raises ValueError if critical requirements are violated.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    # 1. Column check
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}. Found: {list(df.columns)}")

    # 2. Check nulls
    null_counts = df[EXPECTED_COLUMNS].isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        raise ValueError(f"Found missing values in dataset: {null_counts}")

    # 3. Text length validation
    empty_text_count = (df["document_text"].astype(str).str.strip() == "").sum()
    if empty_text_count > 0:
        raise ValueError(f"Found {empty_text_count} rows with empty document_text")

    # 4. Class distribution validation
    categories = expected_categories or DEFAULT_CATEGORIES
    counts = df["label"].value_counts().to_dict()

    missing_classes = [c for c in categories if c not in counts or counts[c] == 0]
    if missing_classes:
        raise ValueError(f"Dataset missing required document categories: {missing_classes}")

    underrepresented = {c: counts[c] for c in categories if counts.get(c, 0) < min_samples_per_class}
    if underrepresented:
        raise ValueError(
            f"Classes below minimum sample threshold ({min_samples_per_class}): {underrepresented}"
        )

    summary = {
        "valid": True,
        "total_rows": len(df),
        "class_distribution": counts,
        "null_counts": null_counts,
        "empty_text_rows": int(empty_text_count),
        "avg_character_length": float(df["document_text"].astype(str).str.len().mean()),
    }
    logger.info(f"Dataset validation passed. Summary: {summary}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/dataset.csv"
    try:
        res = validate_dataset(dataset_file)
        print("Validation Successful:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    except Exception as err:
        print(f"Validation Failed: {err}")
        sys.exit(1)
