"""Ingestion pipeline to parse raw documents into structured dataset."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.extraction import extract_text

logger = logging.getLogger(__name__)


def ingest_raw_folder(
    raw_dir: Path,
    output_csv: Optional[Path] = None,
    allowed_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ingest files from raw_dir subfolders structured by category:
      raw_dir/
        invoice/
          doc1.pdf, doc2.txt
        resume/
          ...
    Extracts text and constructs a DataFrame with [document_text, label, source_file].
    """
    raw_path = Path(raw_dir)
    records = []

    if not raw_path.exists():
        logger.warning(f"Raw directory {raw_path} does not exist.")
        return pd.DataFrame(columns=["document_text", "label", "source_file"])

    categories = [
        d for d in raw_path.iterdir()
        if d.is_dir() and (allowed_categories is None or d.name in allowed_categories)
    ]

    for cat_dir in categories:
        category_name = cat_dir.name
        for doc_file in cat_dir.iterdir():
            if doc_file.suffix.lower() in [".pdf", ".txt"]:
                try:
                    text = extract_text(doc_file, filename=doc_file.name)
                    if text.strip():
                        records.append({
                            "document_text": text.strip(),
                            "label": category_name,
                            "source_file": doc_file.name,
                        })
                except Exception as e:
                    logger.error(f"Error extracting text from {doc_file}: {e}")

    df = pd.DataFrame(records)
    if output_csv is not None and not df.empty:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Ingested {len(df)} documents and saved to {output_path}")

    return df
