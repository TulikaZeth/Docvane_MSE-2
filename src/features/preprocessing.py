"""Text cleaning and feature preprocessing pipeline."""

import re
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """
    Standard text normalization:
    - Lowercase
    - Replace URLs and email addresses with tokens or strip
    - Remove excessive punctuation and special characters
    - Collapse extra whitespaces
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    cleaned = text.lower()

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Normalize email patterns
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " emailaddr ", cleaned)

    # Replace currency symbols with text token
    cleaned = re.sub(r"[\$€£₹]", " currency ", cleaned)

    # Remove non-alphanumeric characters except basic punctuation
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def create_tfidf_vectorizer(
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
    stop_words: Optional[Union[str, List[str]]] = "english",
) -> TfidfVectorizer:
    """Create a configured TfidfVectorizer instance."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        stop_words=stop_words,
        preprocessor=clean_text,
    )
