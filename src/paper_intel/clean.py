from __future__ import annotations

"""Data cleaning utilities for research-paper-intelligence.

This module reads raw paper metadata from data/raw/papers.csv, applies a
sequence of preprocessing steps to textual fields (title and abstract), and
writes the cleaned dataset to data/processed/cleaned_papers.csv.

Preprocessing steps (applied in order):
- Remove duplicate records (by all columns)
- Drop rows with missing abstracts
- Normalize Unicode characters
- Remove HTML tags
- Normalize whitespace
- Convert text to lowercase
- Remove punctuation
- Remove stop words
- Trim leading and trailing spaces

Each processing step logs its action and the number of affected rows.

Usage:
    PYTHONPATH=src python -m paper_intel.clean --in data/raw/papers.csv --out data/processed/cleaned_papers.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import string
import unicodedata
from html import unescape
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Default file paths
DEFAULT_INPUT = Path("data") / "raw" / "papers.csv"
DEFAULT_OUTPUT = Path("data") / "processed" / "cleaned_papers.csv"


# --------------------
# NLTK utilities
# --------------------


def _ensure_nltk_stopwords(language: str = "english") -> None:
    """Ensure the NLTK stopwords corpus is available, downloading if necessary.

    Args:
        language: Stopword language to ensure.
    """
    try:
        stopwords.words(language)
    except LookupError:
        logger.info("NLTK stopwords not found; downloading...")
        nltk.download("stopwords", quiet=True)


# --------------------
# Low-level text helpers
# --------------------


def _normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFC form.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    return unicodedata.normalize("NFC", text)


def _remove_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities.

    This is a simple HTML stripper suitable for typical abstracts.
    """
    if not text:
        return text
    # Unescape first (e.g. &amp; -> &)
    text = unescape(text)
    # Remove tags
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters to a single space."""
    return re.sub(r"\s+", " ", text)


def _to_lower(text: str) -> str:
    return text.lower()


def _remove_punctuation(text: str) -> str:
    """Remove ASCII punctuation characters.

    Keeps internal whitespace; punctuation is removed via translation table.
    """
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def _remove_stopwords(text: str, language: str = "english") -> str:
    """Remove stopwords from a whitespace-tokenized text.

    Args:
        text: Input text (assumed pre-normalized and punctuation removed).
        language: Stopword language.
    """
    if not text:
        return text
    _ensure_nltk_stopwords(language)
    stops = set(stopwords.words(language))
    tokens = [t for t in text.split() if t not in stops]
    return " ".join(tokens)


def _trim(text: str) -> str:
    return text.strip()


# --------------------
# Dataframe-level operations
# --------------------


def load_raw(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load raw papers CSV into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the CSV contents.

    Raises:
        FileNotFoundError: If the input file does not exist.
        pd.errors.EmptyDataError: If the CSV is empty.
    """
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates()
    removed = before - len(df2)
    logger.info("Removed %d duplicate rows", removed)
    return df2


def drop_missing_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    if "abstract" not in df.columns:
        logger.warning("No 'abstract' column found in dataframe; skipping drop_missing_abstracts")
        return df
    before = len(df)
    df2 = df[df["abstract"].notna() & (df["abstract"].astype(str).str.strip() != "")].copy()
    removed = before - len(df2)
    logger.info("Dropped %d rows with missing or empty abstracts", removed)
    return df2


def _apply_text_pipeline(text: Optional[str]) -> str:
    """Apply the sequence of preprocessing steps to a single text value.

    The sequence is:
      - Normalize unicode
      - Remove HTML tags
      - Normalize whitespace
      - Convert to lowercase
      - Remove punctuation
      - Remove stopwords
      - Trim spaces

    Args:
        text: Input text (may be None or NaN-equivalent).

    Returns:
        Cleaned text string.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = _normalize_unicode(s)
    s = _remove_html(s)
    s = _normalize_whitespace(s)
    s = _to_lower(s)
    s = _remove_punctuation(s)
    s = _normalize_whitespace(s)
    s = _remove_stopwords(s)
    s = _trim(s)
    return s


def clean_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Clean specified text columns in the DataFrame.

    Args:
        df: Input DataFrame.
        columns: Column names to clean (if present).

    Returns:
        DataFrame with cleaned columns appended with suffix '_clean'.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found; skipping", col)
            continue
        logger.info("Cleaning column '%s'", col)
        df[f"{col}_clean"] = df[col].apply(_apply_text_pipeline)
        logger.info("Finished cleaning column '%s'", col)
    return df


def save_clean(df: pd.DataFrame, path: Path = DEFAULT_OUTPUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data to %s (%d rows)", path, len(df))


def process(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    text_columns: Optional[List[str]] = None,
) -> int:
    """Run the full cleaning pipeline: load, clean, and save.

    Args:
        input_path: Optional input CSV path.
        output_path: Optional output CSV path.
        text_columns: List of text column names to process; defaults to ['title', 'abstract'].

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    in_path = input_path or DEFAULT_INPUT
    out_path = output_path or DEFAULT_OUTPUT
    cols = text_columns or ["title", "abstract"]

    try:
        df = load_raw(in_path)
        df = remove_duplicates(df)
        df = drop_missing_abstracts(df)
        df = clean_text_columns(df, cols)
        save_clean(df, out_path)
        return 0
    except FileNotFoundError as exc:
        logger.exception("Input file not found: %s", exc)
        return 2
    except pd.errors.EmptyDataError as exc:
        logger.exception("Empty input CSV: %s", exc)
        return 3
    except Exception as exc:
        logger.exception("Unexpected error during cleaning: %s", exc)
        return 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser("clean-papers")
    parser.add_argument("--in", dest="input", type=str, default=None, help="Input CSV path")
    parser.add_argument("--out", dest="output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    exit_code = process(
        input_path=Path(args.input) if args.input else None, output_path=Path(args.output) if args.output else None
    )
    raise SystemExit(exit_code)
