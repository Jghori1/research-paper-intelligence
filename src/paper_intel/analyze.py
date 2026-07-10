from __future__ import annotations

"""Analysis utilities for research-paper-intelligence.

This module loads the cleaned dataset (data/processed/cleaned_papers.csv by
default) and computes a set of summary analyses. Each analysis is implemented
as an independent function and returns a pandas.DataFrame. Results are saved
as CSV files in the data/processed directory.

Computed outputs:
- Top keywords (unigrams)
- Most common authors
- Publication counts by year
- Category frequencies
- Average abstract length (words and characters)
- Most common research topics (bigrams/trigrams)

The implementations favor simplicity and reproducibility: tokenization is
whitespace-based (the cleaning step already normalizes and removes stopwords),
and n-grams are computed from cleaned text when available.
"""

from collections import Counter
import ast
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from itertools import islice
from itertools import tee
from itertools import chain

logger = logging.getLogger(__name__)

DEFAULT_CLEANED = Path("data") / "processed" / "cleaned_papers.csv"
OUT_DIR = Path("data") / "processed"


# --------------------
# Utilities
# --------------------


def _load_dataframe(path: Path = DEFAULT_CLEANED) -> pd.DataFrame:
    logger.info("Loading cleaned dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows", len(df))
    return df


def _safe_parse_list(value: object) -> List[str]:
    """Parse a column value that may be a Python list literal or a separator string.

    Handles common cases produced when writing lists to CSV: a Python list
    literal ("['A','B']") or a semicolon/comma separated string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Try to parse Python literal
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if x is not None]
            except Exception:
                pass
        # Fallback: split on semicolon or comma
        sep = ";" if ";" in s else ","
        parts = [p.strip() for p in s.split(sep) if p.strip()]
        return parts
    # Other types
    return [str(value)]


def _tokenize(text: Optional[str]) -> List[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return [t for t in str(text).split() if t]


def _ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    if n <= 0:
        return []
    # simple n-gram generator
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def _top_n_from_counter(counter: Counter, n: int) -> pd.DataFrame:
    items = counter.most_common(n)
    return pd.DataFrame(items, columns=["item", "count"])


# --------------------
# Analysis functions
# --------------------


def compute_top_keywords(df: pd.DataFrame, text_column: str = "abstract_clean", top_n: int = 50) -> pd.DataFrame:
    """Compute top unigram keywords from a cleaned text column.

    Args:
        df: Input dataframe.
        text_column: Column to extract tokens from (cleaned).
        top_n: Number of top keywords to return.

    Returns:
        DataFrame with columns ['keyword', 'count'] sorted by count desc.
    """
    logger.info("Computing top keywords from column '%s'", text_column)
    counter: Counter = Counter()
    for text in df.get(text_column, pd.Series(dtype=object)):
        tokens = _tokenize(text)
        counter.update(tokens)
    df_out = _top_n_from_counter(counter, top_n)
    df_out = df_out.rename(columns={"item": "keyword"})
    logger.info("Top keywords computed (%d entries)", len(df_out))
    return df_out


def compute_most_common_authors(df: pd.DataFrame, authors_column: str = "authors", top_n: int = 50) -> pd.DataFrame:
    """Compute the most frequent authors in the dataset.

    The authors column may be a list literal or a separator string; _safe_parse_list
    normalizes it.
    """
    logger.info("Computing most common authors from column '%s'", authors_column)
    counter: Counter = Counter()
    for val in df.get(authors_column, pd.Series(dtype=object)):
        for name in _safe_parse_list(val):
            counter.update([name])
    df_out = _top_n_from_counter(counter, top_n)
    df_out = df_out.rename(columns={"item": "author"})
    logger.info("Most common authors computed (%d entries)", len(df_out))
    return df_out


def publication_counts_by_year(df: pd.DataFrame, date_column: str = "published_date") -> pd.DataFrame:
    """Count publications grouped by year extracted from a date-like column.

    Non-parseable dates are ignored.
    """
    logger.info("Computing publication counts by year from '%s'", date_column)
    ser = df.get(date_column, pd.Series(dtype=object)).dropna().astype(str)
    years = []
    for v in ser:
        try:
            y = pd.to_datetime(v, errors="coerce").year
            if pd.notna(y):
                years.append(int(y))
        except Exception:
            continue
    counter = Counter(years)
    items = sorted(counter.items())
    df_out = pd.DataFrame(items, columns=["year", "count"]).sort_values("year")
    logger.info("Publication counts by year computed (%d years)", len(df_out))
    return df_out


def category_frequencies(df: pd.DataFrame, categories_column: str = "categories") -> pd.DataFrame:
    """Compute frequency of categories across the dataset.

    The categories column may be a list literal or a separator string.
    """
    logger.info("Computing category frequencies from '%s'", categories_column)
    counter: Counter = Counter()
    for val in df.get(categories_column, pd.Series(dtype=object)):
        for cat in _safe_parse_list(val):
            counter.update([cat])
    df_out = _top_n_from_counter(counter, 1000)
    df_out = df_out.rename(columns={"item": "category"})
    logger.info("Category frequencies computed (%d categories)", len(df_out))
    return df_out


def average_abstract_length(df: pd.DataFrame, text_column: str = "abstract_clean") -> pd.DataFrame:
    """Compute average abstract length in words and characters.

    Returns a one-row dataframe with columns: avg_words, avg_chars, median_words, median_chars
    """
    logger.info("Computing average abstract length for '%s'", text_column)
    words_counts: List[int] = []
    chars_counts: List[int] = []
    for text in df.get(text_column, pd.Series(dtype=object)):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        s = str(text)
        tokens = _tokenize(s)
        words_counts.append(len(tokens))
        chars_counts.append(len(s))
    if not words_counts:
        logger.warning("No abstracts found to compute lengths")
        return pd.DataFrame([{"avg_words": 0, "avg_chars": 0, "median_words": 0, "median_chars": 0}])
    ser_words = pd.Series(words_counts)
    ser_chars = pd.Series(chars_counts)
    df_out = pd.DataFrame(
        [
            {
                "avg_words": float(ser_words.mean()),
                "avg_chars": float(ser_chars.mean()),
                "median_words": float(ser_words.median()),
                "median_chars": float(ser_chars.median()),
            }
        ]
    )
    logger.info("Average abstract length computed")
    return df_out


def most_common_topics(
    df: pd.DataFrame, text_column: str = "abstract_clean", ngram_range: Tuple[int, int] = (2, 3), top_n: int = 50
) -> pd.DataFrame:
    """Compute most common n-gram topics (bigrams and trigrams by default).

    Args:
        df: Input dataframe.
        text_column: Column to extract tokens from.
        ngram_range: (min_n, max_n) inclusive.
        top_n: Number of top n-grams to return.
    """
    logger.info("Computing most common topics (ngrams=%s) from '%s'", ngram_range, text_column)
    counter: Counter = Counter()
    min_n, max_n = ngram_range
    for text in df.get(text_column, pd.Series(dtype=object)):
        tokens = _tokenize(text)
        for n in range(min_n, max_n + 1):
            for ng in _ngrams(tokens, n):
                counter.update([" ".join(ng)])
    df_out = _top_n_from_counter(counter, top_n)
    df_out = df_out.rename(columns={"item": "topic"})
    logger.info("Most common topics computed (%d entries)", len(df_out))
    return df_out


# --------------------
# Orchestration and saving
# --------------------


def run_all(
    input_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    top_n_keywords: int = 100,
    top_n_authors: int = 100,
    top_n_topics: int = 100,
) -> int:
    input_path = input_path or DEFAULT_CLEANED
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = _load_dataframe(input_path)

        keywords = compute_top_keywords(df, top_n=top_n_keywords)
        keywords_path = out_dir / "top_keywords.csv"
        keywords.to_csv(keywords_path, index=False)
        logger.info("Saved top keywords to %s", keywords_path)

        authors = compute_most_common_authors(df, top_n=top_n_authors)
        authors_path = out_dir / "most_common_authors.csv"
        authors.to_csv(authors_path, index=False)
        logger.info("Saved most common authors to %s", authors_path)

        pub_by_year = publication_counts_by_year(df)
        pub_path = out_dir / "publication_counts_by_year.csv"
        pub_by_year.to_csv(pub_path, index=False)
        logger.info("Saved publication counts by year to %s", pub_path)

        cats = category_frequencies(df)
        cats_path = out_dir / "category_frequencies.csv"
        cats.to_csv(cats_path, index=False)
        logger.info("Saved category frequencies to %s", cats_path)

        avg_len = average_abstract_length(df)
        avg_path = out_dir / "average_abstract_length.csv"
        avg_len.to_csv(avg_path, index=False)
        logger.info("Saved average abstract length to %s", avg_path)

        topics = most_common_topics(df, top_n=top_n_topics)
        topics_path = out_dir / "most_common_topics.csv"
        topics.to_csv(topics_path, index=False)
        logger.info("Saved most common topics to %s", topics_path)

        return 0
    except FileNotFoundError as exc:
        logger.exception("Input file not found: %s", exc)
        return 2
    except pd.errors.EmptyDataError as exc:
        logger.exception("Input CSV is empty: %s", exc)
        return 3
    except Exception as exc:
        logger.exception("Unexpected error during analysis: %s", exc)
        return 4


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("analyze-papers")
    parser.add_argument("--in", dest="input", type=str, default=None, help="Cleaned CSV path")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default=None, help="Output directory for CSV results")
    parser.add_argument("--top-n", dest="top_n", type=int, default=100, help="Number of top items to compute for lists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    raise SystemExit(run_all(input_path=Path(args.input) if args.input else None, out_dir=Path(args.out_dir) if args.out_dir else None, top_n_keywords=args.top_n, top_n_authors=args.top_n, top_n_topics=args.top_n))
