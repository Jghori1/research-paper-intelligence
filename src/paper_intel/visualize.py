from __future__ import annotations

"""Visualization utilities for research-paper-intelligence.

This module creates PNG figures using matplotlib for:
- Keyword frequency bar chart
- Publication trend by year
- Category distribution chart

Plotting logic is organized into reusable functions so it can be imported and
unit-tested (data-to-figure steps separated from file I/O).

Requirements satisfied:
- Uses matplotlib only
- Saves figures as PNG
- Descriptive titles and axis labels

Usage:
    PYTHONPATH=src python -m paper_intel.visualize

"""

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .keywords import count_keywords_from_texts
from .analyze import publication_counts_by_year, category_frequencies

logger = logging.getLogger(__name__)

DEFAULT_CLEANED = Path("data") / "processed" / "cleaned_papers.csv"
OUT_DIR = Path("graphs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------
# Data loaders
# --------------------


def _load_cleaned(path: Path = DEFAULT_CLEANED) -> pd.DataFrame:
    logger.info("Loading cleaned data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows", len(df))
    return df


# --------------------
# Plotting helpers
# --------------------


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", path)


# --------------------
# Keyword frequency
# --------------------


def compute_top_keywords(df: pd.DataFrame, text_column: str = "abstract_clean", top_n: int = 20):
    """Return top_n keywords and counts from a DataFrame column.

    This function delegates tokenization and filtering to keywords.count_keywords_from_texts
    to ensure consistent preprocessing.
    """
    texts = df.get(text_column, pd.Series(dtype=object)).astype(str).tolist()
    counter = count_keywords_from_texts(texts)
    most_common = counter.most_common(top_n)
    return most_common


def plot_keyword_frequency(
    kw_counts: Iterable[Tuple[str, int]],
    out_path: Path = OUT_DIR / "keyword_frequency.png",
    title: str = "Top Keywords by Frequency",
) -> None:
    """Plot a horizontal bar chart of keyword frequencies.

    Args:
        kw_counts: Iterable of (keyword, count) pairs, ordered from most to least common.
        out_path: Output PNG path.
        title: Plot title.
    """
    kws = [k for k, _ in kw_counts]
    counts = [c for _, c in kw_counts]

    fig, ax = plt.subplots(figsize=(8, max(4, len(kws) * 0.35)))
    y_pos = range(len(kws))[::-1]
    ax.barh(y_pos, counts, align="center", color="#4C72B0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(kws)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    _save_figure(fig, out_path)


# --------------------
# Publication trend
# --------------------


def plot_publication_trend(
    df: pd.DataFrame,
    date_column: str = "published_date",
    out_path: Path = OUT_DIR / "publication_trend.png",
    title: str = "Publication Trend by Year",
) -> None:
    """Plot a line chart of publication counts by year.

    Args:
        df: Input DataFrame.
        date_column: Column containing date-like strings.
        out_path: Output PNG path.
        title: Plot title.
    """
    years_df = publication_counts_by_year(df, date_column=date_column)
    if years_df.empty:
        logger.warning("No publication year data available to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years_df["year"], years_df["count"], marker="o", linestyle="-", color="#55A868")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Publications")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    _save_figure(fig, out_path)


# --------------------
# Category distribution
# --------------------


def plot_category_distribution(
    df: pd.DataFrame,
    categories_column: str = "categories",
    top_n: int = 10,
    out_path: Path = OUT_DIR / "category_distribution.png",
    title: str = "Top Categories",
) -> None:
    """Plot a bar chart showing the distribution of categories.

    Args:
        df: Input DataFrame.
        categories_column: Column containing categories (list literal or delimited string).
        top_n: Number of top categories to display.
        out_path: Output PNG path.
        title: Plot title.
    """
    cats_df = category_frequencies(df, categories_column=categories_column)
    if cats_df.empty:
        logger.warning("No category data available to plot")
        return
    top = cats_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    y_pos = range(len(top))[::-1]
    ax.barh(y_pos, top["count"].values, color="#C44E52")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["category"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    _save_figure(fig, out_path)


# --------------------
# Entry point
# --------------------


def generate_all_plots(
    input_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    top_k_keywords: int = 20,
) -> None:
    input_path = input_path or DEFAULT_CLEANED
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_cleaned(input_path)

    # Keywords
    try:
        kw = compute_top_keywords(df, top_n=top_k_keywords)
        plot_keyword_frequency(kw, out_path=out_dir / "keyword_frequency.png")
    except Exception as exc:
        logger.exception("Failed to generate keyword frequency plot: %s", exc)

    # Publication trend
    try:
        plot_publication_trend(df, out_path=out_dir / "publication_trend.png")
    except Exception as exc:
        logger.exception("Failed to generate publication trend plot: %s", exc)

    # Category distribution
    try:
        plot_category_distribution(df, out_path=out_dir / "category_distribution.png")
    except Exception as exc:
        logger.exception("Failed to generate category distribution plot: %s", exc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("visualize-papers")
    parser.add_argument("--in", dest="input", type=str, default=None, help="Cleaned CSV path")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default=None, help="Output directory for PNGs")
    parser.add_argument("--top-k", dest="top_k", type=int, default=20, help="Top-k keywords to display")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    generate_all_plots(input_path=Path(args.input) if args.input else None, out_dir=Path(args.out_dir) if args.out_dir else None, top_k_keywords=args.top_k)
