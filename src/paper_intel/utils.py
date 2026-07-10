from __future__ import annotations

"""Common utility helpers for the research-paper-intelligence project.

This module collects small, well-tested utilities that are useful across the
pipeline: CSV I/O, directory management, logger configuration, text
normalization, and a few convenience helpers.

Design goals:
- Keep functions small, independent, and easy to unit test.
- Avoid side effects on import (logger configuration is explicit).
- Use Path objects for filesystem APIs.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]


# --------------------
# Filesystem helpers
# --------------------


def ensure_dir(path: PathLike, parents: bool = True, exist_ok: bool = True) -> Path:
    """Ensure that a directory exists.

    If the provided path is a file path (looks like it has a suffix), the parent
    directory will be created instead; otherwise the path itself is created.

    Args:
        path: Directory path or a file path whose parent directory should be created.
        parents: Passed to Path.mkdir
        exist_ok: Passed to Path.mkdir

    Returns:
        The directory Path that was ensured to exist.
    """
    p = Path(path)
    # Heuristic: treat values with a suffix as file paths => ensure parent
    target = p if p.suffix == "" else p.parent
    target.mkdir(parents=parents, exist_ok=exist_ok)
    return target


def read_csv(path: PathLike, **kwargs) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame with sensible defaults.

    This wrapper centralizes CSV reading behavior and logging. Additional
    keyword arguments are forwarded to pandas.read_csv.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional kwargs forwarded to pandas.read_csv.

    Returns:
        pandas.DataFrame

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the CSV is empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    logging.getLogger(__name__).info("Reading CSV from %s", p)
    df = pd.read_csv(p, dtype=kwargs.pop("dtype", None), **kwargs)
    logging.getLogger(__name__).info("Loaded DataFrame with %d rows and %d columns", len(df), len(df.columns))
    return df


def write_csv(df: pd.DataFrame, path: PathLike, index: bool = False, **kwargs) -> None:
    """Write a DataFrame to CSV, creating directories as needed.

    Args:
        df: DataFrame to write.
        path: Destination file path.
        index: Whether to write the index to CSV.
        **kwargs: Additional kwargs forwarded to DataFrame.to_csv.
    """
    p = Path(path)
    ensure_dir(p)
    logging.getLogger(__name__).info("Writing DataFrame with %d rows to %s", len(df), p)
    df.to_csv(p, index=index, **kwargs)


# --------------------
# Logging helpers
# --------------------


def configure_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt: Optional[str] = None,
    log_file: Optional[PathLike] = None,
) -> logging.Logger:
    """Configure and return a logger.

    This function configures the root logger handlers (StreamHandler and
    optionally a FileHandler) with a simple formatter. It is safe to call
    multiple times; duplicate handlers are avoided.

    Args:
        name: If provided, returns a child logger for this name; otherwise the root logger.
        level: Logging level.
        fmt: Log message format string.
        datefmt: Optional date format string.
        log_file: Optional path to a file where logs should be written.

    Returns:
        Configured logger instance.
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Ensure we don't add duplicate stream handlers
    stream_exists = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not stream_exists:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if log_file:
        fh_exists = any(getattr(h, "baseFilename", None) == str(Path(log_file).resolve()) for h in root.handlers if isinstance(h, logging.FileHandler))
        if not fh_exists:
            fh = logging.FileHandler(Path(log_file))
            fh.setFormatter(formatter)
            root.addHandler(fh)

    return logging.getLogger(name) if name else root


# --------------------
# Text utilities
# --------------------


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: Optional[str], *, collapse_whitespace: bool = True) -> str:
    """Normalize a text string for downstream processing.

    Steps performed:
    - None -> empty string
    - Convert to str
    - Unicode normalize (NFC)
    - Replace control characters with a single space
    - Optionally collapse whitespace to a single space
    - Trim leading/trailing spaces

    Args:
        text: Input text (may be None).
        collapse_whitespace: Whether to collapse multiple whitespace characters into one.

    Returns:
        Normalized text string.
    """
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFC", s)
    s = _CONTROL_CHAR_RE.sub(" ", s)
    if collapse_whitespace:
        s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


def split_lines(text: Optional[str]) -> List[str]:
    """Split a normalized text into non-empty lines.

    Args:
        text: Input text.

    Returns:
        List of non-empty stripped lines.
    """
    s = normalize_text(text)
    return [line.strip() for line in s.splitlines() if line.strip()]


# --------------------
# Misc helpers
# --------------------


def chunked_iterable(iterable: Iterable, size: int) -> Iterator[Tuple]:
    """Yield successive tuples of length <= size from an iterable.

    Example:
        list(chunked_iterable(range(7), 3)) -> [(0,1,2),(3,4,5),(6,)]
    """
    it = iter(iterable)
    while True:
        chunk = tuple([x for _, x in zip(range(size), it)])
        if not chunk:
            break
        yield chunk


def safe_get_column(df: pd.DataFrame, column: str, default=None):
    """Return a Series for a column if present, otherwise a Series of default values.

    The returned object can be iterated over like a pandas Series.
    """
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df))


__all__ = [
    "ensure_dir",
    "read_csv",
    "write_csv",
    "configure_logger",
    "normalize_text",
    "split_lines",
    "chunked_iterable",
    "safe_get_column",
]
