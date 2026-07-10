from __future__ import annotations

"""Keyword extraction utilities.

This module provides small, reusable functions to extract keyword candidates from
text for downstream analysis. The implementation is intentionally simple and
suitable for unit testing.

Primary functions:
- tokenize_text(text) -> list[str]
- filter_tokens(tokens, stopwords=None, min_length=3) -> list[str]
- count_keywords_from_texts(texts, stopwords=None, min_length=3) -> Counter[str]
- top_keywords_from_texts(texts, top_n=50, stopwords=None, min_length=3) -> list[tuple[str,int]]

Behavior and constraints (per requirements):
- Tokenization splits on whitespace after removing punctuation and normalizing unicode.
- Stop words are removed; a default NLTK English stopword set is available.
- Punctuation is removed.
- Words shorter than three characters are ignored.
- Word frequencies are counted using collections.Counter and the top N are returned.

The functions are small and pure enough to be tested in isolation.
"""

from collections import Counter
import logging
import re
import string
import unicodedata
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)


def _ensure_nltk_stopwords(language: str = "english") -> None:
    """Ensure the NLTK stopwords corpus is available, downloading if necessary."""
    try:
        stopwords.words(language)
    except LookupError:
        logger.debug("NLTK stopwords not found; downloading...")
        nltk.download("stopwords", quiet=True)


def get_default_stopwords(language: str = "english") -> Set[str]:
    """Return the default set of stopwords (NLTK).

    Args:
        language: Language for stopwords.

    Returns:
        Set of stopword strings.
    """
    _ensure_nltk_stopwords(language)
    return set(stopwords.words(language))


def normalize_text(text: str) -> str:
    """Normalize unicode and strip control characters from text.

    Args:
        text: Input text.

    Returns:
        Normalized string.
    """
    if text is None:
        return ""
    # Normalize unicode to NFC form and remove control characters
    s = unicodedata.normalize("NFC", str(text))
    # Replace non-printable/control chars with space
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)
    return s


_PUNCT_TRANSLATOR = str.maketrans({p: "" for p in string.punctuation})


def remove_punctuation(text: str) -> str:
    """Remove ASCII punctuation from text.

    Note: This removes characters in string.punctuation. Non-ASCII punctuation may
    remain but will typically be removed by the token filters below.
    """
    return text.translate(_PUNCT_TRANSLATOR)


def tokenize_text(text: str) -> List[str]:
    """Convert input text into a list of lowercase tokens.

    Steps:
    - Normalize unicode
    - Remove punctuation
    - Split on whitespace
    - Lowercase tokens

    Args:
        text: Input text.

    Returns:
        List of token strings (may be empty).
    """
    if text is None:
        return []
    s = normalize_text(text)
    s = remove_punctuation(s)
    # Collapse whitespace and split
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    tokens = [tok.lower() for tok in s.split(" ") if tok]
    return tokens


def filter_tokens(
    tokens: Iterable[str], stopword_set: Optional[Set[str]] = None, min_length: int = 3
) -> List[str]:
    """Filter tokens by removing stopwords and short tokens.

    Args:
        tokens: Iterable of token strings (assumed already lowercased).
        stopword_set: Optional set of stopwords to remove. If None, the NLTK English
            stopwords set is used.
        min_length: Minimum token length to keep (inclusive).

    Returns:
        List of filtered tokens.
    """
    if stopword_set is None:
        stopword_set = get_default_stopwords()
    filtered: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        if len(tok) < min_length:
            continue
        # keep alphabetic tokens (ignore tokens with digits/underscores)
        if not tok.isalpha():
            continue
        if tok in stopword_set:
            continue
        filtered.append(tok)
    return filtered


def count_keywords_from_texts(
    texts: Iterable[str], stopword_set: Optional[Set[str]] = None, min_length: int = 3
) -> Counter:
    """Count keyword frequencies across multiple text entries.

    Args:
        texts: Iterable of text strings.
        stopword_set: Optional stopword set.
        min_length: Minimum token length to consider.

    Returns:
        collections.Counter mapping token -> frequency.
    """
    counter: Counter = Counter()
    if stopword_set is None:
        stopword_set = get_default_stopwords()

    for text in texts:
        tokens = tokenize_text(text)
        tokens = filter_tokens(tokens, stopword_set=stopword_set, min_length=min_length)
        counter.update(tokens)
    return counter


def top_keywords_from_texts(
    texts: Iterable[str], top_n: int = 50, stopword_set: Optional[Set[str]] = None, min_length: int = 3
) -> List[Tuple[str, int]]:
    """Return the top N keywords from the provided texts.

    Args:
        texts: Iterable of text strings.
        top_n: Number of top keywords to return.
        stopword_set: Optional stopword set.
        min_length: Minimum token length to consider.

    Returns:
        List of tuples (keyword, count) sorted by count descending.
    """
    counter = count_keywords_from_texts(texts, stopword_set=stopword_set, min_length=min_length)
    return counter.most_common(top_n)


def extract_keywords_from_dataframe(
    df,
    column: str = "abstract_clean",
    top_n: int = 50,
    stopword_set: Optional[Set[str]] = None,
    min_length: int = 3,
):
    """Convenience wrapper to extract top keywords from a DataFrame column.

    Args:
        df: pandas.DataFrame-like object with the text column.
        column: Column name to extract text from.
        top_n: Number of top keywords to return.
        stopword_set: Optional stopword set.
        min_length: Minimum token length.

    Returns:
        List of (keyword, count) pairs.
    """
    texts = df.get(column, []) if hasattr(df, "get") else []
    return top_keywords_from_texts(texts, top_n=top_n, stopword_set=stopword_set, min_length=min_length)
