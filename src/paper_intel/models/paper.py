from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Paper:
    """Domain model representing basic metadata for a research paper.

    Attributes:
        title: Paper title.
        authors: Ordered list of author names.
        abstract: Short abstract or summary if available.
        doi: Digital Object Identifier if available.
        published_date: Publication date in ISO-8601 string form (YYYY-MM-DD or full
            timestamp). Kept as a string for CSV/JSON friendliness.
        source: The original source URL or path.
        categories: List of category terms (e.g., ["cs.CL", "cs.AI"]).
    """

    title: str
    authors: List[str]
    abstract: Optional[str]
    doi: Optional[str]
    published_date: Optional[str]
    source: str
    categories: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Paper to a JSON-serializable dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "doi": self.doi,
            "published_date": self.published_date,
            "source": self.source,
            "categories": self.categories,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Paper":
        """Create a Paper from a dictionary.

        The function is permissive about missing fields and preserves the published_date
        as a string when present.
        """
        pd = data.get("published_date")
        if pd is None:
            published_date = None
        else:
            # Keep the raw ISO string — callers may normalize if needed.
            published_date = str(pd)

        return Paper(
            title=str(data.get("title", "Untitled")),
            authors=[str(a) for a in data.get("authors", [])],
            abstract=data.get("abstract"),
            doi=data.get("doi"),
            published_date=published_date,
            source=str(data.get("source", "")),
            categories=[str(c) for c in data.get("categories", [])],
        )


__all__ = ["Paper"]
