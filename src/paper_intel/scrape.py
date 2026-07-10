from __future__ import annotations

"""arXiv scraping utilities.

This module provides a clear, well-factored implementation to query the arXiv
Atom API, convert Atom <entry> elements into the project's Paper domain model,
and persist results to a CSV file.

Design goals:
- Keep network, parsing, and persistence concerns separated and testable.
- Use retries and timeouts for robustness against transient network failures.
- Provide clear type annotations and concise, actionable logging messages.

Functionality is intentionally unchanged from the previous implementation: it
pages through arXiv results, extracts title/authors/abstract/published date/
categories/link, and writes a CSV at data/raw/papers.csv by default.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import NetworkError, ParseError
from .models.paper import Paper

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS}
DEFAULT_OUT = Path("data") / "raw" / "papers.csv"
_DEFAULT_BATCH_SLEEP = 0.34


# --------------------
# Network utilities
# --------------------


def _build_query_params(search_query: str, start: int, max_results: int) -> Dict[str, str]:
    """Return query parameters for the arXiv API call.

    Args:
        search_query: arXiv search expression.
        start: Offset for results.
        max_results: Number of results to request.

    Returns:
        Mapping of query parameter names to values.
    """
    return {"search_query": search_query, "start": str(start), "max_results": str(max_results)}


def _build_query_url(search_query: str, start: int = 0, max_results: int = 100) -> str:
    """Construct a fully encoded arXiv API URL.

    This helper keeps the URL construction small and dependency-free.
    """
    params = _build_query_params(search_query, start, max_results)
    # requests.utils.requote_uri ensures characters are safe for URLs
    q = "&".join(f"{k}={requests.utils.requote_uri(v)}" for k, v in params.items())
    url = f"{ARXIV_API_URL}?{q}"
    logger.debug("Built arXiv URL: %s", url)
    return url


def _create_session(
    retries: int = 5, backoff_factor: float = 0.5, status_forcelist: Optional[List[int]] = None
) -> requests.Session:
    """Create an HTTP session with a retry policy.

    Args:
        retries: Total retry attempts for transient failures.
        backoff_factor: Controls sleep time between retries.
        status_forcelist: HTTP status codes that should trigger a retry.

    Returns:
        Configured requests.Session instance.
    """
    status_forcelist = status_forcelist or [429, 500, 502, 503, 504]
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    logger.debug("HTTP session created with retries=%s backoff=%s", retries, backoff_factor)
    return session


def _fetch_xml(session: requests.Session, url: str, timeout: float = 10.0) -> str:
    """Fetch XML from the given URL.

    Raises:
        NetworkError: If the request fails after retries.
    """
    logger.debug("Requesting XML from %s", url)
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.error("Error fetching %s: %s", url, exc)
        raise NetworkError(f"Failed to fetch {url}: {exc}") from exc


# --------------------
# Parsing utilities
# --------------------


def _get_text(elem: ET.Element, tag: str, ns: str = "atom") -> Optional[str]:
    """Return the trimmed text content of a child tag, or None.

    Args:
        elem: Parent XML element.
        tag: Local tag name (without namespace prefix).
        ns: Namespace key defined in NS.
    """
    node = elem.find(f"{ns}:{tag}", NS)
    return node.text.strip() if node is not None and node.text else None


def _parse_authors(entry: ET.Element) -> List[str]:
    """Extract author names from an <entry> element."""
    names: List[str] = []
    for author in entry.findall("atom:author", NS):
        name = author.find("atom:name", NS)
        if name is not None and name.text:
            names.append(name.text.strip())
    return names


def _parse_categories(entry: ET.Element) -> List[str]:
    """Extract category terms from an <entry> element."""
    cats: List[str] = []
    for cat in entry.findall("atom:category", NS):
        term = cat.attrib.get("term")
        if term:
            cats.append(term)
    return cats


def _parse_link(entry: ET.Element) -> Optional[str]:
    """Find the preferred link for the paper (alternate) or fall back to <id>.

    arXiv typically exposes a link rel="alternate"; use it when present.
    """
    for link_el in entry.findall("atom:link", NS):
        if link_el.attrib.get("rel") == "alternate":
            href = link_el.attrib.get("href")
            if href:
                return href
    return _get_text(entry, "id")


def _parse_entry(entry: ET.Element) -> Paper:
    """Convert an Atom <entry> into a Paper domain object.

    Raises:
        ParseError: When required fields (title) are missing or other parsing fails.
    """
    try:
        title = _get_text(entry, "title") or _get_text(entry, "id")
        if not title:
            raise ParseError("missing title")

        authors = _parse_authors(entry)
        abstract = _get_text(entry, "summary")
        published = _get_text(entry, "published")
        categories = _parse_categories(entry)
        link = _parse_link(entry) or ""

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            doi=None,
            published_date=published,
            source=link,
            categories=categories,
        )
    except ParseError:
        raise
    except Exception as exc:
        logger.exception("Unexpected parse error for entry: %s", exc)
        raise ParseError(f"failed to parse entry: {exc}") from exc


def _parse_feed(xml_text: str) -> List[Paper]:
    """Parse the Atom feed text and return Paper instances for each entry.

    Non-fatal parse errors for individual entries are logged and skipped.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse XML: %s", exc)
        raise ParseError(f"invalid XML: {exc}") from exc

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", NS):
        try:
            papers.append(_parse_entry(entry))
        except ParseError as exc:
            logger.warning("Skipping malformed entry: %s", exc)
    return papers


# --------------------
# High-level orchestration
# --------------------


def scrape_arxiv(search_query: str, max_results: int = 100, batch_size: int = 100) -> List[Paper]:
    """Retrieve up to ``max_results`` papers matching ``search_query`` from arXiv.

    The function pages through results using batches of size ``batch_size``. It
    returns the collected Paper objects. Network and parsing errors are surfaced
    as NetworkError or ParseError respectively.
    """
    session = _create_session()
    papers: List[Paper] = []
    offset = 0

    while offset < max_results:
        limit = min(batch_size, max_results - offset)
        url = _build_query_url(search_query, start=offset, max_results=limit)
        xml = _fetch_xml(session, url)
        batch = _parse_feed(xml)
        if not batch:
            logger.info("arXiv returned no entries; stopping at offset %d", offset)
            break
        papers.extend(batch)
        offset += len(batch)
        logger.info("Fetched %d entries (offset now %d)", len(batch), offset)
        time.sleep(_DEFAULT_BATCH_SLEEP)

    return papers


def write_papers_csv(papers: Iterable[Paper], path: Path) -> None:
    """Persist an iterable of Paper objects to CSV.

    Args:
        papers: Iterable of Paper domain objects.
        path: File system path to write the CSV to.
    """
    rows = [p.to_dict() for p in papers]
    df = pd.DataFrame(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %d papers to %s", len(rows), path)


def main(search_query: str = "cat:cs.CL", max_results: int = 100, out_path: Optional[Path] = None) -> int:
    """Top-level entry point for command-line execution.

    Returns an exit code: 0 on success, non-zero on different failure classes.
    """
    out = out_path or DEFAULT_OUT
    try:
        papers = scrape_arxiv(search_query, max_results=max_results)
        write_papers_csv(papers, out)
        return 0
    except NetworkError as exc:
        logger.exception("Network error during scrape: %s", exc)
        return 2
    except ParseError as exc:
        logger.exception("Parsing error during scrape: %s", exc)
        return 3
    except Exception as exc:
        logger.exception("Unexpected error during scrape: %s", exc)
        return 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser("arxiv-scrape")
    parser.add_argument("query", nargs="?", default="cat:cs.CL", help="arXiv search query")
    parser.add_argument("--max", type=int, default=100, help="Maximum results to fetch")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    raise SystemExit(main(args.query, args.max, Path(args.out) if args.out else None))
