from __future__ import annotations

import logging
import time
from dataclasses import asdict
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


def _build_query_url(search_query: str, start: int = 0, max_results: int = 100) -> str:
    """Construct the arXiv API query URL.

    Args:
        search_query: arXiv search expression (e.g. "cat:cs.CL" or "all:machine learning").
        start: Result offset.
        max_results: Number of results to retrieve.

    Returns:
        Fully formed URL for the arXiv API.
    """
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
    }
    # Build query string manually to avoid pulling in extra deps
    q = "&".join(f"{k}={requests.utils.requote_uri(v)}" for k, v in params.items())
    return f"{ARXIV_API_URL}?{q}"


def _create_session(retries: int = 5, backoff_factor: float = 1.0, status_forcelist: Optional[List[int]] = None) -> requests.Session:
    """Create a requests.Session configured with retries.

    Args:
        retries: Total number of retry attempts.
        backoff_factor: Backoff multiplier between attempts.
        status_forcelist: HTTP status codes that should trigger a retry.

    Returns:
        Configured requests.Session.
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
    return session


def _fetch_xml(session: requests.Session, url: str, timeout: float = 10.0) -> str:
    """Fetch XML content from the given URL using the provided session.

    Args:
        session: Configured requests.Session.
        url: URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Response text (XML).

    Raises:
        NetworkError: When the request fails after retries.
    """
    logger.debug("Fetching URL: %s", url)
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.error("Network error fetching %s: %s", url, exc)
        raise NetworkError(f"Failed to fetch {url}: {exc}") from exc


def _parse_entry(elem: ET.Element) -> Paper:
    """Parse a single Atom <entry> element into a Paper.

    Args:
        elem: XML Element for the entry.

    Returns:
        Paper instance.

    Raises:
        ParseError: If required fields cannot be parsed.
    """
    try:
        def _text(tag: str, ns_key: str = "atom") -> Optional[str]:
            node = elem.find(f"{ns_key}:{tag}", NS)
            return node.text.strip() if node is not None and node.text else None

        title = _text("title") or _text("id")
        summary = _text("summary")
        published = _text("published")

        # Authors
        authors = []
        for a in elem.findall("atom:author", NS):
            name = a.find("atom:name", NS)
            if name is not None and name.text:
                authors.append(name.text.strip())

        # Categories
        categories = []
        for c in elem.findall("atom:category", NS):
            term = c.attrib.get("term")
            if term:
                categories.append(term)

        # Link: prefer the alternate link or the id
        link = None
        for l in elem.findall("atom:link", NS):
            if l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href")
                break
        if not link:
            # fallback to <id>
            link = _text("id")

        if not title:
            raise ParseError("Missing title in entry")

        return Paper(
            title=title,
            authors=authors,
            abstract=summary,
            doi=None,
            published_date=published and _parse_date_safe(published),
            source=link or "",
        )
    except ParseError:
        raise
    except Exception as exc:
        logger.exception("Failed to parse entry: %s", exc)
        raise ParseError(f"Failed to parse entry: {exc}") from exc


def _parse_date_safe(value: str) -> Optional[str]:
    """Return ISO date string or None if parsing fails.

    arXiv published dates are ISO-8601; we keep as string for CSV friendliness.
    """
    try:
        # Basic validation — ensure it's non-empty
        return value.strip()
    except Exception:
        return None


def _parse_feed(xml_text: str) -> List[Paper]:
    """Parse Atom feed XML and return a list of Paper objects.

    Args:
        xml_text: Raw Atom XML string.

    Returns:
        List of Paper instances.
    """
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", NS):
        try:
            entries.append(_parse_entry(entry))
        except ParseError as exc:
            logger.warning("Skipping entry due to parse error: %s", exc)
    return entries


def scrape_arxiv(search_query: str, max_results: int = 100, batch_size: int = 100) -> List[Paper]:
    """Scrape arXiv for papers matching the given query.

    The function pages through results in batches and returns a list of Paper objects.

    Args:
        search_query: arXiv search expression (see arXiv API docs).
        max_results: Maximum number of papers to retrieve in total.
        batch_size: Number of results to request per API call (max 2000 typically; keep small).

    Returns:
        List of Paper instances.
    """
    session = _create_session()
    papers: List[Paper] = []
    fetched = 0

    while fetched < max_results:
        to_fetch = min(batch_size, max_results - fetched)
        url = _build_query_url(search_query, start=fetched, max_results=to_fetch)
        xml = _fetch_xml(session, url)
        batch = _parse_feed(xml)
        if not batch:
            logger.info("No more entries returned by arXiv API; stopping at %d results", fetched)
            break
        papers.extend(batch)
        fetched += len(batch)
        logger.info("Fetched %d papers (total=%d)", len(batch), fetched)
        # Be polite to the API
        time.sleep(0.34)
    return papers


def write_papers_csv(papers: Iterable[Paper], path: Path) -> None:
    """Write papers to a CSV file.

    Args:
        papers: Iterable of Paper objects.
        path: Destination CSV path.
    """
    rows: List[Dict[str, object]] = []
    for p in papers:
        d = p.to_dict()
        # Ensure categories field exists (arXiv parsing currently uses Paper.source for URL)
        d.setdefault("categories", None)
        rows.append(d)

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %d papers to %s", len(rows), path)


def main(search_query: str = "cat:cs.CL", max_results: int = 100, out_path: Optional[Path] = None) -> int:
    """High-level entry point to perform a scrape and persist results.

    Args:
        search_query: arXiv API search expression.
        max_results: Maximum number of entries to retrieve.
        out_path: Output CSV path. Defaults to data/raw/papers.csv in repo root.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    out_path = out_path or (Path.cwd() / "data" / "raw" / "papers.csv")
    try:
        papers = scrape_arxiv(search_query, max_results=max_results)
        write_papers_csv(papers, out_path)
        return 0
    except NetworkError as exc:
        logger.exception("Network failure during scrape: %s", exc)
        return 2
    except ParseError as exc:
        logger.exception("Parsing failure during scrape: %s", exc)
        return 3
    except Exception as exc:
        logger.exception("Unexpected error during scrape: %s", exc)
        return 4


if __name__ == "__main__":
    # Minimal CLI for ad-hoc execution
    import argparse

    parser = argparse.ArgumentParser("arxiv-scrape")
    parser.add_argument("query", nargs="?", default="cat:cs.CL", help="arXiv search query")
    parser.add_argument("--max", type=int, default=100, help="Maximum results to fetch")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    raise SystemExit(main(args.query, args.max, Path(args.out) if args.out else None))
