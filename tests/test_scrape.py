from paper_intel.scrape import _parse_feed
from pathlib import Path


def test_parse_feed_fixture():
    path = Path('tests') / 'fixtures' / 'arxiv_sample.xml'
    xml = path.read_text(encoding='utf-8')
    papers = _parse_feed(xml)
    assert len(papers) == 2
    p1 = papers[0]
    assert 'Test Paper One' in p1.title
    assert 'Alice Smith' in p1.authors
    assert p1.published_date is not None
    assert isinstance(p1.categories, list)

