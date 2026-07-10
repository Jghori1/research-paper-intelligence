from collections import Counter
from paper_intel import keywords


def test_tokenize_and_filter(monkeypatch):
    texts = ["This is a Test, with punctuation!", "Another test-case: numbers 123"]
    # patch default stopwords to avoid NLTK
    monkeypatch.setattr(keywords, 'get_default_stopwords', lambda: set(['this','is','a','with','and']))
    toks = keywords.tokenize_text(texts[0])
    toks_filtered = keywords.filter_tokens(toks, stopword_set=keywords.get_default_stopwords(), min_length=3)
    assert all(len(t) >= 3 for t in toks_filtered)
    counter = keywords.count_keywords_from_texts(texts, stopword_set=keywords.get_default_stopwords(), min_length=3)
    top = keywords.top_keywords_from_texts(texts, top_n=5, stopword_set=keywords.get_default_stopwords(), min_length=3)
    assert isinstance(counter, Counter)
    assert isinstance(top, list)

