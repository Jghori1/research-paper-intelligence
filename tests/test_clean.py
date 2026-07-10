import io
import os
import pandas as pd
import logging

from paper_intel import clean


def test_remove_duplicates():
    df = pd.DataFrame({'a':[1,1,2], 'b':["x","x","y"]})
    df2 = clean.remove_duplicates(df)
    assert len(df2) == 2


def test_drop_missing_abstracts():
    df = pd.DataFrame({'title':['t1','t2','t3'], 'abstract':['a1', None, '  ']})
    df2 = clean.drop_missing_abstracts(df)
    assert len(df2) == 1


def test_apply_text_pipeline(monkeypatch):
    # avoid NLTK network calls by patching stopwords
    monkeypatch.setattr(clean, 'stopwords', type('S', (), {'words': lambda lang: ['the','and','is']}))
    raw = "  This is <b>HTML</b> text, with punctuation! And   extra   spaces.  "
    out = clean._apply_text_pipeline(raw)
    # expected: remove html, punctuation, stopwords ('the','and','is'), lowercase, single spaces
    assert 'html' in out
    assert 'punctuation' in out
    assert 'and' not in out
    assert out == out.strip()
