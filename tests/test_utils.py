import os
import pandas as pd
from paper_intel.utils import write_csv, read_csv, ensure_dir


def test_read_write_csv(tmp_path):
    df = pd.DataFrame({'a':[1,2,3], 'b':['x','y','z']})
    out = tmp_path / "subdir" / "test.csv"
    write_csv(df, out)
    assert out.exists()
    df2 = read_csv(out)
    assert len(df2) == 3


def test_ensure_dir(tmp_path):
    file_path = tmp_path / "nested" / "file.txt"
    d = ensure_dir(file_path)
    assert d.exists()
    assert d.is_dir()
