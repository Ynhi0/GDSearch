import pandas as pd
from src.utils.metric_normalization import to_percent, to_percent_series


def test_to_percent_basic():
    cases = [
        (0.78, 78.0),
        (0.928, 92.8),
        (78.0, 78.0),
        ('92.8%', 92.8),
        ('0.928', 92.8),
        (9280, 92.8),  # should be divided down
    ]

    for inp, expected in cases:
        out = to_percent(inp)
        assert abs(out - expected) < 1e-6, f"to_percent({inp!r}) == {out}, expected {expected}"


def test_to_percent_series():
    s = pd.Series([0.78, '92.8%', 9280, '0.928', 78.0])
    out = to_percent_series(s)
    expected = [78.0, 92.8, 92.8, 92.8, 78.0]
    assert list(out.round(6)) == [round(e, 6) for e in expected]


def test_to_percent_nan_and_strings():
    s = pd.Series([float('nan'), 'nan', None, '150%'])
    out = to_percent_series(s)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])
    # 150% should be coerced to 150.0 and then clamped by visualization functions as needed
    assert out.iloc[3] == 150.0
