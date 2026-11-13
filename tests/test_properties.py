"""Property tests: normalize matches the original implementation on everything that implementation accepted, is
idempotent, and gaps plus bars cover a range exactly."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import ohlcv_guard as og
from legacy import legacy_normalize

TF = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
BASE_MS = 1_704_067_200_000  # 2024-01-01


@st.composite
def frames(draw, valid=True):
    tf = draw(st.sampled_from(list(TF)))
    n = draw(st.integers(0, 40))
    slots = draw(st.lists(st.integers(0, 60), min_size=n, max_size=n))
    price = st.floats(1.0, 1e6, allow_nan=False)
    opens = draw(st.lists(price, min_size=n, max_size=n))
    closes = draw(st.lists(price, min_size=n, max_size=n))
    up = draw(st.lists(st.floats(0, 100), min_size=n, max_size=n))
    down = draw(st.lists(st.floats(0, 0.99), min_size=n, max_size=n))
    vol = draw(st.lists(st.floats(0, 1e9), min_size=n, max_size=n))
    df = pd.DataFrame({"ts": [BASE_MS + s * TF[tf] for s in slots], "open": opens, "close": closes, "volume": vol})
    df["high"] = np.maximum(df["open"], df["close"]) + np.asarray(up, dtype=float)
    df["low"] = np.minimum(df["open"], df["close"]) * (1 - np.asarray(down, dtype=float))
    if draw(st.booleans()):
        df["trades"] = draw(st.lists(st.integers(0, 10**6), min_size=n, max_size=n))
    if draw(st.booleans()) and n:  # exact duplicate rows
        df = pd.concat([df, df.sample(n=min(3, n), replace=True, random_state=draw(st.integers(0, 99)))],
                       ignore_index=True)
    if not valid and n:
        row = draw(st.integers(0, len(df) - 1))
        col, value = draw(st.sampled_from([("open", np.inf), ("high", np.nan), ("low", -1.0), ("volume", -1.0),
                                           ("volume", np.nan), ("close", 0.0), ("ts", None)]))
        if col == "ts":
            df.loc[row, "ts"] = df.loc[row, "ts"] + 1  # off the grid
        else:
            df[col] = df[col].astype("float64")
            df.loc[row, col] = value
    return df, tf


SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(frames())
def test_matches_the_original_on_what_it_accepted(case):
    df, tf = case
    try:
        expected, dropped = legacy_normalize(df, tf)
    except ValueError:
        with pytest.raises(ValueError):
            og.normalize(df, tf)
        return
    got = og.normalize(df, tf)
    pd.testing.assert_frame_equal(got.bars, expected)
    assert got.duplicates_dropped == dropped


@SETTINGS
@given(frames(valid=False))
def test_whatever_the_original_refused_is_refused(case):
    df, tf = case
    try:
        legacy_normalize(df, tf)
    except ValueError:
        with pytest.raises(og.InvalidBarsError):
            og.normalize(df, tf)


@SETTINGS
@given(frames())
def test_normalize_is_idempotent(case):
    df, tf = case
    try:
        once = og.normalize(df, tf).bars
    except og.InvalidBarsError:
        return
    twice = og.normalize(once, tf)
    pd.testing.assert_frame_equal(twice.bars, once)
    assert twice.duplicates_dropped == 0


@SETTINGS
@given(frames(), st.integers(-5, 5), st.integers(55, 70))
def test_gaps_and_bars_cover_the_range_exactly(case, lead, tail):
    df, tf = case
    try:
        bars = og.normalize(df, tf).bars
    except og.InvalidBarsError:
        return
    step = pd.Timedelta(milliseconds=TF[tf])
    start = pd.Timestamp(BASE_MS + lead * TF[tf], unit="ms", tz="UTC")
    end = pd.Timestamp(BASE_MS + tail * TF[tf], unit="ms", tz="UTC")
    inside = bars[(bars["ts"] >= start) & (bars["ts"] < end)]
    gaps = og.find_gaps(bars, tf, start, end)
    assert len(inside) + sum(g.missing for g in gaps) == (end - start) // step
    for g in gaps:
        assert not ((bars["ts"] >= g.start) & (bars["ts"] < g.end)).any()
