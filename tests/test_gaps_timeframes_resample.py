from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import ohlcv_guard as og
from ohlcv_guard import InvalidBarsError, InvalidRangeError, Timeframe, UnknownTimeframeError, find_gaps, resample

JAN1 = pd.Timestamp("2024-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def test_the_installed_version_is_the_package_version():
    from importlib.metadata import version

    assert version("ohlcv-guard") == og.__version__


# ---- timeframes ----

@pytest.mark.parametrize(("value", "ms"), [
    ("1m", 60_000), ("4h", 4 * 3_600_000), ("1d", 86_400_000), ("30s", 30_000), (" 15m ", 900_000), ("90m", 5_400_000),
    (60_000, 60_000), (timedelta(minutes=5), 300_000), (pd.Timedelta(hours=2), 7_200_000), (Timeframe(1_000), 1_000),
])
def test_timeframes_parse(value, ms):
    assert og.timeframe_ms(value) == ms


@pytest.mark.parametrize("value", ["2w", "1M", "0m", "m", "-1h", "1.5h", "", 0, -60_000, True, 1.5, None,
                                   timedelta(microseconds=1500)])
def test_bad_timeframes_are_refused(value):
    with pytest.raises(UnknownTimeframeError):
        og.timeframe_ms(value)


def test_timeframe_labels():
    assert [Timeframe(ms).label for ms in (60_000, 5_400_000, 86_400_000, 1_500)] == ["1m", "90m", "1d", "1500ms"]
    assert str(Timeframe.parse("4h")) == "4h" and Timeframe.parse("1h").timedelta == HOUR


def test_utc_range():
    assert og.utc_range("2024-01-01", "2024-01-02") == (JAN1, JAN1 + 24 * HOUR)
    with pytest.raises(InvalidRangeError, match="must be before end"):
        og.utc_range("2024-01-02", "2024-01-01")


def test_empty_bars_schema():
    empty = og.empty_bars()
    assert list(empty.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(empty["ts"].dtype) == og.TS_DTYPE
    assert "volume" not in og.empty_bars(with_volume=False).columns


# ---- gaps (ported) ----

def test_contiguous_bars_have_no_gaps(make_bars):
    bars = make_bars(periods=24)
    assert find_gaps(bars, "1h") == []
    assert find_gaps(bars, "1h", JAN1, JAN1 + 24 * HOUR) == []


def test_internal_gaps_are_reported_exactly(make_bars):
    bars = make_bars(periods=24).drop(index=[5, 6, 7, 15]).reset_index(drop=True)
    gaps = find_gaps(bars, "1h")
    assert [(g.start, g.end, g.missing) for g in gaps] == [
        (JAN1 + 5 * HOUR, JAN1 + 8 * HOUR, 3),
        (JAN1 + 15 * HOUR, JAN1 + 16 * HOUR, 1),
    ]


def test_leading_and_trailing_gaps_within_requested_range(make_bars):
    bars = make_bars(start="2024-01-01 02:00", periods=20)  # 02:00 .. 21:00
    gaps = find_gaps(bars, "1h", JAN1, JAN1 + 24 * HOUR)
    assert [(g.start, g.end, g.missing) for g in gaps] == [
        (JAN1, JAN1 + 2 * HOUR, 2),
        (JAN1 + 22 * HOUR, JAN1 + 24 * HOUR, 2),
    ]


def test_empty_bars_are_one_gap_covering_the_range(make_bars):
    empty = make_bars(periods=1).iloc[0:0]
    gaps = find_gaps(empty, "1h", JAN1, JAN1 + 6 * HOUR)
    assert [(g.start, g.end, g.missing) for g in gaps] == [(JAN1, JAN1 + 6 * HOUR, 6)]


def test_gap_detection_uses_the_timeframe(make_bars):
    bars = make_bars(periods=60, timeframe="1m").drop(index=[10, 11]).reset_index(drop=True)
    gaps = find_gaps(bars, "1m")
    assert len(gaps) == 1 and gaps[0].missing == 2
    assert gaps[0].start == JAN1 + pd.Timedelta(minutes=10)


def test_off_grid_range_bounds_are_rounded_up_to_the_grid(make_bars):
    bars = make_bars(start="2024-01-01 01:00", periods=5)
    assert find_gaps(bars, "1h", JAN1 + pd.Timedelta(minutes=30), JAN1 + 6 * HOUR) == []


# ---- gaps (new) ----

def test_unsorted_or_duplicated_bars_are_refused(make_bars):
    bars = make_bars(periods=6).drop(index=[2]).reset_index(drop=True)
    shuffled = bars.iloc[[4, 0, 1, 2, 3]]
    with pytest.raises(InvalidBarsError, match="sorted"):
        find_gaps(shuffled, "1h")
    with pytest.raises(InvalidBarsError, match="sorted"):
        find_gaps(pd.concat([bars, bars.iloc[[0]]]).sort_values("ts"), "1h")
    find_gaps(shuffled, "1h", assume_normalized=True)  # not checked: the caller's responsibility


def test_a_single_bar_and_bounds_off_the_grid_on_both_ends(make_bars):
    bar = make_bars(start="2024-01-01 03:00", periods=1)
    gaps = find_gaps(bar, "1h", JAN1 + pd.Timedelta(minutes=1), JAN1 + 5 * HOUR + pd.Timedelta(minutes=1))
    assert [(g.start, g.end, g.missing) for g in gaps] == [(JAN1 + HOUR, JAN1 + 3 * HOUR, 2),
                                                           (JAN1 + 4 * HOUR, JAN1 + 6 * HOUR, 2)]


def test_ten_million_missing_minutes_are_one_gap(make_bars):
    bars = pd.concat([make_bars(periods=1, timeframe="1m"),
                      make_bars(start=str(JAN1 + pd.Timedelta(minutes=10_000_001))[:19], periods=1, timeframe="1m")],
                     ignore_index=True)
    [gap] = find_gaps(bars, "1m")
    assert gap.missing == 10_000_000


def test_a_custom_time_column(make_bars):
    bars = make_bars(periods=5).drop(index=[2]).rename(columns={"ts": "time"})
    assert find_gaps(bars, "1h", ts="time")[0].missing == 1


# ---- resample ----

def test_resample_builds_whole_and_partial_bins(ohlc_bars):
    hourly = ohlc_bars([(10, 11), (11, 13), (13, 9), (9, 10), (10, 12), (12, 12)])  # 00:00..05:00
    hourly = hourly.drop(index=[5]).reset_index(drop=True)  # 05:00 missing: the second 4h bin is partial
    out = resample(hourly, "1h", "4h")
    assert out.to_dict("records") == [
        {"ts": JAN1, "open": 10.0, "high": 14.0, "low": 8.0, "close": 10.0, "volume": 40.0, "complete": True},
        {"ts": JAN1 + 4 * HOUR, "open": 10.0, "high": 13.0, "low": 9.0, "close": 12.0, "volume": 10.0,
         "complete": False},
    ]


def test_resample_keeps_extra_columns_and_their_dtype(make_bars):
    bars = make_bars(periods=8).assign(trades=np.arange(8, dtype="int64"), taker_buy_volume=1.0)
    out = resample(bars, "1h", "4h", keep=("trades", "taker_buy_volume"), mark_partial=False)
    assert out["trades"].tolist() == [6, 22] and out["trades"].dtype == "int64"
    assert out["taker_buy_volume"].tolist() == [4.0, 4.0] and "complete" not in out


def test_resample_to_utc_days_and_without_volume(make_bars):
    bars = make_bars(start="2024-01-01 22:00", periods=4).drop(columns="volume")
    out = resample(bars, "1h", "1d", volume=None)
    assert out["ts"].tolist() == [JAN1, JAN1 + 24 * HOUR] and out["complete"].tolist() == [False, False]


@pytest.mark.parametrize(("source", "target"), [("1h", "90m"), ("4h", "1h"), ("1h", "5m")])
def test_resample_needs_a_whole_multiple(make_bars, source, target):
    with pytest.raises(InvalidBarsError, match="multiple"):
        resample(make_bars(periods=4), source, target)


def test_resample_refuses_unsorted_bars_and_unknown_columns(make_bars):
    bars = make_bars(periods=4)
    with pytest.raises(InvalidBarsError, match="sorted"):
        resample(bars.iloc[::-1], "1h", "4h")
    with pytest.raises(InvalidBarsError, match="keep"):
        resample(bars, "1h", "4h", keep=("trades",))


def test_resample_of_nothing_is_nothing(make_bars):
    assert resample(make_bars(periods=1).iloc[0:0], "1h", "4h").empty


def test_resample_keeps_total_volume(make_bars):
    bars = make_bars(periods=50, timeframe="5m")
    assert resample(bars, "5m", "1h")["volume"].sum() == pytest.approx(bars["volume"].sum())
