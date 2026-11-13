import json

import numpy as np
import pandas as pd
import pytest

import ohlcv_guard as og
from ohlcv_guard import Checks, InvalidBarsError, VolumePolicy, normalize, validate

JAN1 = pd.Timestamp("2024-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)
MINUTE = pd.Timedelta(minutes=1)


def flat_from(bars, start, price=None):
    """Bars from position `start` on frozen at one price with no volume."""
    bars = bars.copy()
    p = bars.loc[start - 1, "close"] if price is None else price
    bars.loc[start:, ["open", "high", "low", "close"]] = p
    bars.loc[start:, "volume"] = 0.0
    return bars


# ---- TAKER_BUY_VOLUME_EXCEEDS_TOTAL ----

def test_taker_buy_volume_above_the_total_is_a_warning_and_the_rows_stay(make_bars):
    bars = make_bars(periods=6).assign(taker_buy_volume=1.0)
    bars.loc[3, "taker_buy_volume"] = bars.loc[3, "volume"] * 1.7
    result = normalize(bars, "1h")
    issue = result.report.get("TAKER_BUY_VOLUME_EXCEEDS_TOTAL")
    assert issue.severity == "warning" and issue.count == 1 and issue.examples == (JAN1 + 3 * HOUR,)
    pd.testing.assert_frame_equal(result.bars, normalize(bars, "1h", checks=Checks(taker_buy_volume=False)).bars)


def test_equal_taker_and_total_volume_is_fine(make_bars):
    bars = make_bars(periods=4)
    assert normalize(bars.assign(taker_buy_volume=bars["volume"]), "1h").report.warnings == ()


def test_a_warning_can_be_escalated(make_bars):
    bars = make_bars(periods=4).assign(taker_buy_volume=1e9)
    with pytest.raises(InvalidBarsError, match="escalated") as exc:
        normalize(bars, "1h", escalate=["TAKER_BUY_VOLUME_EXCEEDS_TOTAL"])
    assert exc.value.issues[0].severity == "error"
    assert not validate(bars, "1h", escalate=["TAKER_BUY_VOLUME_EXCEEDS_TOTAL"]).ok
    with pytest.raises(ValueError, match="only warning codes"):
        normalize(bars, "1h", escalate=["OFF_GRID"])


# ---- FROZEN_RUN ----

def test_a_settled_market_publishing_flat_bars_is_a_frozen_run(make_bars):
    bars = flat_from(make_bars(periods=200, timeframe="1m"), 80)  # 120 frozen minutes
    issue = normalize(bars, "1m").report.get("FROZEN_RUN")
    assert issue.count == 120 and issue.first_ts == JAN1 + 80 * MINUTE and issue.examples == (JAN1 + 80 * MINUTE,)
    assert "1 run(s)" in issue.message and "longest 120" in issue.message


@pytest.mark.parametrize(("frozen", "found"), [(59, None), (60, 60)])
def test_frozen_run_threshold(make_bars, frozen, found):
    bars = flat_from(make_bars(periods=100, timeframe="1m"), 100 - frozen)
    issue = normalize(bars, "1m").report.get("FROZEN_RUN")
    assert (issue.count if issue else None) == found


def test_one_trading_bar_splits_frozen_runs(make_bars):
    bars = flat_from(make_bars(periods=200, timeframe="1m"), 10)
    bars.loc[100, ["high", "volume"]] = [bars.loc[100, "high"] + 1, 5.0]
    issue = normalize(bars, "1m").report.get("FROZEN_RUN")
    assert "2 run(s)" in issue.message and issue.count == 189 and len(issue.examples) == 2


def test_flat_bars_with_volume_or_unknown_volume_are_not_frozen(make_bars):
    bars = flat_from(make_bars(periods=100, timeframe="1m"), 10)
    assert normalize(bars.assign(volume=1.0), "1m").report.get("FROZEN_RUN") is None
    unknown = bars.assign(volume=np.where(bars.index >= 10, np.nan, 1.0))
    assert normalize(unknown, "1m", volume=VolumePolicy(unknown="nan")).report.get("FROZEN_RUN") is None


def test_a_series_without_volume_freezes_on_price_alone(make_bars):
    bars = flat_from(make_bars(periods=100, timeframe="1m"), 10).drop(columns="volume")
    assert normalize(bars, "1m", volume=VolumePolicy(kind="none")).report.get("FROZEN_RUN").count == 90


def test_frozen_runs_can_be_turned_off(make_bars):
    bars = flat_from(make_bars(periods=100, timeframe="1m"), 10)
    assert normalize(bars, "1m", checks=Checks(frozen_run=None)).report.warnings == ()


# ---- SPARSE_FOR_TIMEFRAME ----

def daily_bars_labelled_hourly(days=40):
    ts = pd.date_range("2015-03-02 22:00", periods=days, freq="D", tz="UTC").as_unit("ms")
    rng = np.random.default_rng(1)
    close = 1.1 + np.cumsum(rng.normal(0, 0.01, days))
    return pd.DataFrame({"ts": ts, "open": close, "high": close + 0.006, "low": close - 0.006, "close": close,
                         "volume": 150_000.0})


def test_one_bar_a_day_in_an_hourly_series_is_sparse():
    issue = normalize(daily_bars_labelled_hourly(), "1h").report.get("SPARSE_FOR_TIMEFRAME")
    assert issue.count == 40 and "median of 1 bar(s) a day where 1h implies up to 24" in issue.message
    assert "2015-03, 2015-04" in issue.message


def test_a_quiet_but_hourly_series_with_weekends_is_not_sparse(make_bars):
    bars = make_bars(periods=24 * 60)  # 60 days hourly
    weekday = bars["ts"].dt.day_of_week  # (dt.weekday is deprecated from pandas 3.1)
    weekdays_only = bars[weekday < 5].reset_index(drop=True)
    assert normalize(weekdays_only, "1h").report.get("SPARSE_FOR_TIMEFRAME") is None


def test_sparse_needs_enough_days_and_a_timeframe_under_a_day(make_bars):
    assert normalize(daily_bars_labelled_hourly(4), "1h").report.get("SPARSE_FOR_TIMEFRAME") is None
    daily = make_bars(periods=60, timeframe="1d")
    assert normalize(daily, "1d").report.get("SPARSE_FOR_TIMEFRAME") is None
    assert normalize(daily_bars_labelled_hourly(), "1h", checks=Checks(sparse_for_timeframe=False)).report.ok


# ---- PRICE_JUMP (opt-in) ----

def test_price_jumps_are_off_by_default_and_found_when_asked(ohlc_bars):
    bars = ohlc_bars([(100, 101), (101, 130), (130, 129), (129, 100)])
    assert normalize(bars, "1h").report.get("PRICE_JUMP") is None
    issue = normalize(bars, "1h", checks=Checks(price_jump=0.2)).report.get("PRICE_JUMP")
    assert issue.count == 2 and issue.examples == (JAN1 + HOUR, JAN1 + 3 * HOUR)


@pytest.mark.parametrize("kwargs", [{"frozen_run": 1}, {"sparse_ratio": 1.5}, {"sparse_min_days": 0},
                                    {"price_jump": 0}])
def test_check_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        Checks(**kwargs)


# ---- volume policy ----

def test_the_default_policy_is_strict(make_bars):
    bars = make_bars(periods=3)
    bars.loc[1, "volume"] = np.nan
    with pytest.raises(InvalidBarsError, match="volume"):
        normalize(bars, "1h")


def test_unknown_volume_as_nan(make_bars):
    bars = make_bars(periods=3)
    bars.loc[1, "volume"] = np.nan
    result = normalize(bars, "1h", volume=VolumePolicy(unknown="nan"))
    assert result.report.ok and np.isnan(result.bars.loc[1, "volume"])
    assert og.volume_known(result.bars, VolumePolicy(unknown="nan")).tolist() == [True, False, True]
    bars.loc[2, "volume"] = -1.0
    with pytest.raises(InvalidBarsError):
        normalize(bars, "1h", volume=VolumePolicy(unknown="nan"))  # negative stays an error


def test_zeros_in_declared_ranges_are_unknown_and_not_reported(make_bars):
    bars = make_bars(periods=48)
    bars["volume"] = 0.0  # legacy storage: unknown volume written as 0
    legacy = VolumePolicy(kind="tick", unknown="zero_in_ranges", unknown_ranges=[(JAN1, JAN1 + 24 * HOUR)],
                          zero="warn_if_price_moves")
    report = normalize(bars, "1h", volume=legacy).report
    assert report.get("VOLUME_ZERO_WITH_PRICE_MOVE").count == 24  # only the day outside the range
    assert report.get("VOLUME_ZERO_WITH_PRICE_MOVE").first_ts == JAN1 + 24 * HOUR
    known = og.volume_known(bars, legacy)
    assert not known[:24].any() and known[24:].all()


def test_zero_volume_with_a_moving_price_is_reported_only_when_asked(make_bars):
    bars = make_bars(periods=4).assign(volume=0.0)
    assert normalize(bars, "1h").report.warnings == ()
    assert normalize(bars, "1h", volume=VolumePolicy(zero="warn_if_price_moves")).report.counts() == {
        "VOLUME_ZERO_WITH_PRICE_MOVE": 4}


def test_a_series_without_volume(make_bars):
    index = make_bars(periods=5).drop(columns="volume")
    result = normalize(index, "1h", volume=VolumePolicy(kind="none"))
    assert "volume" not in result.bars.columns and result.report.meta["volume_kind"] == "none"
    mark = make_bars(periods=5).assign(volume=0.0)
    kept = normalize(mark, "1h", volume=VolumePolicy(kind="none")).bars
    assert "volume" not in kept.columns  # not applicable: not carried
    assert not og.volume_known(kept, VolumePolicy(kind="none")).any()


@pytest.mark.parametrize("kwargs", [
    {"kind": "shares"}, {"unknown": "maybe"}, {"zero": "deny"},
    {"unknown": "zero_in_ranges"},  # no ranges
    {"unknown_ranges": [("2024-01-01", "2024-02-01")]},  # ranges without zero_in_ranges
    {"unknown": "zero_in_ranges", "unknown_ranges": [("2024-02-01", "2024-01-01")]},
    {"kind": "none", "unknown": "nan"},
])
def test_volume_policies_are_validated(kwargs):
    with pytest.raises(ValueError):
        VolumePolicy(**kwargs)


# ---- report ----

def test_report_serialization_is_stable(make_bars):
    bars = flat_from(make_bars(periods=100, timeframe="1m"), 10).assign(taker_buy_volume=1e9)
    report = normalize(bars, "1m").report
    data = json.loads(report.to_json())
    assert [w["code"] for w in data["warnings"]] == ["FROZEN_RUN", "TAKER_BUY_VOLUME_EXCEEDS_TOTAL"]
    assert data["warnings"][0]["first_ts"] == "2024-01-01T00:10:00+00:00"
    assert report.to_json() == normalize(bars, "1m").report.to_json()
    assert report.ok and set(report.counts()) == {"FROZEN_RUN", "TAKER_BUY_VOLUME_EXCEEDS_TOTAL"}


def test_examples_are_capped_at_five(make_bars):
    bars = make_bars(periods=20).assign(taker_buy_volume=1e9)
    issue = normalize(bars, "1h").report.get("TAKER_BUY_VOLUME_EXCEEDS_TOTAL")
    assert issue.count == 20 and len(issue.examples) == 5


def test_raise_for_errors(make_bars):
    bars = make_bars(periods=3)
    bars.loc[0, "open"] = 0.0
    report = validate(bars, "1h")
    with pytest.raises(InvalidBarsError):
        report.raise_for_errors()
    validate(make_bars(periods=3), "1h").raise_for_errors()  # nothing to raise
