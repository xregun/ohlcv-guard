from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import ohlcv_guard as og
from ohlcv_guard import ConflictingDuplicatesError, InvalidBarsError, normalize, to_utc_timestamp, validate
from ohlcv_guard.timeutil import TS_DTYPE

JAN1 = pd.Timestamp("2024-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def norm(bars, tf="1h", **kw):
    r = normalize(bars, tf, **kw)
    return r.bars, r.duplicates_dropped


# ---- UTC normalization (ported) ----

def test_epoch_millisecond_integers_become_utc(make_bars):
    bars = make_bars(periods=3)
    bars["ts"] = bars["ts"].astype("int64")
    df, _ = norm(bars)
    assert str(df["ts"].dtype) == TS_DTYPE
    assert df["ts"].iloc[0] == JAN1


def test_naive_datetimes_are_treated_as_utc(make_bars):
    bars = make_bars(periods=3)
    bars["ts"] = bars["ts"].dt.tz_localize(None)
    df, _ = norm(bars)
    assert df["ts"].iloc[0] == JAN1
    assert str(df["ts"].dt.tz) == "UTC"


@pytest.mark.parametrize(("zone", "local_wall_clock"), [
    ("America/New_York", "2023-12-31 19:00"),  # UTC-5
    ("Asia/Karachi", "2024-01-01 05:00"),      # UTC+5
])
def test_aware_datetimes_are_converted_to_utc(make_bars, zone, local_wall_clock):
    bars = make_bars(periods=3)
    bars["ts"] = bars["ts"].dt.tz_convert(zone)
    assert str(bars["ts"].iloc[0].tz_localize(None)) == f"{local_wall_clock}:00"
    df, _ = norm(bars)
    assert df["ts"].iloc[0] == JAN1
    assert str(df["ts"].dtype) == TS_DTYPE


def test_iso_strings_with_offsets_are_converted_to_utc(make_bars):
    bars = make_bars(periods=2)
    bars["ts"] = ["2024-01-01T05:00:00+05:00", "2023-12-31T20:00:00-05:00"]
    df, _ = norm(bars)
    assert list(df["ts"]) == [JAN1, JAN1 + HOUR]


def test_to_utc_timestamp_handles_naive_and_aware_values():
    assert to_utc_timestamp("2024-01-01") == JAN1
    assert to_utc_timestamp(datetime(2024, 1, 1, 5, tzinfo=timezone(timedelta(hours=5)))) == JAN1
    assert to_utc_timestamp(datetime(2024, 1, 1)).tzinfo is not None


# ---- ordering (ported) ----

def test_unsorted_bars_are_sorted_and_rows_stay_intact(make_bars):
    bars = make_bars(periods=50)
    shuffled = bars.sample(frac=1, random_state=7).reset_index(drop=True)
    assert not shuffled["ts"].is_monotonic_increasing
    df, _ = norm(shuffled)
    assert df["ts"].is_monotonic_increasing
    pd.testing.assert_frame_equal(df, bars, check_dtype=False)


# ---- duplicates (ported) ----

def test_exact_duplicates_are_dropped_and_counted(make_bars):
    bars = make_bars(periods=10)
    with_dupes = pd.concat([bars, bars.iloc[[2, 5, 5]]], ignore_index=True)
    df, dropped = norm(with_dupes)
    assert dropped == 3
    assert len(df) == 10
    assert not df["ts"].duplicated().any()


def test_conflicting_duplicates_raise(make_bars):
    bars = make_bars(periods=10)
    conflict = bars.iloc[[4]].copy()
    conflict["close"] = conflict["close"] + 1  # same ts, different value; still valid OHLC
    conflict["high"] = conflict["high"] + 1
    with pytest.raises(ConflictingDuplicatesError, match="1 timestamp"):
        norm(pd.concat([bars, conflict], ignore_index=True))


# ---- integrity (ported) ----

def test_bars_off_the_timeframe_grid_raise(make_bars):
    bars = make_bars(periods=3)
    bars.loc[1, "ts"] = bars.loc[1, "ts"] + pd.Timedelta(minutes=1)
    with pytest.raises(InvalidBarsError, match="aligned"):
        norm(bars)


@pytest.mark.parametrize(("column", "value"), [
    ("high", 1.0),          # high below open/close
    ("low", 1_000_000.0),   # low above open/close
    ("volume", -1.0),
    ("open", 0.0),
    ("close", np.nan),
])
def test_ohlc_sanity_violations_raise(make_bars, column, value):
    bars = make_bars(periods=5)
    bars.loc[2, column] = value
    with pytest.raises(InvalidBarsError):
        norm(bars)


def test_missing_required_column_raises(make_bars):
    with pytest.raises(InvalidBarsError, match="volume"):
        norm(make_bars(periods=3).drop(columns="volume"))


# ---- new: values ----

@pytest.mark.parametrize(("column", "value", "code"), [
    ("open", np.inf, "NON_FINITE_PRICE"),
    ("low", -np.inf, "NON_FINITE_PRICE"),
    ("close", np.nan, "NON_FINITE_PRICE"),
    ("low", -5.0, "NON_POSITIVE_PRICE"),
    ("high", 1.0, "OHLC_RELATION"),
    ("volume", np.inf, "BAD_VOLUME"),
    ("volume", np.nan, "BAD_VOLUME"),
    ("volume", -0.5, "BAD_VOLUME"),
])
def test_each_value_rule_has_its_code(make_bars, column, value, code):
    bars = make_bars(periods=5)
    if column == "high":
        bars.loc[2, "high"] = bars.loc[2, "low"] - 1
    else:
        bars.loc[2, column] = value
    with pytest.raises(InvalidBarsError, match="fail OHLC/volume sanity checks") as exc:
        norm(bars)
    assert [i.code for i in exc.value.issues] == [code]
    assert exc.value.issues[0].examples == (JAN1 + 2 * HOUR,)


def test_infinite_high_and_open_are_refused(make_bars):
    """Accepted before 0.1: an inf open with an inf high passed every comparison."""
    bars = make_bars(periods=4)
    bars.loc[1, ["open", "high"]] = np.inf
    assert validate(bars, "1h").counts() == {"NON_FINITE_PRICE": 1}


def test_zero_volume_is_a_genuine_value(make_bars):
    bars = make_bars(periods=4)
    bars["volume"] = 0.0
    assert normalize(bars, "1h").report.ok


@pytest.mark.parametrize("value", ["abc", "1.2.3", True, object()])
def test_non_numbers_are_named_with_their_column(make_bars, value):
    bars = make_bars(periods=4)
    bars["open"] = bars["open"].astype(object)
    bars.loc[1, "open"] = value
    with pytest.raises(InvalidBarsError, match="in open aren't numbers") as exc:
        norm(bars)
    assert exc.value.issues[0].code == "NON_NUMERIC"
    assert exc.value.issues[0].examples == (JAN1 + HOUR,)


def test_numeric_strings_and_integers_are_numbers(make_bars):
    bars = make_bars(periods=3)
    bars["volume"] = ["1", "2.5", "3e2"]
    bars["open"] = bars["open"].round().astype("int64")
    bars["low"] = np.minimum(bars["low"], bars["open"])
    bars["high"] = np.maximum(bars["high"], bars["open"])
    df, _ = norm(bars)
    assert df["volume"].tolist() == [1.0, 2.5, 300.0] and df["open"].dtype == "float64"


def test_a_bad_cell_is_reported_once(make_bars):
    bars = make_bars(periods=4)
    bars["close"] = bars["close"].astype(object)
    bars.loc[2, "close"] = "n/a"
    assert validate(bars, "1h").counts() == {"NON_NUMERIC": 1}  # not also NON_FINITE_PRICE


@pytest.mark.parametrize("value", [np.nan, -1, 2.5, np.inf])
def test_bad_trade_counts_are_refused(make_bars, value):
    bars = make_bars(periods=4)
    bars["trades"] = [10.0, value, 3.0, 4.0]
    with pytest.raises(InvalidBarsError, match="trades") as exc:
        norm(bars)
    assert exc.value.issues[0].code == "BAD_TRADES"


def test_trade_counts_become_int64(make_bars):
    bars = make_bars(periods=3).assign(trades=[1.0, 2.0, 3.0])
    assert norm(bars)[0]["trades"].dtype == "int64"


@pytest.mark.parametrize(("column", "value"), [("quote_volume", np.inf), ("taker_buy_volume", -1.0)])
def test_float_extras_refuse_infinity_and_negatives(make_bars, column, value):
    bars = make_bars(periods=3).assign(quote_volume=1.0, taker_buy_volume=0.5)
    bars.loc[1, column] = value
    with pytest.raises(InvalidBarsError) as exc:
        norm(bars)
    assert exc.value.issues[0].code == "BAD_VOLUME" and exc.value.issues[0].columns == (column,)


def test_float_extras_may_be_unknown(make_bars):
    bars = make_bars(periods=3).assign(quote_volume=[1.0, np.nan, 2.0])
    assert normalize(bars, "1h").report.ok


def test_columns_outside_the_schema_are_dropped_and_extras_kept_in_order(make_bars):
    bars = make_bars(periods=3).assign(note="x", taker_buy_volume=1.0, quote_volume=2.0)
    assert list(norm(bars)[0].columns) == ["ts", "open", "high", "low", "close", "volume", "quote_volume",
                                           "taker_buy_volume"]


def test_a_custom_schema(make_bars):
    bars = make_bars(periods=3).rename(columns={"ts": "time", "volume": "vol"}).assign(spread=0.1)
    schema = og.Schema(ts="time", volume="vol", extra=("spread",), integer=(), taker_buy_volume=None)
    df = normalize(bars, "1h", schema=schema).bars
    assert list(df.columns) == ["time", "open", "high", "low", "close", "vol", "spread"]


# ---- new: timestamps ----

@pytest.mark.parametrize(("unit", "factor"), [("s", 1e-3), ("us", 1e3), ("ns", 1e6)])
def test_numbers_in_another_epoch_unit_are_named(make_bars, unit, factor):
    bars = make_bars(periods=3)
    bars["ts"] = (bars["ts"].astype("int64") * factor).astype("int64")
    with pytest.raises(InvalidBarsError, match=f"look like epoch {unit}, not ms") as exc:
        norm(bars)
    assert exc.value.issues[0].code == "TS_UNIT_SUSPECT"
    df = normalize(bars, "1h", schema=og.Schema(unit=unit)).bars
    assert df["ts"].iloc[0] == JAN1


def test_unreadable_and_missing_timestamps(make_bars):
    bars = make_bars(periods=3)
    bars["ts"] = ["2024-01-01", "not a time", "2024-01-01T02:00"]
    with pytest.raises(InvalidBarsError, match="can't be read") as exc:
        norm(bars)
    assert exc.value.issues[0].code == "MALFORMED_TIMESTAMP"
    bars["ts"] = [JAN1, None, JAN1 + 2 * HOUR]
    with pytest.raises(InvalidBarsError, match="no timestamp"):
        norm(bars)


def test_mixed_offsets_in_one_column(make_bars):
    bars = make_bars(periods=3)
    bars["ts"] = ["2024-01-01T00:00:00Z", "2024-01-01T02:00:00+01:00", "2023-12-31T21:00:00-05:00"]
    assert list(norm(bars)[0]["ts"]) == [JAN1, JAN1 + HOUR, JAN1 + 2 * HOUR]


def test_dst_transitions_in_local_aware_input(make_bars):
    """Europe/Berlin springs forward on 2024-03-31 (02:00 -> 03:00) and falls back on 2024-10-27: UTC is unbroken."""
    for start in ("2024-03-30 22:00", "2024-10-26 22:00"):
        bars = make_bars(start=start, periods=8)
        bars["ts"] = bars["ts"].dt.tz_convert("Europe/Berlin")
        df = norm(bars)[0]
        assert og.find_gaps(df, "1h") == [] and df["ts"].diff().dropna().eq(HOUR).all()


def test_dates_before_1990_need_an_explicit_reading(make_bars):
    bars = make_bars(start="1985-01-01", periods=3)
    bars["ts"] = bars["ts"].astype("int64")  # ms, but implausible ones: refused rather than guessed
    assert normalize(bars, "1h").bars["ts"].iloc[0] == pd.Timestamp("1985-01-01", tz="UTC")


# ---- new: duplicates ----

def test_duplicates_differing_in_an_extra_column_conflict(make_bars):
    bars = make_bars(periods=4).assign(trades=[1, 2, 3, 4])
    twin = bars.iloc[[1]].assign(trades=99)
    with pytest.raises(ConflictingDuplicatesError):
        norm(pd.concat([bars, twin], ignore_index=True))


def test_float_noise_is_a_conflict_not_a_duplicate(make_bars):
    bars = make_bars(periods=4)
    twin = bars.iloc[[1]].copy()
    twin["volume"] = twin["volume"] + 1e-12
    with pytest.raises(ConflictingDuplicatesError):
        norm(pd.concat([bars, twin], ignore_index=True))


def test_many_conflicts_show_five(make_bars):
    bars = make_bars(periods=10)
    twins = bars.copy()
    twins["volume"] = twins["volume"] + 1
    with pytest.raises(ConflictingDuplicatesError, match="10 timestamp") as exc:
        norm(pd.concat([bars, twins], ignore_index=True))
    assert len(exc.value.issues[0].examples) == 5


# ---- new: stages and validate ----

def test_normalize_stops_at_the_first_stage_and_validate_reports_all(make_bars):
    bars = make_bars(periods=6)
    bars.loc[1, "ts"] += pd.Timedelta(minutes=1)  # off the grid
    bars.loc[3, "open"] = np.inf
    bars = pd.concat([bars, bars.iloc[[4]].assign(close=bars.loc[4, "close"] + 0.001)], ignore_index=True)
    with pytest.raises(ConflictingDuplicatesError):
        norm(bars)
    assert set(validate(bars, "1h").counts()) == {"CONFLICTING_DUPLICATE", "OFF_GRID", "NON_FINITE_PRICE"}


def test_validate_stops_after_a_missing_column(make_bars):
    report = validate(make_bars(periods=3).drop(columns=["close"]), "1h")
    assert report.counts() == {"MISSING_COLUMN": 1} and not report.ok


def test_validate_reports_the_dropped_duplicates(make_bars):
    bars = make_bars(periods=3)
    report = validate(pd.concat([bars, bars.iloc[[0]]]), "1h")
    assert report.ok and report.duplicates_dropped == 1 and report.rows == 4


def test_empty_bars_are_valid(make_bars):
    r = normalize(make_bars(periods=1).iloc[0:0], "1h")
    assert r.bars.empty and r.report.ok


def test_errors_carry_the_report(make_bars):
    bars = make_bars(periods=3)
    bars.loc[1, "open"] = -1.0
    with pytest.raises(InvalidBarsError) as exc:
        norm(bars)
    assert exc.value.report is not None and exc.value.report.rows == 3


def test_an_unknown_timeframe_is_a_usage_error(make_bars):
    with pytest.raises(og.UnknownTimeframeError):
        validate(make_bars(periods=3), "1w")


def test_the_input_frame_is_never_changed(make_bars):
    bars = make_bars(periods=6).sample(frac=1, random_state=1)
    before = bars.copy()
    normalize(bars, "1h")
    validate(bars, "1h")
    pd.testing.assert_frame_equal(bars, before)
