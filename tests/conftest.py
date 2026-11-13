import numpy as np
import pandas as pd
import pytest

from ohlcv_guard import timeframe_ms


def _make_bars(start="2024-01-01", periods=24, timeframe="1h", seed=0, base=40_000.0) -> pd.DataFrame:
    """Valid synthetic bars in the canonical schema (UTC, millisecond timestamps)."""
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    ts = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=periods, freq=step).as_unit("ms")
    close = base + np.cumsum(rng.normal(0, 50, periods))
    open_ = np.concatenate([[base], close[:-1]])
    return pd.DataFrame(
        {
            "ts": pd.Series(ts),
            "open": open_,
            "high": np.maximum(open_, close) + rng.uniform(1, 20, periods),
            "low": np.minimum(open_, close) - rng.uniform(1, 20, periods),
            "close": close,
            "volume": rng.uniform(10, 100, periods),
        }
    )


@pytest.fixture
def make_bars():
    return _make_bars


def _ohlc_bars(pairs, start="2024-01-01", timeframe="1h") -> pd.DataFrame:
    """Bars from explicit (open, close) pairs; high/low = body +- 1. Hand-checkable prices."""
    step = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
    ts = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(pairs), freq=step).as_unit("ms")
    return pd.DataFrame({
        "ts": pd.Series(ts),
        "open": [float(o) for o, _ in pairs],
        "high": [float(max(o, c) + 1) for o, c in pairs],
        "low": [float(min(o, c) - 1) for o, c in pairs],
        "close": [float(c) for _, c in pairs],
        "volume": 10.0,
    })


@pytest.fixture
def ohlc_bars():
    return _ohlc_bars
