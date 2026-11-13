"""UTC normalization of timestamps.

Numbers are epoch timestamps in ``unit`` (default milliseconds). Naive datetimes and strings without an offset are
taken to be UTC; timezone-aware values are converted to UTC. Results have millisecond resolution.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

import numpy as np
import pandas as pd

from ohlcv_guard.errors import InvalidRangeError

EpochUnit = Literal["s", "ms", "us", "ns"]
TimeLike = datetime | str | pd.Timestamp | np.datetime64
TS_DTYPE = "datetime64[ms, UTC]"


def is_epoch_numbers(values: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values))


def to_utc_series(values: pd.Series, unit: EpochUnit = "ms") -> pd.Series:
    """A Series of timestamps as ``datetime64[ms, UTC]``."""
    ts = pd.to_datetime(values, unit=unit, utc=True) if is_epoch_numbers(values) else pd.to_datetime(values, utc=True)
    return ts.dt.as_unit("ms")


def to_utc_timestamp(value: TimeLike) -> pd.Timestamp:
    """One point in time in UTC (naive input is taken to be UTC)."""
    ts = pd.Timestamp(value)
    return cast(pd.Timestamp, ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC"))


def utc_range(start: TimeLike, end: TimeLike) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``(start, end)`` in UTC; raises ``InvalidRangeError`` unless start < end."""
    start_ts, end_ts = to_utc_timestamp(start), to_utc_timestamp(end)
    if start_ts >= end_ts:
        raise InvalidRangeError(f"start ({start_ts}) must be before end ({end_ts})")
    return start_ts, end_ts


def epoch_ms(ts: pd.Series) -> np.ndarray:
    """Epoch milliseconds of a datetime Series."""
    return cast(np.ndarray, ts.dt.as_unit("ms").astype("int64").to_numpy())


def ceil_to_grid(value: TimeLike, step_ms: int) -> int:
    """The first grid point (epoch ms, a multiple of ``step_ms``) at or after ``value``."""
    ms = to_utc_timestamp(value).value // 1_000_000
    return -(-ms // step_ms) * step_ms


# Plausible epoch magnitudes for 1990-2100 per unit; anything else in a numeric column is probably another unit.
_PLAUSIBLE = {"s": (6.3e8, 4.2e9), "ms": (6.3e11, 4.2e12), "us": (6.3e14, 4.2e15), "ns": (6.3e17, 4.2e18)}


def guess_epoch_unit(values: pd.Series) -> EpochUnit | None:
    """The unit whose plausible range (1990-2100) holds the median of these epoch numbers, or None."""
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return None
    median = float(np.median(np.abs(finite.to_numpy(dtype="float64"))))
    for unit, (lo, hi) in _PLAUSIBLE.items():
        if lo <= median <= hi:
            return unit  # type: ignore[return-value]
    return None
