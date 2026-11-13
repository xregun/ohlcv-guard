"""Bars of a longer timeframe from a shorter one, on the UTC grid (00:00, 04:00, ... for 4h; UTC days for 1d).

First open, highest high, lowest low, last close, summed volume (and summed ``keep`` columns). Only the bars given go
in: a missing bar stays missing, and a bin built from fewer bars than it spans is marked ``complete = False``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd

from ohlcv_guard.errors import InvalidBarsError
from ohlcv_guard.timeframes import Timeframe, TimeframeLike


def resample(bars: pd.DataFrame, source: TimeframeLike, target: TimeframeLike, *, keep: Iterable[str] = (),
             mark_partial: bool = True, ts: str = "ts", volume: str | None = "volume") -> pd.DataFrame:
    """``bars`` (sorted, one timeframe ``source``) as ``target`` bars. ``keep``: extra columns to sum (e.g. ``trades``,
    ``taker_buy_volume``). ``mark_partial``: add ``complete``, True when every ``source`` bar of the bin is there.
    ``volume=None`` for a series without volume."""
    src, dst = Timeframe.parse(source), Timeframe.parse(target)
    if dst.ms < src.ms or dst.ms % src.ms:
        raise InvalidBarsError(f"can't build {dst} bars from {src} bars: {dst} must be a multiple of {src}")
    times = bars[ts]
    if len(times) > 1 and not times.is_monotonic_increasing:
        raise InvalidBarsError("resample needs bars sorted by time: normalize() them first")
    keep = tuple(keep)
    missing = [c for c in keep if c not in bars.columns]
    if missing:
        raise InvalidBarsError(f"columns to keep aren't in the bars: {missing}")
    grouped = bars.groupby(times.dt.floor(dst.timedelta), sort=True)
    aggs: dict[str, tuple[str, str]] = {"open": ("open", "first"), "high": ("high", "max"), "low": ("low", "min"),
                                        "close": ("close", "last")}
    if volume is not None:
        aggs[volume] = (volume, "sum")
    for col in keep:
        aggs[col] = (col, "sum")
    out = grouped.agg(**aggs)
    if mark_partial:
        out["complete"] = grouped.size().to_numpy() == dst.ms // src.ms
    out = out.rename_axis(ts).reset_index()
    for col in keep:
        if pd.api.types.is_integer_dtype(bars[col]):
            out[col] = out[col].astype(np.int64)
    return cast(pd.DataFrame, out)
