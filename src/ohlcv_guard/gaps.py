"""Missing bars, reported as ``[start, end)`` runs of missing open times. Nothing is ever filled in."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ohlcv_guard.errors import InvalidBarsError
from ohlcv_guard.timeframes import Timeframe, TimeframeLike
from ohlcv_guard.timeutil import TimeLike, ceil_to_grid, epoch_ms


@dataclass(frozen=True)
class Gap:
    """A run of missing bars: open times in ``[start, end)``."""

    start: pd.Timestamp
    end: pd.Timestamp
    missing: int


def find_gaps(bars: pd.DataFrame, timeframe: TimeframeLike, start: TimeLike | None = None,
              end: TimeLike | None = None, *, ts: str = "ts", assume_normalized: bool = False) -> list[Gap]:
    """Missing bars between the first and the last bar, and, with ``start`` / ``end``, also before the first and after
    the last bar inside ``[start, end)`` (e.g. a symbol listed after ``start``). Range bounds off the grid are rounded
    up to it. A gap is any missing grid point: a weekend in a Forex series is a gap too (this package knows no market
    calendars). ``bars`` must be sorted with unique times (``normalize`` output); that is checked unless
    ``assume_normalized``."""
    step = Timeframe.parse(timeframe).ms
    points = epoch_ms(bars[ts]) if len(bars) else np.array([], dtype="int64")
    if not assume_normalized and len(points) > 1 and not (np.diff(points) > 0).all():
        raise InvalidBarsError("find_gaps needs bars sorted by time without duplicate times: normalize() them first")

    # Virtual bars just before `start` and at the first open time >= `end` turn leading and trailing gaps into
    # ordinary gaps between neighbours.
    if start is not None:
        first_expected = ceil_to_grid(start, step)
        points = np.concatenate([[first_expected - step], points[points >= first_expected]])
    if end is not None:
        end_boundary = ceil_to_grid(end, step)
        points = np.concatenate([points[points < end_boundary], [end_boundary]])

    if len(points) < 2:
        return []
    expected_next = points[:-1] + step
    is_gap = points[1:] > expected_next
    return [
        Gap(start=pd.Timestamp(int(a), unit="ms", tz="UTC"), end=pd.Timestamp(int(b), unit="ms", tz="UTC"),
            missing=int(b - a) // step)
        for a, b in zip(expected_next[is_gap], points[1:][is_gap], strict=True)
    ]
