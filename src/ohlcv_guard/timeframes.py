"""Timeframes: ``"<n><unit>"`` with unit ``s``, ``m``, ``h`` or ``d`` (e.g. ``"1m"``, ``"4h"``, ``"1d"``), a positive
number of milliseconds, or a ``timedelta``.

Bars sit on a grid anchored at the Unix epoch in UTC: a ``4h`` bar opens at 00:00, 04:00, ... UTC and a ``1d`` bar at
00:00 UTC. Weeks and months aren't supported, because their grid needs an anchor this package doesn't assume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Union, cast

import pandas as pd

from ohlcv_guard.errors import UnknownTimeframeError

UNIT_MS = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
DAY_MS = UNIT_MS["d"]
_PATTERN = re.compile(r"^([1-9][0-9]*)([smhd])$")

TimeframeLike = Union[str, int, timedelta, pd.Timedelta, "Timeframe"]


@dataclass(frozen=True)
class Timeframe:
    """A bar duration in whole milliseconds."""

    ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.ms, int) or isinstance(self.ms, bool) or self.ms <= 0:
            raise UnknownTimeframeError(f"a timeframe must be a positive whole number of milliseconds, got {self.ms!r}")

    @classmethod
    def parse(cls, value: TimeframeLike) -> Timeframe:
        """``"15m"``, ``900000``, ``timedelta(minutes=15)`` or a ``Timeframe``."""
        if isinstance(value, Timeframe):
            return value
        if isinstance(value, str):
            match = _PATTERN.match(value.strip())
            if not match:
                raise UnknownTimeframeError(
                    f"Unknown timeframe {value!r}: use <n><unit> with unit s, m, h or d (e.g. '1m', '4h', '1d')")
            return cls(int(match.group(1)) * UNIT_MS[match.group(2)])
        if isinstance(value, bool):
            raise UnknownTimeframeError(f"Unknown timeframe {value!r}")
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, (timedelta, pd.Timedelta)):
            td = pd.Timedelta(value)
            if td.value % 1_000_000:
                raise UnknownTimeframeError(f"a timeframe must be whole milliseconds, got {value!r}")
            return cls(int(td.value // 1_000_000))
        raise UnknownTimeframeError(f"Unknown timeframe {value!r}")

    @property
    def timedelta(self) -> pd.Timedelta:
        return cast(pd.Timedelta, pd.Timedelta(milliseconds=self.ms))

    @property
    def label(self) -> str:
        """The shortest ``<n><unit>`` spelling (``"90m"`` for 5 400 000 ms, ``"1500ms"`` when no unit divides it)."""
        for unit in ("d", "h", "m", "s"):
            if self.ms % UNIT_MS[unit] == 0:
                return f"{self.ms // UNIT_MS[unit]}{unit}"
        return f"{self.ms}ms"

    def __str__(self) -> str:
        return self.label


def timeframe_ms(value: TimeframeLike) -> int:
    """The timeframe's length in milliseconds."""
    return Timeframe.parse(value).ms
