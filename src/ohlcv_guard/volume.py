"""What a series' volume means, and how unknown volume is represented.

Three meanings are kept apart:

- **genuine zero**: the market was open and nothing traded (or no quote changed): ``0``, always allowed;
- **unknown**: there was volume, but the source doesn't say how much: ``NaN`` (``unknown="nan"``), or, for data that
  already stores it as ``0``, zeros inside declared ranges (``unknown="zero_in_ranges"``);
- **not applicable**: the series has no volume at all (an index, a mark price): ``kind="none"``.

The default policy (``traded``, ``forbid``, ``allow``) is the strictest: volume is required, finite and >= 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

from ohlcv_guard.timeutil import TimeLike, to_utc_timestamp

VolumeKind = Literal["traded", "tick", "none"]
Unknown = Literal["forbid", "nan", "zero_in_ranges"]
Zero = Literal["allow", "warn_if_price_moves"]


@dataclass(frozen=True, init=False)
class VolumePolicy:
    """``kind``: ``traded`` (an exchange's traded volume), ``tick`` (a count of price changes, e.g. Forex quotes) or
    ``none``. ``unknown``: how unknown volume appears. ``unknown_ranges``: ``[start, end)`` UTC ranges where a stored
    ``0`` means unknown (``unknown="zero_in_ranges"``). ``zero``: whether a zero volume on a bar whose price moved is
    reported (``VOLUME_ZERO_WITH_PRICE_MOVE``)."""

    kind: VolumeKind = "traded"
    unknown: Unknown = "forbid"
    unknown_ranges: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()
    zero: Zero = "allow"

    def __init__(self, kind: VolumeKind = "traded", unknown: Unknown = "forbid",
                 unknown_ranges: Sequence[tuple[TimeLike, TimeLike]] = (), zero: Zero = "allow") -> None:
        if kind not in ("traded", "tick", "none"):
            raise ValueError(f"kind must be 'traded', 'tick' or 'none', got {kind!r}")
        if unknown not in ("forbid", "nan", "zero_in_ranges"):
            raise ValueError(f"unknown must be 'forbid', 'nan' or 'zero_in_ranges', got {unknown!r}")
        if zero not in ("allow", "warn_if_price_moves"):
            raise ValueError(f"zero must be 'allow' or 'warn_if_price_moves', got {zero!r}")
        ranges = []
        for start, end in unknown_ranges:
            a, b = to_utc_timestamp(start), to_utc_timestamp(end)
            if a >= b:
                raise ValueError(f"an unknown-volume range must have start < end, got {a} .. {b}")
            ranges.append((a, b))
        if ranges and unknown != "zero_in_ranges":
            raise ValueError("unknown_ranges need unknown='zero_in_ranges'")
        if unknown == "zero_in_ranges" and not ranges:
            raise ValueError("unknown='zero_in_ranges' needs unknown_ranges")
        if kind == "none" and (unknown != "forbid" or zero != "allow"):
            raise ValueError("a series without volume (kind='none') takes no unknown or zero setting")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "unknown", unknown)
        object.__setattr__(self, "unknown_ranges", tuple(sorted(ranges)))
        object.__setattr__(self, "zero", zero)

    @property
    def has_volume(self) -> bool:
        return self.kind != "none"

    def in_unknown_ranges(self, ts: pd.Series) -> np.ndarray:
        """True for the times inside a declared unknown-volume range."""
        out = np.zeros(len(ts), dtype=bool)
        for a, b in self.unknown_ranges:
            out |= ((ts >= a) & (ts < b)).to_numpy()
        return out


DEFAULT_VOLUME = VolumePolicy()


def volume_known(bars: pd.DataFrame, policy: VolumePolicy = DEFAULT_VOLUME, ts: str = "ts",
                 volume: str = "volume") -> pd.Series:
    """True where the bar's volume is known (a genuine value, zero included); False where it is unknown or the series
    has none. ``bars`` should be normalized."""
    if not policy.has_volume or volume not in bars.columns:
        return cast(pd.Series, pd.Series(False, index=bars.index))
    known = np.array(bars[volume].notna(), dtype=bool)  # a writable copy (pandas 3 hands out read-only arrays)
    if policy.unknown == "zero_in_ranges":
        zero = (bars[volume] == 0).to_numpy()
        known &= ~(zero & policy.in_unknown_ranges(bars[ts]))
    return cast(pd.Series, pd.Series(known, index=bars.index))
