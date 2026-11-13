"""Which warnings run, with their thresholds, and the context every rule receives."""

from __future__ import annotations

from dataclasses import dataclass

from ohlcv_guard.schema import Schema
from ohlcv_guard.volume import VolumePolicy


@dataclass(frozen=True)
class Checks:
    """Warnings (errors always run).

    - ``taker_buy_volume``: ``TAKER_BUY_VOLUME_EXCEEDS_TOTAL`` when the taker buy volume column is present;
    - ``frozen_run``: ``FROZEN_RUN`` for at least this many consecutive bars with open = high = low = close and zero
      volume (any volume for a series without one); None turns it off;
    - ``sparse_for_timeframe``: ``SPARSE_FOR_TIMEFRAME`` for a calendar month (with at least ``sparse_min_days`` days
      that have bars) whose median bars per day is below ``sparse_ratio`` of what the timeframe implies (e.g. one bar a
      day in an hourly series); for timeframes under a day;
    - ``price_jump``: ``PRICE_JUMP`` for a close-to-close change larger than this fraction (e.g. 0.2); None = off.
    """

    taker_buy_volume: bool = True
    frozen_run: int | None = 60
    sparse_for_timeframe: bool = True
    sparse_ratio: float = 0.25
    sparse_min_days: int = 5
    price_jump: float | None = None

    def __post_init__(self) -> None:
        if self.frozen_run is not None and self.frozen_run < 2:
            raise ValueError("frozen_run must be at least 2 bars (or None)")
        if not 0 < self.sparse_ratio < 1:
            raise ValueError("sparse_ratio must be between 0 and 1")
        if self.sparse_min_days < 1:
            raise ValueError("sparse_min_days must be at least 1")
        if self.price_jump is not None and self.price_jump <= 0:
            raise ValueError("price_jump must be a positive fraction (or None)")


DEFAULT_CHECKS = Checks()


@dataclass(frozen=True)
class Context:
    schema: Schema
    volume: VolumePolicy
    checks: Checks
    tf_ms: int
    tf_label: str
