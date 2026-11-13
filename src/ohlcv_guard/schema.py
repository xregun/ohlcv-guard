"""The canonical bar schema.

Required: a timestamp column (the bar's open time) and ``open``, ``high``, ``low``, ``close``; ``volume`` too unless the
series has no volume (``VolumePolicy(kind="none")``). Extra columns named in the schema are kept when present; any
other column is dropped. Prices and volumes become ``float64``; integer extras (``trades`` by default) ``int64``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from ohlcv_guard.timeutil import TS_DTYPE, EpochUnit

PRICE_COLUMNS = ("open", "high", "low", "close")
DEFAULT_EXTRA = ("quote_volume", "trades", "taker_buy_volume")


@dataclass(frozen=True)
class Schema:
    """Column names and the epoch unit of numeric timestamps."""

    ts: str = "ts"
    unit: EpochUnit = "ms"
    volume: str = "volume"
    extra: tuple[str, ...] = DEFAULT_EXTRA
    integer: tuple[str, ...] = ("trades",)  # extras stored as int64
    taker_buy_volume: str | None = "taker_buy_volume"  # compared with volume by TAKER_BUY_VOLUME_EXCEEDS_TOTAL


DEFAULT_SCHEMA = Schema()


def empty_bars(schema: Schema = DEFAULT_SCHEMA, with_volume: bool = True) -> pd.DataFrame:
    """An empty frame with the canonical columns and dtypes."""
    columns: dict[str, pd.Series] = {schema.ts: pd.Series(dtype=TS_DTYPE)}
    for col in (*PRICE_COLUMNS, *([schema.volume] if with_volume else [])):
        columns[col] = pd.Series(dtype="float64")
    return cast(pd.DataFrame, pd.DataFrame(columns))
