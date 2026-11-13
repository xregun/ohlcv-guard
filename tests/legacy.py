"""The behaviour ohlcv-guard started from (0.0, frozen): ``normalize`` must return exactly this for every input this
accepts. Used only by the parity tests."""

import pandas as pd

from ohlcv_guard import ConflictingDuplicatesError, InvalidBarsError, timeframe_ms
from ohlcv_guard.timeutil import epoch_ms, to_utc_series

PRICE_COLUMNS = ["open", "high", "low", "close"]
REQUIRED_COLUMNS = ["ts", *PRICE_COLUMNS, "volume"]
OPTIONAL_COLUMNS = ["quote_volume", "trades", "taker_buy_volume"]


def legacy_normalize(bars: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, int]:
    tf_ms = timeframe_ms(timeframe)
    missing = [col for col in REQUIRED_COLUMNS if col not in bars.columns]
    if missing:
        raise InvalidBarsError(f"Bars are missing required columns: {missing}")
    columns = REQUIRED_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in bars.columns]
    df = bars[columns].copy()
    df["ts"] = to_utc_series(df["ts"])
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col]).astype("int64" if col == "trades" else "float64")
    df = df.sort_values("ts", kind="stable", ignore_index=True)
    rows_before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    dropped = rows_before - len(df)
    conflicting = df.loc[df["ts"].duplicated(keep=False), "ts"].unique()
    if len(conflicting):
        raise ConflictingDuplicatesError(f"{len(conflicting)} timestamp(s) have conflicting bars")
    misaligned = df.loc[epoch_ms(df["ts"]) % tf_ms != 0, "ts"]
    if len(misaligned):
        raise InvalidBarsError("not aligned")
    prices = df[PRICE_COLUMNS]
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    bad = (prices.isna().any(axis=1) | (prices <= 0).any(axis=1) | (df["low"] > body_low) | (df["high"] < body_high)
           | df["volume"].isna() | (df["volume"] < 0))
    if bad.any():
        raise InvalidBarsError("sanity")
    return df, dropped
