"""Value rules (errors) on converted, sorted bars. Each returns ``(issue, rows)`` pairs: the issue and the boolean rows
behind it. ``skip`` masks cells that couldn't be converted (already reported as NON_NUMERIC), so one bad cell isn't
counted twice."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ohlcv_guard import report as r
from ohlcv_guard.checks.config import Context
from ohlcv_guard.schema import PRICE_COLUMNS

Found = list[tuple[r.Issue, np.ndarray]]


def _skipped(skip: dict[str, np.ndarray], cols: tuple[str, ...], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for col in cols:
        if col in skip:
            out |= skip[col]
    return out


def _add(found: Found, code: str, message: str, ts: pd.Series, rows: np.ndarray, columns: tuple[str, ...]) -> None:
    issue = r.Issue.from_mask(code, "error", message, ts, rows, columns)
    if issue is not None:
        found.append((issue, rows))


def price_checks(df: pd.DataFrame, ctx: Context, skip: dict[str, np.ndarray]) -> Found:
    ts = df[ctx.schema.ts]
    prices = df[list(PRICE_COLUMNS)].to_numpy(dtype="float64")
    all_finite = np.isfinite(prices).all(axis=1)
    o, h, lo, c = prices.T
    with np.errstate(invalid="ignore"):
        relation = all_finite & ((lo > np.minimum(o, c)) | (h < np.maximum(o, c)))
    found: Found = []
    _add(found, r.NON_FINITE_PRICE, "{n} bar(s) have a missing or infinite price, e.g. ts={first}", ts,
         ~all_finite & ~_skipped(skip, PRICE_COLUMNS, len(df)), PRICE_COLUMNS)
    _add(found, r.NON_POSITIVE_PRICE, "{n} bar(s) have a zero or negative price, e.g. ts={first}", ts,
         all_finite & (prices <= 0).any(axis=1), PRICE_COLUMNS)
    _add(found, r.OHLC_RELATION, "{n} bar(s) have a low above the open/close or a high below them, e.g. ts={first}",
         ts, relation, PRICE_COLUMNS)
    return found


def volume_checks(df: pd.DataFrame, ctx: Context, skip: dict[str, np.ndarray]) -> Found:
    """Volume: finite and >= 0 (NaN allowed only with ``unknown="nan"``); float extras: no infinity, nothing negative
    (NaN allowed: unknown). One BAD_VOLUME issue for all the columns."""
    s, ts, n = ctx.schema, df[ctx.schema.ts], len(df)
    rows = np.zeros(n, dtype=bool)
    columns: list[str] = []
    checked = ([s.volume] if ctx.volume.has_volume and s.volume in df.columns else []) + \
        [c for c in s.extra if c in df.columns and c not in s.integer]
    for col in checked:
        x = df[col].to_numpy(dtype="float64")
        bad = np.isinf(x) | (x < 0)
        if col == s.volume and ctx.volume.unknown != "nan":
            bad |= np.isnan(x)
        bad &= ~_skipped(skip, (col,), n)
        if bad.any():
            rows |= bad
            columns.append(col)
    found: Found = []
    _add(found, r.BAD_VOLUME, f"{{n}} bar(s) have a missing, infinite or negative {'/'.join(columns)}, "
         "e.g. ts={first}", ts, rows, tuple(columns))
    return found


def integer_checks(df: pd.DataFrame, ctx: Context, raw: dict[str, pd.Series], ts: pd.Series | None) -> list[r.Issue]:
    """Integer extras (``trades``), before the int64 cast: whole, finite and >= 0."""
    out = []
    for col, values in raw.items():
        x = values.to_numpy(dtype="float64")
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(x) | (x < 0) | (np.mod(x, 1) != 0)
        issue = r.Issue.from_mask(r.BAD_TRADES, "error",
                                  f"{{n}} bar(s) have a missing, negative or fractional {col}, e.g. ts={{first}}",
                                  ts, bad, (col,))
        if issue:
            out.append(issue)
    return out
