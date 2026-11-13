"""Warning rules on valid, sorted bars: accepted data that deserves a look."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ohlcv_guard import report as r
from ohlcv_guard.checks.config import Context
from ohlcv_guard.timeframes import DAY_MS
from ohlcv_guard.volume import volume_known


def taker_buy_volume(df: pd.DataFrame, ctx: Context) -> r.Issue | None:
    s = ctx.schema
    if not (ctx.checks.taker_buy_volume and ctx.volume.has_volume and s.taker_buy_volume
            and s.taker_buy_volume in df.columns and s.volume in df.columns):
        return None
    tb, v = df[s.taker_buy_volume].to_numpy("float64"), df[s.volume].to_numpy("float64")
    with np.errstate(invalid="ignore"):
        mask = tb > v * (1 + 1e-9) + 1e-12
    return r.Issue.from_mask(r.TAKER_BUY_VOLUME_EXCEEDS_TOTAL, "warning",
                             "{n} bar(s) have a taker buy volume above their total volume, e.g. ts={first}",
                             df[s.ts], mask, (s.taker_buy_volume, s.volume))


def frozen_run(df: pd.DataFrame, ctx: Context) -> r.Issue | None:
    """Runs of consecutive bars with one price (open = high = low = close) and zero volume: a halted, settled or
    stale market whose source kept publishing bars."""
    min_run, s = ctx.checks.frozen_run, ctx.schema
    if min_run is None or len(df) < min_run:
        return None
    o, h, lo, c = (df[col].to_numpy("float64") for col in ("open", "high", "low", "close"))
    flat = (o == h) & (h == lo) & (lo == c)
    if ctx.volume.has_volume and s.volume in df.columns:
        flat &= df[s.volume].to_numpy("float64") == 0
    if not flat.any():
        return None
    run_id = np.cumsum(~flat)  # constant inside a run of flat bars
    lengths = pd.Series(flat).groupby(run_id).transform("sum").to_numpy()
    in_run = flat & (lengths >= min_run)
    if not in_run.any():
        return None
    starts = in_run & ~np.concatenate([[False], in_run[:-1]])
    ts = df[s.ts]
    runs = int(starts.sum())
    longest = int(lengths[in_run].max())
    message = (f"{{n}} bar(s) in {runs} run(s) of at least {min_run} bars with one price and no volume "
               f"(longest {longest}), from {{first}}")
    return r.Issue.from_mask(r.FROZEN_RUN, "warning", message, ts, in_run, ("open", "high", "low", "close", s.volume),
                             tuple(ts[starts].iloc[: r.MAX_EXAMPLES]))


def zero_volume_moving(df: pd.DataFrame, ctx: Context) -> r.Issue | None:
    s = ctx.schema
    if ctx.volume.zero != "warn_if_price_moves" or s.volume not in df.columns:
        return None
    known = volume_known(df, ctx.volume, s.ts, s.volume).to_numpy()
    mask = known & (df[s.volume].to_numpy("float64") == 0) & (df["high"].to_numpy() > df["low"].to_numpy())
    return r.Issue.from_mask(r.VOLUME_ZERO_WITH_PRICE_MOVE, "warning",
                             "{n} bar(s) have zero volume although their price moved, e.g. ts={first}",
                             df[s.ts], mask, (s.volume,))


def sparse_for_timeframe(df: pd.DataFrame, ctx: Context) -> r.Issue | None:
    """Months whose typical day has far fewer bars than the timeframe implies: e.g. one daily bar a day stored as 1h."""
    cfg = ctx.checks
    if not cfg.sparse_for_timeframe or ctx.tf_ms >= DAY_MS or df.empty:
        return None
    ts = df[ctx.schema.ts]
    day = ts.dt.floor("D")
    per_day = day.value_counts()
    months = per_day.index.tz_localize(None).to_period("M")
    stats = pd.DataFrame({"bars": per_day.to_numpy(), "month": months}).groupby("month")["bars"].agg(["median", "size"])
    expected = DAY_MS / ctx.tf_ms
    flagged = stats[(stats["size"] >= cfg.sparse_min_days) & (stats["median"] < cfg.sparse_ratio * expected)]
    if flagged.empty:
        return None
    bar_month = ts.dt.tz_localize(None).dt.to_period("M")
    mask = bar_month.isin(flagged.index).to_numpy()
    shown = ", ".join(str(m) for m in flagged.index[:6]) + (f" and {len(flagged) - 6} more" if len(flagged) > 6 else "")
    lowest = float(flagged["median"].min())
    message = (f"{{n}} bar(s) in {len(flagged)} month(s) have a median of {lowest:g} bar(s) a day where "
               f"{ctx.tf_label} implies up to {expected:g} ({shown}): bars of another timeframe, or a sparse source?")
    return r.Issue.from_mask(r.SPARSE_FOR_TIMEFRAME, "warning", message, ts, mask, (ctx.schema.ts,))


def price_jump(df: pd.DataFrame, ctx: Context) -> r.Issue | None:
    limit = ctx.checks.price_jump
    if limit is None or len(df) < 2:
        return None
    close = df["close"].to_numpy("float64")
    change = np.concatenate([[0.0], close[1:] / close[:-1] - 1])
    message = f"{{n}} bar(s) close more than {limit:.0%} away from the previous close, e.g. ts={{first}}"
    return r.Issue.from_mask(r.PRICE_JUMP, "warning", message, df[ctx.schema.ts], np.abs(change) > limit, ("close",))


RULES = (taker_buy_volume, frozen_run, zero_volume_moving, sparse_for_timeframe, price_jump)


def warning_issues(df: pd.DataFrame, ctx: Context) -> list[r.Issue]:
    return [i for i in (rule(df, ctx) for rule in RULES) if i is not None]
