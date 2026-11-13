"""``normalize`` (bars you can trust, or an exception) and ``validate`` (a report, never an exception for bad data).

Policy: bad data raises; it is never repaired. The only change ever made is dropping *exact* duplicate rows (counted).
Missing bars are reported (``find_gaps``), never invented. Checks run in stages, and ``normalize`` stops at the first
stage with errors: columns -> conversion (timestamps, numbers) -> duplicates -> timeframe grid -> values. Warnings run
on bars that passed every stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from ohlcv_guard import report as r
from ohlcv_guard.checks.config import DEFAULT_CHECKS, Checks, Context
from ohlcv_guard.checks.values import integer_checks, price_checks, volume_checks
from ohlcv_guard.checks.warnings import warning_issues
from ohlcv_guard.errors import InvalidBarsError
from ohlcv_guard.schema import DEFAULT_SCHEMA, PRICE_COLUMNS, Schema
from ohlcv_guard.timeframes import Timeframe, TimeframeLike
from ohlcv_guard.timeutil import epoch_ms, guess_epoch_unit, is_epoch_numbers
from ohlcv_guard.volume import DEFAULT_VOLUME, VolumePolicy


@dataclass(frozen=True)
class Normalized:
    """Bars sorted by time with canonical dtypes, the exact duplicates dropped, and the report (warnings only)."""

    bars: pd.DataFrame
    duplicates_dropped: int
    report: r.Report


class _Stop(Exception):  # internal control flow
    pass


class _Run:
    def __init__(self, bars: pd.DataFrame, ctx: Context, collect: bool, escalate: frozenset[str]) -> None:
        self.bars, self.ctx, self.collect, self.escalate = bars, ctx, collect, escalate
        self.errors: list[r.Issue] = []
        self.warnings: list[r.Issue] = []
        self.dropped = 0
        self.meta: dict[str, object] = {"timeframe": ctx.tf_label, "volume_kind": ctx.volume.kind}

    def report(self) -> r.Report:
        return r.Report(tuple(self.errors), tuple(self.warnings), len(self.bars), self.dropped, self.meta)

    def stage(self, issues: Iterable[r.Issue | None], message: str | None = None, fatal: bool = False) -> None:
        """Record a stage's errors; ``normalize`` raises here, ``validate`` carries on unless ``fatal``."""
        found = [i for i in issues if i is not None]
        self.errors.extend(found)
        if not found:
            return
        if not self.collect:
            exc = r.error_for(tuple(found), self.report())
            if message is not None and type(exc) is InvalidBarsError:
                exc = InvalidBarsError(message, tuple(found), self.report())
            raise exc
        if fatal:
            raise _Stop

    def run(self) -> pd.DataFrame | None:
        s, ctx = self.ctx.schema, self.ctx
        required = [s.ts, *PRICE_COLUMNS, *([s.volume] if ctx.volume.has_volume else [])]
        missing = [col for col in required if col not in self.bars.columns]
        if missing:
            message = f"Bars are missing required columns: {missing}"
            self.stage([r.Issue(r.MISSING_COLUMN, "error", message, len(missing), columns=tuple(missing))], fatal=True)
        volume_cols = [s.volume] if s.volume in self.bars.columns and ctx.volume.has_volume else []
        columns = [s.ts, *PRICE_COLUMNS, *volume_cols, *[c for c in s.extra if c in self.bars.columns and c not in
                                                         (s.ts, *PRICE_COLUMNS, s.volume)]]
        df = self.bars[columns].copy()

        # ---- conversion: timestamps ----
        raw_ts = df[s.ts]
        conversion: list[r.Issue | None] = []
        ts_ok = True
        if is_epoch_numbers(raw_ts):
            guessed = guess_epoch_unit(raw_ts)
            if guessed is not None and guessed != s.unit:
                conversion.append(r.Issue(r.TS_UNIT_SUSPECT, "error",
                                          f"numeric timestamps look like epoch {guessed}, not {s.unit}: "
                                          f"pass Schema(unit={guessed!r}) if that's what they are", len(df),
                                          columns=(s.ts,)))
                ts_ok = False
        if ts_ok:
            try:
                if is_epoch_numbers(raw_ts):
                    converted = pd.to_datetime(raw_ts, unit=s.unit, utc=True)
                else:
                    converted = pd.to_datetime(raw_ts, utc=True)
                df[s.ts] = converted.dt.as_unit("ms")
            except (ValueError, TypeError, OverflowError) as exc:
                conversion.append(r.Issue(r.MALFORMED_TIMESTAMP, "error", f"timestamps can't be read: {exc}", len(df),
                                          columns=(s.ts,)))
                ts_ok = False
        if ts_ok:
            conversion.append(r.Issue.from_mask(r.MISSING_TIMESTAMP, "error", "{n} bar(s) have no timestamp", None,
                                                df[s.ts].isna().to_numpy(), (s.ts,)))
            ts_ok = not df[s.ts].isna().any()
        ts_for_issues = df[s.ts] if ts_ok else None

        # ---- conversion: numbers ----
        skip: dict[str, np.ndarray] = {}
        integers: dict[str, pd.Series] = {}
        for col in columns[1:]:
            values = df[col]
            if pd.api.types.is_bool_dtype(values):
                bad = np.ones(len(df), dtype=bool)
                numbers = pd.Series(np.nan, index=df.index)
            else:
                numbers = pd.to_numeric(values, errors="coerce")
                bad = np.array(numbers.isna() & values.notna(), dtype=bool)  # writable (pandas 3: read-only views)
                if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
                    bad |= values.map(lambda v: isinstance(v, bool)).to_numpy(dtype=bool)
            if bad.any():
                skip[col] = bad
                message = f"{{n}} value(s) in {col} aren't numbers"
                conversion.append(r.Issue.from_mask(r.NON_NUMERIC, "error", message, ts_for_issues, bad, (col,)))
            if col in s.integer:
                integers[col] = numbers.astype("float64")
                df[col] = numbers.astype("float64")
            else:
                df[col] = numbers.astype("float64")
        if integers:
            raw_int = {c: v.where(~skip[c], 0.0) if c in skip else v for c, v in integers.items()}
            conversion.extend(integer_checks(df, ctx, raw_int, ts_for_issues))
        self.stage(conversion, fatal=not ts_ok)
        for col in integers:
            if not self.errors:
                df[col] = df[col].astype("int64")

        # ---- duplicates ----
        df = df.sort_values(s.ts, kind="stable", ignore_index=True)
        rows_before = len(df)
        df = df.drop_duplicates(ignore_index=True)
        self.dropped = rows_before - len(df)
        dup = df[s.ts].duplicated(keep=False).to_numpy()
        if dup.any():
            conflicting = df.loc[dup, s.ts].unique()
            shown = ", ".join(str(ts) for ts in conflicting[:5])
            self.stage([r.Issue(r.CONFLICTING_DUPLICATE, "error",
                                f"{len(conflicting)} timestamp(s) have conflicting bars, e.g. {shown}",
                                len(conflicting), pd.Timestamp(conflicting[0]), pd.Timestamp(conflicting[-1]), (s.ts,),
                                tuple(pd.Timestamp(t) for t in conflicting[: r.MAX_EXAMPLES]))])

        # ---- timeframe grid ----
        misaligned = epoch_ms(df[s.ts]) % ctx.tf_ms != 0
        if misaligned.any():
            first = df.loc[misaligned, s.ts].iloc[0]
            self.stage([r.Issue.from_mask(r.OFF_GRID, "error",
                                          f"{{n}} bar(s) are not aligned to the {ctx.tf_label} grid, e.g. {first}",
                                          df[s.ts], misaligned, (s.ts,))])

        # ---- values ----
        found = price_checks(df, ctx, skip) + volume_checks(df, ctx, skip)
        if found:
            union = np.any(np.vstack([rows for _, rows in found]), axis=0)
            first = df.loc[union, s.ts].iloc[0]
            codes = ", ".join(issue.code for issue, _ in found)
            self.stage([issue for issue, _ in found],
                       f"{int(union.sum())} bar(s) fail OHLC/volume sanity checks ({codes}), e.g. ts={first}")

        # ---- warnings ----
        if not self.errors or self.collect:
            for issue in warning_issues(df, ctx):
                if issue.code in self.escalate:
                    self.stage([issue.escalated()])
                else:
                    self.warnings.append(issue)
        return cast(pd.DataFrame, df)


def _context(timeframe: TimeframeLike, schema: Schema, volume: VolumePolicy, checks: Checks) -> Context:
    tf = Timeframe.parse(timeframe)
    label = timeframe if isinstance(timeframe, str) else tf.label
    return Context(schema, volume, checks, tf.ms, label)


def normalize(bars: pd.DataFrame, timeframe: TimeframeLike, *, schema: Schema = DEFAULT_SCHEMA,
              volume: VolumePolicy = DEFAULT_VOLUME, checks: Checks = DEFAULT_CHECKS,
              escalate: Iterable[str] = ()) -> Normalized:
    """Canonical bars (sorted by time, UTC milliseconds, float64 / int64), or ``InvalidBarsError``
    (``ConflictingDuplicatesError`` for two different bars at one time). Exact duplicates are dropped and counted;
    nothing else is changed. ``escalate``: warning codes to treat as errors."""
    ctx = _context(timeframe, schema, volume, checks)
    run = _Run(bars, ctx, collect=False, escalate=_codes(escalate))
    df = run.run()
    assert df is not None
    return Normalized(df, run.dropped, run.report())


def validate(bars: pd.DataFrame, timeframe: TimeframeLike, *, schema: Schema = DEFAULT_SCHEMA,
             volume: VolumePolicy = DEFAULT_VOLUME, checks: Checks = DEFAULT_CHECKS,
             escalate: Iterable[str] = ()) -> r.Report:
    """Every error and warning, without raising for bad data. Stages after a failed one still run where they can (not
    after a missing column or unreadable timestamps)."""
    ctx = _context(timeframe, schema, volume, checks)
    run = _Run(bars, ctx, collect=True, escalate=_codes(escalate))
    with suppress(_Stop):
        run.run()
    return run.report()


def _codes(escalate: Iterable[str]) -> frozenset[str]:
    codes = frozenset(escalate)
    unknown = codes - set(r.WARNING_CODES)
    if unknown:
        raise ValueError(f"only warning codes can be escalated; unknown: {sorted(unknown)}")
    return codes
