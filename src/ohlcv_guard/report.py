"""Structured results: one ``Issue`` per problem code, aggregated over the rows it affects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ohlcv_guard.errors import InvalidBarsError

Severity = Literal["error", "warning"]
MAX_EXAMPLES = 5

# ---- error codes: the bars are refused ----
MISSING_COLUMN = "MISSING_COLUMN"
MALFORMED_TIMESTAMP = "MALFORMED_TIMESTAMP"
MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
TS_UNIT_SUSPECT = "TS_UNIT_SUSPECT"
NON_NUMERIC = "NON_NUMERIC"
BAD_TRADES = "BAD_TRADES"
CONFLICTING_DUPLICATE = "CONFLICTING_DUPLICATE"
OFF_GRID = "OFF_GRID"
NON_FINITE_PRICE = "NON_FINITE_PRICE"
NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
OHLC_RELATION = "OHLC_RELATION"
BAD_VOLUME = "BAD_VOLUME"

# ---- warning codes: the bars are accepted, and something deserves a look ----
TAKER_BUY_VOLUME_EXCEEDS_TOTAL = "TAKER_BUY_VOLUME_EXCEEDS_TOTAL"
FROZEN_RUN = "FROZEN_RUN"
VOLUME_ZERO_WITH_PRICE_MOVE = "VOLUME_ZERO_WITH_PRICE_MOVE"
SPARSE_FOR_TIMEFRAME = "SPARSE_FOR_TIMEFRAME"
PRICE_JUMP = "PRICE_JUMP"

ERROR_CODES = (MISSING_COLUMN, MALFORMED_TIMESTAMP, MISSING_TIMESTAMP, TS_UNIT_SUSPECT, NON_NUMERIC, BAD_TRADES,
               CONFLICTING_DUPLICATE, OFF_GRID, NON_FINITE_PRICE, NON_POSITIVE_PRICE, OHLC_RELATION, BAD_VOLUME)
WARNING_CODES = (TAKER_BUY_VOLUME_EXCEEDS_TOTAL, FROZEN_RUN, VOLUME_ZERO_WITH_PRICE_MOVE, SPARSE_FOR_TIMEFRAME,
                 PRICE_JUMP)


def _iso(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    return (ts.tz_convert("UTC") if ts.tzinfo else ts).isoformat()


@dataclass(frozen=True)
class Issue:
    """One kind of problem: how many rows it affects, where it starts and ends, and up to five example times."""

    code: str
    severity: Severity
    message: str
    count: int
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    columns: tuple[str, ...] = ()
    examples: tuple[pd.Timestamp, ...] = ()

    @classmethod
    def from_mask(cls, code: str, severity: Severity, message: str, ts: pd.Series | None, mask: Any,
                  columns: tuple[str, ...] = (), examples: tuple[pd.Timestamp, ...] | None = None) -> Issue | None:
        """An issue for the rows where ``mask`` is true, or None when there are none. ``{n}`` and ``{first}`` in
        ``message`` are filled in."""
        flags = np.asarray(mask, dtype=bool)
        count = int(flags.sum())
        if count == 0:
            return None
        first = last = None
        if ts is not None:
            hit = ts[flags]
            valid = hit.dropna()
            if len(valid):
                first, last = valid.iloc[0], valid.iloc[-1]
            if examples is None:
                examples = tuple(valid.iloc[:MAX_EXAMPLES])
        return cls(code, severity, message.format(n=count, first=first), count, first, last, columns,
                   tuple(examples or ())[:MAX_EXAMPLES])

    def escalated(self) -> Issue:
        return Issue(self.code, "error", f"{self.message} (escalated to an error)", self.count, self.first_ts,
                     self.last_ts, self.columns, self.examples)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message, "count": self.count,
                "first_ts": _iso(self.first_ts), "last_ts": _iso(self.last_ts), "columns": list(self.columns),
                "examples": [_iso(e) for e in self.examples]}


@dataclass(frozen=True)
class Report:
    """Errors (the bars are refused) and warnings (accepted, worth a look), one ``Issue`` per code."""

    errors: tuple[Issue, ...] = ()
    warnings: tuple[Issue, ...] = ()
    rows: int = 0  # rows given
    duplicates_dropped: int = 0  # exact duplicate rows removed (the only change ever made to the data)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """No errors (warnings allowed)."""
        return not self.errors

    @property
    def issues(self) -> tuple[Issue, ...]:
        return self.errors + self.warnings

    def counts(self) -> dict[str, int]:
        """Rows affected per code."""
        return {i.code: i.count for i in self.issues}

    def get(self, code: str) -> Issue | None:
        return next((i for i in self.issues if i.code == code), None)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "rows": self.rows, "duplicates_dropped": self.duplicates_dropped,
                "errors": [i.to_dict() for i in sorted(self.errors, key=lambda i: i.code)],
                "warnings": [i.to_dict() for i in sorted(self.warnings, key=lambda i: i.code)],
                "meta": dict(sorted(self.meta.items()))}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    def raise_for_errors(self) -> None:
        """Raise ``InvalidBarsError`` (``ConflictingDuplicatesError`` for duplicates) if there are errors."""
        if self.errors:
            raise error_for(self.errors, self)


def error_for(errors: tuple[Issue, ...], report: Report | None) -> InvalidBarsError:
    """The exception for a stage's errors: its first issue's message, or a combined one for value checks."""
    from ohlcv_guard.errors import ConflictingDuplicatesError

    if any(i.code == CONFLICTING_DUPLICATE for i in errors):
        issue = next(i for i in errors if i.code == CONFLICTING_DUPLICATE)
        return ConflictingDuplicatesError(issue.message, errors, report)
    return InvalidBarsError("; ".join(i.message for i in errors), errors, report)
