"""Exceptions. All subclass ``ValueError``, so code that already treats bad input as a ``ValueError`` keeps working."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ohlcv_guard.report import Issue, Report


class OHLCVError(ValueError):
    """Base class of every error raised by ohlcv-guard."""


class InvalidBarsError(OHLCVError):
    """Bars failed a schema, value, duplicate or grid check.

    ``issues`` holds the error issues that caused it; ``report`` the full report (errors and warnings) when available.
    """

    def __init__(self, message: str, issues: tuple[Issue, ...] = (), report: Report | None = None) -> None:
        super().__init__(message)
        self.issues = issues
        self.report = report


class ConflictingDuplicatesError(InvalidBarsError):
    """Two bars share a timestamp but differ in their values."""


class InvalidRangeError(OHLCVError):
    """A time range whose start is not before its end."""


class UnknownTimeframeError(OHLCVError):
    """A timeframe that can't be parsed (e.g. ``"7x"``) or isn't supported (e.g. weeks)."""
