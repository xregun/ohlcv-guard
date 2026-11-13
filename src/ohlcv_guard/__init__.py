"""ohlcv-guard: strict validation for OHLCV bars.

Bad data raises and is never repaired; the only change ever made is dropping exact duplicate rows. Missing bars are
reported, never invented. Accepted data that deserves a look (a frozen market, sparse bars, taker volume above the
total) is reported as warnings.

    import ohlcv_guard as og

    result = og.normalize(bars, "1h")        # or InvalidBarsError
    result.bars, result.duplicates_dropped, result.report.warnings
    report = og.validate(bars, "1h")         # every error and warning, never raises for bad data
    gaps = og.find_gaps(result.bars, "1h", start, end)
"""

from ohlcv_guard.checks.config import DEFAULT_CHECKS, Checks
from ohlcv_guard.errors import (
    ConflictingDuplicatesError,
    InvalidBarsError,
    InvalidRangeError,
    OHLCVError,
    UnknownTimeframeError,
)
from ohlcv_guard.gaps import Gap, find_gaps
from ohlcv_guard.normalize import Normalized, normalize, validate
from ohlcv_guard.report import ERROR_CODES, WARNING_CODES, Issue, Report
from ohlcv_guard.resample import resample
from ohlcv_guard.schema import DEFAULT_SCHEMA, PRICE_COLUMNS, Schema, empty_bars
from ohlcv_guard.timeframes import Timeframe, timeframe_ms
from ohlcv_guard.timeutil import TS_DTYPE, epoch_ms, to_utc_series, to_utc_timestamp, utc_range
from ohlcv_guard.volume import DEFAULT_VOLUME, VolumePolicy, volume_known

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CHECKS",
    "DEFAULT_SCHEMA",
    "DEFAULT_VOLUME",
    "ERROR_CODES",
    "PRICE_COLUMNS",
    "TS_DTYPE",
    "WARNING_CODES",
    "Checks",
    "ConflictingDuplicatesError",
    "Gap",
    "InvalidBarsError",
    "InvalidRangeError",
    "Issue",
    "Normalized",
    "OHLCVError",
    "Report",
    "Schema",
    "Timeframe",
    "UnknownTimeframeError",
    "VolumePolicy",
    "__version__",
    "empty_bars",
    "epoch_ms",
    "find_gaps",
    "normalize",
    "resample",
    "timeframe_ms",
    "to_utc_series",
    "to_utc_timestamp",
    "utc_range",
    "validate",
    "volume_known",
]
