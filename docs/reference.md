# Reference

## `normalize(bars, timeframe, *, schema, volume, checks, escalate) -> Normalized`

**What it returns:** canonical bars, or raises.
- `Normalized.bars`: sorted by time, with the schema's columns in order and canonical dtypes.
- `Normalized.duplicates_dropped`: the number of exact duplicate rows removed.
- `Normalized.report`: its warnings.

**Exceptions:**
- `InvalidBarsError`: `.issues` holds the error issues and `.report` the full report.
- `ConflictingDuplicatesError`: a subclass of `InvalidBarsError`.
- `UnknownTimeframeError`: for a timeframe that can't be parsed. That's a usage error, not a data error.

The input frame is never modified.

## `validate(...) -> Report`

The same checks, without raising for bad data.
- **Stages run where they can:** after a missing column or unreadable timestamps, nothing further can be checked.
- **Warnings** run on whatever was readable.

## `Report`

| Member | Meaning |
|---|---|
| `errors`, `warnings` | tuples of `Issue`, one per code |
| `ok` | no errors |
| `counts()` | `{code: rows}` |
| `get(code)` | the issue for a code, or None |
| `rows` | rows given |
| `duplicates_dropped` | exact duplicates removed |
| `meta` | `timeframe`, `volume_kind` |
| `to_dict()`, `to_json()` | stable (sorted codes, ISO-8601 UTC times) |
| `raise_for_errors()` | raises as `normalize` would |

## `Issue`

| Field | Meaning |
|---|---|
| `code` | from `ERROR_CODES` / `WARNING_CODES` |
| `severity` | `"error"` / `"warning"` |
| `message` | readable text |
| `count` | rows affected (for `CONFLICTING_DUPLICATE`: timestamps; for `MISSING_COLUMN`: columns) |
| `first_ts`, `last_ts` | the first and last affected time |
| `columns` | the columns involved |
| `examples` | up to five times (for `FROZEN_RUN`: the run starts) |

## `find_gaps(bars, timeframe, start=None, end=None, *, ts="ts", assume_normalized=False) -> list[Gap]`

- **What it returns:** `[start, end)` runs of missing grid points. With `start` / `end`, gaps before the first bar and
  after the last bar inside the range are included, and bounds off the grid are rounded up.
- **Input:** the bars must be sorted with unique times. That's checked unless `assume_normalized=True`.

## `resample(bars, source, target, *, keep=(), mark_partial=True, ts="ts", volume="volume") -> DataFrame`

Bars of `target` built from sorted bars of `source`, where `target` must be a whole multiple of `source`:
- **Prices and volume:** first open, max high, min low, last close, summed volume;
- **`keep`:** summed, with integer dtypes preserved;
- **`complete`:** True when every source bar of the bin was there.

## Timeframes

- **`Timeframe.parse(value)`** accepts `"<n><unit>"` with unit s, m, h or d; a positive integer of milliseconds; a
  `timedelta`; or a `Timeframe`. `.ms`, `.label`, `.timedelta`.
- **`timeframe_ms(value)`** gives the length in milliseconds.

## Time helpers

`to_utc_series`, `to_utc_timestamp`, `utc_range`, `epoch_ms`, `empty_bars`.
