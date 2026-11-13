# Changelog

## 0.1.0 (unreleased)

First release.

- **`normalize` / `validate`:**
  - canonical bars (UTC milliseconds, float64 / int64, sorted);
  - exact duplicates dropped and counted;
  - errors raised in stages: columns, conversion, duplicates, grid, values.
- **Structured `Report` and `Issue`:** error codes and warning codes, with counts, first and last times and up to five
  examples; stable JSON.
- **Warnings:** `TAKER_BUY_VOLUME_EXCEEDS_TOTAL`, `FROZEN_RUN`, `SPARSE_FOR_TIMEFRAME`,
  `VOLUME_ZERO_WITH_PRICE_MOVE` and `PRICE_JUMP`; any of them can be escalated to an error.
- **`VolumePolicy`:** traded, tick or no volume; unknown volume as NaN or as zeros in declared ranges;
  `volume_known`.
- **`find_gaps`:** `[start, end)` runs, including leading and trailing gaps in a range; checks its input is sorted.
- **`resample`:** whole multiples only, `complete` marks partial bins, extra columns kept.
- **`Timeframe`:** any `<n>{s,m,h,d}`, milliseconds or a `timedelta`.
- **Stricter than the code it started from:**
  - infinite prices are refused;
  - text, booleans and missing trade counts raise `InvalidBarsError` instead of pandas errors;
  - numeric timestamps in the wrong epoch unit are named as such.
