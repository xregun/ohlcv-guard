# ohlcv-guard

Strict validation for OHLCV bars (candles) in pandas.

- **Bad data raises; it is never repaired.** The only change ever made is dropping *exact* duplicate rows, and they
  are counted.
- **Missing bars are reported, never invented.**
- **Accepted data that deserves a look is reported as structured warnings**: a market frozen at one price, one bar a
  day in an "hourly" series, taker buy volume above the total.
- **Volume semantics are explicit**: genuine zero, unknown, and not applicable are three different things.

```python
import ohlcv_guard as og

result = og.normalize(bars, "1h")          # canonical bars, or og.InvalidBarsError
result.bars                                # sorted, UTC, datetime64[ms, UTC], float64 / int64
result.duplicates_dropped                  # exact duplicates removed
result.report.warnings                     # og.Issue(code, severity, message, count, first_ts, last_ts, examples)

report = og.validate(bars, "1h")           # every error and warning; never raises for bad data
print(report.to_json(indent=2))

gaps = og.find_gaps(result.bars, "1h", "2024-01-01", "2024-02-01")   # [Gap(start, end, missing), ...]
four_hourly = og.resample(result.bars, "1h", "4h")                    # adds `complete` for partial bins
```

## Install

```bash
pip install ohlcv-guard
```

Python 3.10+, pandas 2.1+ and NumPy 1.26+.

## The canonical schema

| Column | Required | Type after `normalize` |
|---|---|---|
| `ts` (the bar's open time) | yes | `datetime64[ms, UTC]` |
| `open`, `high`, `low`, `close` | yes | `float64` |
| `volume` | yes, unless `VolumePolicy(kind="none")` | `float64` |
| `quote_volume`, `taker_buy_volume` | kept when present | `float64` |
| `trades` | kept when present | `int64` |

Any other column is dropped. Names, the epoch unit of numeric timestamps and the extra columns are configurable:
`og.Schema(ts="time", unit="s", volume="vol", extra=("spread",), integer=())`.

**Timestamps:**
- numbers are epoch milliseconds by default;
- naive datetimes and strings without an offset are taken to be UTC;
- timezone-aware values are converted to UTC.

**The timeframe grid:** bars must sit on the UTC epoch grid of the timeframe (00:00, 04:00, ... for `4h`; 00:00 UTC
for `1d`). Weeks and months aren't supported.

## Errors: the bars are refused

| Code | Meaning |
|---|---|
| `MISSING_COLUMN` | a required column is absent |
| `MALFORMED_TIMESTAMP`, `MISSING_TIMESTAMP` | timestamps that can't be read, or empty ones |
| `TS_UNIT_SUSPECT` | numeric timestamps that look like epoch seconds (or µs, ns) when milliseconds were declared |
| `NON_NUMERIC` | a value that isn't a number (text, booleans) |
| `BAD_TRADES` | a missing, negative or fractional trade count |
| `CONFLICTING_DUPLICATE` | two different bars at one time (`ConflictingDuplicatesError`) |
| `OFF_GRID` | a bar not on the timeframe's grid |
| `NON_FINITE_PRICE`, `NON_POSITIVE_PRICE` | NaN, ±inf, zero or negative prices |
| `OHLC_RELATION` | a low above the open or close, or a high below them |
| `BAD_VOLUME` | volume missing (unless declared unknown), infinite or negative; extras infinite or negative |

`normalize` checks in stages and stops at the first one with errors: columns → conversion → duplicates → grid →
values. `validate` carries on where it can and reports everything.

## Warnings: accepted, worth a look

| Code | Default | Meaning |
|---|---|---|
| `TAKER_BUY_VOLUME_EXCEEDS_TOTAL` | on | taker buy volume above the bar's volume |
| `FROZEN_RUN` | 60 bars | consecutive bars with one price and no volume: a halted, settled or stale market |
| `SPARSE_FOR_TIMEFRAME` | on | months whose median bars per day is under 25% of what the timeframe implies |
| `VOLUME_ZERO_WITH_PRICE_MOVE` | off | zero volume on a bar whose price moved (`VolumePolicy(zero="warn_if_price_moves")`) |
| `PRICE_JUMP` | off | a close-to-close move larger than a fraction you give (`Checks(price_jump=0.2)`) |

Thresholds are set with `og.Checks(...)`. Any warning can be turned into an error with
`normalize(..., escalate=["FROZEN_RUN"])`.

## Volume semantics

```python
og.VolumePolicy(kind="traded")                      # default: an exchange's traded volume, required, >= 0
og.VolumePolicy(kind="tick")                        # a count of price changes (e.g. Forex quotes)
og.VolumePolicy(kind="none")                        # an index or a mark price: no volume
og.VolumePolicy(unknown="nan")                      # NaN means "unknown" and is allowed
og.VolumePolicy(kind="tick", unknown="zero_in_ranges",
                unknown_ranges=[("2021-01-01", "2021-11-01")])   # legacy data that stored "unknown" as 0
og.volume_known(bars, policy)                       # True where the volume is a real value (zero included)
```

See [docs/volume.md](docs/volume.md) and the full [reference](docs/reference.md).

## What it doesn't do

- **Market calendars:** a Forex weekend is a gap to `find_gaps`.
- **Outlier detection** beyond the opt-in `PRICE_JUMP`.
- **Filling, interpolating, or fixing bars.**

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
