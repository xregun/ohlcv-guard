# Volume semantics

A volume column can mean three different things. ohlcv-guard keeps them apart.

| Meaning | Example | Representation |
|---|---|---|
| **Genuine zero**: the market was open, nothing traded | a quiet minute on an exchange | `0`, always allowed |
| **Unknown**: there was volume, the source doesn't say how much | a vendor's months without tick data | `NaN` with `unknown="nan"`; or legacy zeros inside `unknown_ranges` |
| **Not applicable**: the series has no volume | an index price, a mark price | `VolumePolicy(kind="none")` |

## Kinds

- `traded`: an exchange's traded volume (base units or contracts).
- `tick`: a count of price changes, as Forex and CFD sources publish. It isn't traded volume.
- `none`: no volume. The column may be absent; when present, it is ignored and not carried into the result.

## Unknown volume

- `forbid` (default): `NaN` volume is an error (`BAD_VOLUME`).
- `nan`: `NaN` is allowed and means unknown. Negative or infinite volume stays an error.
- `zero_in_ranges`: for data that already stores unknown volume as `0`. Zeros inside the declared `[start, end)`
  ranges are treated as unknown. Nothing is rewritten: the stored zeros stay zeros, and `volume_known` tells them
  apart.

## Zero volume

- `allow` (default): a zero is a genuine value.
- `warn_if_price_moves`: report `VOLUME_ZERO_WITH_PRICE_MOVE` for known zeros on bars whose high is above their low.
  Zeros inside unknown ranges aren't reported.

## Frozen markets

`FROZEN_RUN` uses the volume too: a run of bars with one price counts only if the volume is a known zero (for
`kind="none"`, on price alone). Unknown volume never makes a bar frozen.
