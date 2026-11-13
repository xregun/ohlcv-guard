"""Run: python examples/quickstart.py  (generated data only)."""

import numpy as np
import pandas as pd

import ohlcv_guard as og

rng = np.random.default_rng(7)
ts = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
close = 100 + np.cumsum(rng.normal(0, 0.5, len(ts)))
open_ = np.concatenate([[100.0], close[:-1]])
bars = pd.DataFrame({"ts": ts, "open": open_, "close": close, "volume": rng.uniform(10, 50, len(ts)),
                     "high": np.maximum(open_, close) + 0.2, "low": np.minimum(open_, close) - 0.2})

# 1. Clean data: canonical bars and an empty report.
result = og.normalize(bars, "1h")
print("ok:", result.report.ok, "| rows:", len(result.bars), "| warnings:", result.report.counts())

# 2. Something to look at: the market freezes after 20 hours (one price, no volume).
frozen = bars.copy()
frozen.loc[20:, ["open", "high", "low", "close"]] = frozen.loc[19, "close"]
frozen.loc[20:, "volume"] = 0.0
report = og.validate(frozen, "1h", checks=og.Checks(frozen_run=12))
print(report.get("FROZEN_RUN").message)

# 3. Bad data: refused, with the reason.
broken = bars.copy()
broken.loc[5, "high"] = broken.loc[5, "low"] - 1
try:
    og.normalize(broken, "1h")
except og.InvalidBarsError as exc:
    print("refused:", exc, [i.code for i in exc.issues])

# 4. Gaps and resampling.
holes = result.bars.drop(index=[10, 11, 30]).reset_index(drop=True)
print("gaps:", [(str(g.start), g.missing) for g in og.find_gaps(holes, "1h")])
print(og.resample(holes, "1h", "4h").head(4))
