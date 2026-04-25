"""
Three-class target for the multi-class extension (thesis section 3.10).

The binary target used in the main experiment (1 = up, 0 = not up) does not
distinguish between weak and strong moves. The supervisor suggested an
extension along the lines of "strong down / weak / strong up".

This module re-uses the existing OHLCV data and the same feature columns,
adding only a new categorical target:

    y3 = -1  if 5d log return < -threshold   ("strong down")
    y3 =  0  if -threshold <= ret <= +threshold  ("weak / neutral")
    y3 = +1  if 5d log return > +threshold   ("strong up")

Where threshold equals one rolling standard deviation of 5-day log returns
estimated on a 252-day (~1 year) window. Using a rolling local SD instead
of a global constant keeps the class definitions adaptive to market
volatility — strong moves in 2022 are not the same magnitude as in 2024.

Output:
    data/processed/btc_features_3class.parquet
    Same as btc_features.parquet plus 'y3' column. The original binary
    'y' is preserved so downstream scripts can pick whichever target
    they need.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/processed/btc_features.parquet")
OUTPUT_PATH = Path("data/processed/btc_features_3class.parquet")

FORWARD_HORIZON = 5
ROLLING_VOL_WINDOW = 252  # ~1 trading year

# How many SD of 5-day log returns delimit the "strong" classes.
# 0.5 produces roughly balanced classes on BTC daily; smaller values
# make the neutral class shrink, larger values shrink the tails.
THRESHOLD_K_SD = 0.5


def main() -> None:
    assert INPUT_PATH.exists(), (
        f"{INPUT_PATH} not found. Run `python -m features.engineer` first."
    )
    df = pd.read_parquet(INPUT_PATH)

    # 5-day log return — the same horizon as the binary target.
    fwd_log_ret = np.log(df["close"].shift(-FORWARD_HORIZON) / df["close"])

    # Rolling SD on a backward-looking window. This is past-only at time t
    # and therefore safe to use as part of a target definition (no leakage
    # into the predictors; the target itself is constructed from future
    # close prices, which is allowed).
    rolling_sd = (
        fwd_log_ret.rolling(window=ROLLING_VOL_WINDOW, min_periods=60).std()
    )
    threshold = THRESHOLD_K_SD * rolling_sd

    y3 = pd.Series(np.nan, index=df.index, dtype="float64", name="y3")
    y3.loc[fwd_log_ret < -threshold] = -1.0
    y3.loc[(fwd_log_ret >= -threshold) & (fwd_log_ret <= threshold)] = 0.0
    y3.loc[fwd_log_ret > threshold] = 1.0

    df = df.copy()
    df["y3"] = y3
    df["y3_threshold"] = threshold  # logged for transparency / sanity checks

    # Drop rows where the new target is undefined (warm-up + last 5 rows).
    before = len(df)
    df = df.dropna(subset=["y3"])
    df["y3"] = df["y3"].astype(int)
    after = len(df)
    print(f"Dropped {before - after} rows with undefined y3 (warm-up + horizon).")

    # Sanity: target must take exactly the three values.
    unique = sorted(df["y3"].unique().tolist())
    assert unique == [-1, 0, 1], f"Unexpected y3 values: {unique}"

    print("\nClass distribution:")
    print(df["y3"].value_counts(normalize=True).rename({-1: "strong_down",
                                                          0: "weak_neutral",
                                                          1: "strong_up"})
                                                .round(3).to_string())
    print("\nClass distribution by regime:")
    pivot = df.groupby(["regime", "y3"]).size().unstack(fill_value=0)
    pivot.columns = ["strong_down", "weak_neutral", "strong_up"]
    print(pivot.to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"\nsaved -> {OUTPUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
