"""
Feature engineering for BTC/USDT daily OHLCV data.

Reads raw data from data/raw/btc_daily.parquet, computes features and the
binary classification target (direction of 5-day forward return), and saves
the result to data/processed/btc_features.parquet.

Features implemented:
    - log_return_1d                     : daily log return of close price
    - momentum_{5,10,20}                : cumulative log return over window
    - rolling_vol_{10,20}               : rolling std of log returns
    - rsi_14                            : Relative Strength Index (ta package)
    - sma_{20,50}                       : simple moving averages of close
    - sma_20_over_50_ratio              : close-to-SMA and SMA crossover proxy
    - regime                            : manual market regime label for stability analysis

Target:
    - y = 1 if close[t+5] > close[t] else 0
    - Because y uses future information, the last 5 rows will have NaN y and
      are dropped at the end.

References:
    Jansen (2020), Ch. 4: Momentum & volatility as predictive features.
    Jansen (2020), Ch. 6: Avoiding data leakage when building features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

# --- config ---
INPUT_PATH = Path("data/raw/btc_daily.parquet")
OUTPUT_PATH = Path("data/processed/btc_features.parquet")

FORWARD_HORIZON = 5  # predict direction 5 days ahead
MOMENTUM_WINDOWS = (5, 10, 20)
VOL_WINDOWS = (10, 20)
RSI_WINDOW = 14
SMA_WINDOWS = (20, 50)


# -----------------------------------------------------------------------------
# Regime definition (Option C: stability-in-regimes analysis)
# -----------------------------------------------------------------------------
# Regimes are defined manually based on well-known BTC market phases.
# They are used ONLY for ex-post evaluation (split results by regime),
# never as model input — doing so would be label leakage.
REGIMES: list[tuple[str, str, str]] = [
    ("2019-01-01", "2020-05-31", "pre_2020"),       # pre-halving / pre-COVID
    ("2020-06-01", "2021-11-30", "bull_2020_2021"), # post-COVID rally to ATH
    ("2021-12-01", "2022-12-31", "bear_2022"),      # bear market, FTX collapse
    ("2023-01-01", "2023-12-31", "recovery_2023"),  # recovery / consolidation
    ("2024-01-01", "2099-01-01", "bull_2024_plus"), # post-ETF bull run
]


def assign_regime(index: pd.DatetimeIndex) -> pd.Series:
    """Assign a regime label to each date based on REGIMES intervals."""
    regime = pd.Series(index=index, dtype="object", name="regime")
    for start, end, label in REGIMES:
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        regime.loc[mask] = label
    return regime


# -----------------------------------------------------------------------------
# Individual feature groups
# -----------------------------------------------------------------------------
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log return of the close price."""
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_momentum(df: pd.DataFrame, windows: tuple[int, ...] = MOMENTUM_WINDOWS) -> pd.DataFrame:
    """
    Momentum = cumulative log return over the last `w` days.
    Implemented as rolling sum of daily log returns, which is equivalent to
    log(close[t] / close[t-w]) and more numerically stable.
    """
    for w in windows:
        df[f"momentum_{w}"] = df["log_return_1d"].rolling(window=w).sum()
    return df


def add_volatility(df: pd.DataFrame, windows: tuple[int, ...] = VOL_WINDOWS) -> pd.DataFrame:
    """Rolling standard deviation of daily log returns (realized volatility proxy)."""
    for w in windows:
        df[f"rolling_vol_{w}"] = df["log_return_1d"].rolling(window=w).std()
    return df


def add_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    """Relative Strength Index using the `ta` package."""
    rsi = RSIIndicator(close=df["close"], window=window, fillna=False)
    df[f"rsi_{window}"] = rsi.rsi()
    return df


def add_sma(df: pd.DataFrame, windows: tuple[int, ...] = SMA_WINDOWS) -> pd.DataFrame:
    """Simple moving averages and a normalized crossover-style ratio."""
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(window=w).mean()

    # Normalized crossover proxy: >1 means short SMA above long SMA (uptrend).
    # Using ratio instead of difference keeps the feature scale-free.
    short, long = min(SMA_WINDOWS), max(SMA_WINDOWS)
    df[f"sma_{short}_over_{long}_ratio"] = df[f"sma_{short}"] / df[f"sma_{long}"]
    return df


def add_target(df: pd.DataFrame, horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    """
    Binary target: 1 if close rises over the next `horizon` days, else 0.

    We use close.shift(-horizon) > close, which looks INTO THE FUTURE on purpose —
    this is the label. The model will never see raw future data; the label is only
    used as the `y` to predict. Rows with NaN target (last `horizon` rows) are
    dropped by caller.
    """
    future_close = df["close"].shift(-horizon)
    df["y"] = (future_close > df["close"]).astype("float")  # float so NaNs are allowed
    # Mark the forward-horizon rows explicitly as NaN (shift already does this,
    # but .astype(float) on NaN comparison returns 0.0 in some pandas versions)
    df.loc[future_close.isna(), "y"] = np.nan
    return df


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "log_return_1d",
    "momentum_5", "momentum_10", "momentum_20",
    "rolling_vol_10", "rolling_vol_20",
    "rsi_14",
    "sma_20", "sma_50", "sma_20_over_50_ratio",
]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations and return a cleaned frame."""
    df = raw.copy()
    df = add_returns(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = add_sma(df)
    df = add_target(df)
    df["regime"] = assign_regime(df.index)
    return df


def sanity_check(df: pd.DataFrame) -> None:
    """Hard asserts - fail loudly if anything is off."""
    # Target must be strictly binary after NaN drop.
    assert df["y"].isin([0.0, 1.0]).all(), "Target contains non-binary values"
    assert df["y"].notna().all(), "Target still has NaNs after dropna"

    # All features must be present and finite.
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"
        assert df[col].notna().all(), f"NaNs remain in feature {col}"
        assert np.isfinite(df[col]).all(), f"Inf values in feature {col}"

    # Regime column must cover every row.
    assert df["regime"].notna().all(), "Some rows have no regime assigned"

    # Index must be monotonic increasing (time order).
    assert df.index.is_monotonic_increasing, "Index is not sorted ascending"
    assert df.index.is_unique, "Duplicate timestamps in index"


def main() -> None:
    print(f"Reading raw data from {INPUT_PATH}...")
    raw = pd.read_parquet(INPUT_PATH)
    print(f"  {len(raw)} rows, date range {raw.index.min().date()} -> {raw.index.max().date()}")

    print("Building features...")
    df = build_features(raw)

    # Drop rows where either features OR target are undefined.
    # Feature NaNs come from rolling windows at the start.
    # Target NaNs come from the last FORWARD_HORIZON rows.
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["y"])
    after = len(df)
    print(f"  dropped {before - after} rows with NaN (warmup + forward horizon)")

    # Cast target back to int once we know there are no NaNs.
    df["y"] = df["y"].astype(int)

    print("Running sanity checks...")
    sanity_check(df)

    # Class balance — log for transparency (financial targets are rarely 50/50).
    pos_rate = df["y"].mean()
    print(f"  positive class rate (up-move): {pos_rate:.3f}")
    print(f"  regime distribution:")
    print(df["regime"].value_counts().sort_index().to_string().replace("\n", "\n    "))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
