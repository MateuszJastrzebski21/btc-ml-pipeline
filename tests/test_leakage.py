"""
Look-ahead and correctness tests. Run offline on a synthetic price path,
so they need no network access and no downloaded data.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from evaluation.walk_forward import walk_forward_evaluate
from features.engineer import FEATURE_COLUMNS, FORWARD_HORIZON, build_features
from models.persistence import get_model_1d, get_model_5d


@pytest.fixture
def raw() -> pd.DataFrame:
    """Geometric random walk, 600 daily bars, OHLCV-shaped."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=600, freq="D")
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1.0},
        index=idx,
    )


def test_features_do_not_use_future_bars(raw):
    """Features up to day t must be identical whether or not later bars exist."""
    cut = 400
    full = build_features(raw)[FEATURE_COLUMNS]
    truncated = build_features(raw.iloc[:cut])[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated)


def test_features_ignore_a_shock_in_the_future(raw):
    """A 10x price shock on day t+1 must not move any feature on days <= t."""
    t = 300
    shocked = raw.copy()
    shocked.iloc[t + 1:, shocked.columns.get_loc("close")] *= 10
    a = build_features(raw)[FEATURE_COLUMNS].iloc[: t + 1]
    b = build_features(shocked)[FEATURE_COLUMNS].iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)


def test_target_is_forward_direction(raw):
    df = build_features(raw)
    close = raw["close"]
    expected = (close.shift(-FORWARD_HORIZON) > close).astype(float)
    labelled = df["y"].notna()
    pd.testing.assert_series_equal(
        df.loc[labelled, "y"], expected[labelled], check_names=False
    )
    # the last FORWARD_HORIZON rows cannot be labelled
    assert df["y"].iloc[-FORWARD_HORIZON:].isna().all()


def test_regime_is_not_a_model_input():
    assert "regime" not in FEATURE_COLUMNS


def test_walk_forward_trains_strictly_on_the_past(raw):
    df = build_features(raw).dropna(subset=FEATURE_COLUMNS + ["y"])
    X, y = df[FEATURE_COLUMNS], df["y"].astype(int)
    res = walk_forward_evaluate(
        LogisticRegression(max_iter=1000), X, y, n_splits=5, verbose=False
    )
    folds = res.per_fold_metrics
    assert (folds["train_end"] < folds["test_start"]).all()
    # test windows are contiguous and non-overlapping
    assert (folds["test_start"].iloc[1:].values > folds["test_end"].iloc[:-1].values).all()
    # every prediction is out-of-fold and appears exactly once
    assert res.predictions.index.is_unique
    assert res.predictions.index.min() > folds["train_end"].iloc[0]


def test_persistence_baselines_follow_their_rule(raw):
    df = build_features(raw).dropna(subset=FEATURE_COLUMNS + ["y"])
    X, y = df[FEATURE_COLUMNS], df["y"].astype(int)
    for (_, model), col in ((get_model_1d(), "log_return_1d"), (get_model_5d(), "momentum_5")):
        pred = model.fit(X, y).predict(X)
        np.testing.assert_array_equal(pred, (X[col] > 0).astype(int).values)
