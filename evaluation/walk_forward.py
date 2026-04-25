"""
Walk-forward evaluation using sklearn's TimeSeriesSplit.

This module provides a single high-level function, `walk_forward_evaluate`,
that takes an sklearn-compatible estimator, runs it through an expanding-window
time-series cross-validation, and returns both per-fold metrics and raw
out-of-fold predictions (including predicted probabilities).

The returned predictions are a key asset for downstream analyses:
    - Probability calibration (see evaluation/calibration.py)
    - Threshold tuning and trading metrics (see evaluation/trading_metrics.py)
    - Stability-per-regime analysis (see evaluation/regime_analysis.py)

Data-leakage policy (per Jansen 2020, Ch. 6, pp. 168-170):
    - Any fitted preprocessing (e.g. StandardScaler) MUST be part of the
      sklearn Pipeline passed in, so that .fit() is called on training data
      only inside each fold.
    - The features used here are already past-only by construction
      (rolling windows, RSI, SMA — all use .shift() or rolling on past bars).
    - The target y uses future information but is only the label; it never
      enters the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    """
    Holds everything produced by a single walk-forward run.

    Attributes:
        model_name: Human-readable name for this run (e.g. "LogReg").
        per_fold_metrics: DataFrame with one row per fold, columns =
            [fold, n_train, n_test, accuracy, f1, roc_auc, log_loss,
             train_start, train_end, test_start, test_end].
        predictions: DataFrame indexed by timestamp with columns =
            [fold, y_true, y_pred, y_proba]. Contains out-of-fold predictions
            only — never training predictions. Can be concatenated across
            folds to cover (almost) the full history.
    """
    model_name: str
    per_fold_metrics: pd.DataFrame
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> pd.Series:
        """Mean and std of metrics across folds (for reporting in the thesis)."""
        metric_cols = ["accuracy", "f1", "roc_auc", "log_loss"]
        means = self.per_fold_metrics[metric_cols].mean()
        stds = self.per_fold_metrics[metric_cols].std()
        out = {}
        for m in metric_cols:
            out[f"{m}_mean"] = means[m]
            out[f"{m}_std"] = stds[m]
        return pd.Series(out, name=self.model_name)


# -----------------------------------------------------------------------------
# Core evaluator
# -----------------------------------------------------------------------------
def walk_forward_evaluate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int | None = None,
    model_name: str | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Run walk-forward cross-validation with TimeSeriesSplit.

    Args:
        estimator: Any sklearn-compatible classifier with fit / predict /
            predict_proba. Preprocessing (e.g. scaling) should be wrapped
            in a Pipeline so that fit is strictly per-fold.
        X: Feature DataFrame, chronologically sorted, no NaNs.
        y: Binary target Series aligned with X.
        n_splits: Number of time-series folds (Jansen uses 5 as a default).
        test_size: If given, each test fold has exactly this many rows
            (expanding train, fixed test). If None, sklearn's default is used.
        model_name: Label for plotting / logging. Defaults to class name.
        verbose: Print one line per fold.

    Returns:
        WalkForwardResult with per_fold_metrics and concatenated predictions.
    """
    assert len(X) == len(y), "X and y must have equal length"
    assert X.index.equals(y.index), "X and y must share index"
    assert X.index.is_monotonic_increasing, "Index must be sorted ascending"

    if model_name is None:
        model_name = type(estimator).__name__

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    fold_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fresh clone per fold — no state leaks between folds.
        model = clone(estimator)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Robustly get positive-class probabilities.
        # DummyClassifier(strategy="most_frequent") returns a single class
        # in some sklearn versions — handle that edge case.
        proba_raw = model.predict_proba(X_test)
        classes = getattr(model, "classes_", np.array([0, 1]))
        if proba_raw.shape[1] == 2:
            pos_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
            y_proba = proba_raw[:, pos_idx]
        else:
            # Only one class in training — constant probability.
            y_proba = np.full(len(X_test), float(classes[0]))

        metrics = _compute_metrics(y_test, y_pred, y_proba)

        row = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_start": X_train.index[0],
            "train_end": X_train.index[-1],
            "test_start": X_test.index[0],
            "test_end": X_test.index[-1],
            **metrics,
        }
        fold_rows.append(row)

        pred_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                },
                index=X_test.index,
            )
        )

        if verbose:
            print(
                f"  [{model_name}] fold {fold}/{n_splits} | "
                f"train {row['train_start'].date()}..{row['train_end'].date()} "
                f"({row['n_train']} rows) -> "
                f"test {row['test_start'].date()}..{row['test_end'].date()} "
                f"({row['n_test']} rows) | "
                f"acc={metrics['accuracy']:.3f} "
                f"f1={metrics['f1']:.3f} "
                f"auc={metrics['roc_auc']:.3f} "
                f"logloss={metrics['log_loss']:.3f}"
            )

    per_fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_frames).sort_index()

    return WalkForwardResult(
        model_name=model_name,
        per_fold_metrics=per_fold_metrics,
        predictions=predictions,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute the four headline metrics used throughout the thesis.

    ROC-AUC and log-loss require probabilities and break when only one class
    is present in y_true. We degrade gracefully by returning NaN in those
    edge cases instead of crashing — important for very small test folds.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC undefined when y_true has only one class.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan

    # Log-loss undefined if probas hit exactly 0 or 1 for wrong class; clip.
    try:
        proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
        metrics["log_loss"] = log_loss(y_true, proba_clipped, labels=[0, 1])
    except ValueError:
        metrics["log_loss"] = np.nan

    return metrics


# -----------------------------------------------------------------------------
# Smoke test entry point
# -----------------------------------------------------------------------------
def _smoke_test() -> None:
    """
    Sanity check: run walk-forward on a DummyClassifier to verify the pipeline
    end-to-end before plugging in real models. Matches Jansen's approach of
    always having a trivial baseline to compare against.
    """
    from pathlib import Path

    from sklearn.dummy import DummyClassifier

    features_path = Path("data/processed/btc_features.parquet")
    assert features_path.exists(), (
        f"Features file not found: {features_path}\n"
        "Run `python -m features.engineer` first."
    )

    df = pd.read_parquet(features_path)

    feature_cols = [
        "log_return_1d",
        "momentum_5", "momentum_10", "momentum_20",
        "rolling_vol_10", "rolling_vol_20",
        "rsi_14",
        "sma_20", "sma_50", "sma_20_over_50_ratio",
    ]
    X = df[feature_cols]
    y = df["y"]

    print(f"Loaded features: {X.shape}, target positive rate: {y.mean():.3f}")
    print("Running walk-forward smoke test with DummyClassifier...")

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    result = walk_forward_evaluate(dummy, X, y, n_splits=5, model_name="Dummy")

    print("\nPer-fold metrics:")
    print(result.per_fold_metrics.to_string(index=False))
    print("\nSummary:")
    print(result.summary().to_string())
    print(f"\nPredictions frame shape: {result.predictions.shape}")


if __name__ == "__main__":
    _smoke_test()
