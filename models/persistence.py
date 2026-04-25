"""
Persistence (naive lag) baselines.

These are the "trivial benchmarks" that any ML model must beat to be
considered useful. The implementation follows directly the supervisor's
suggestion: "predict that tomorrow will be the same as today".

Two flavours are provided, addressing two reasonable interpretations:

    PersistenceClassifier1Day:
        Predicts that the direction of the NEXT 5-day move equals the
        direction of the LAST 1-day move.
            y_pred[t] = 1 if close[t] > close[t-1] else 0
        Literal interpretation of "tomorrow like today".

    PersistenceClassifier5Day (default):
        Predicts that the direction of the NEXT 5-day move equals the
        direction of the LAST 5-day move.
            y_pred[t] = 1 if close[t] > close[t-5] else 0
        Horizon-matched: persistence on the same scale as the target.

Why include these baselines:
    Jansen (Ch. 6, p. 154) recommends always benchmarking against a
    trivial naive predictor before declaring a model "useful". Markets
    exhibit short-term momentum and autocorrelation, so persistence
    often beats random-guess accuracy by a wide margin — sometimes more
    than the ML models themselves. A model that fails to beat persistence
    is, in practical terms, learning nothing.

No data leakage:
    For each prediction at time t, both flavours use only past close
    prices (close[t-1], close[t-5], ..., close[t]). The target y[t] uses
    close[t+5], which is in the future. There is zero overlap of
    information between predictor and target.

Implementation note:
    These baselines are deterministic rules — they do not "learn" from
    the training set. To fit cleanly into the existing
    walk_forward_evaluate infrastructure (which expects an sklearn-
    compatible classifier), they are wrapped in a class implementing
    .fit() (no-op except for setting classes_) and .predict_proba().
    Predicted probabilities are pseudo-probabilities (0.99 / 0.01).
    They are not calibrated; they exist only so that probability-based
    metrics (ROC-AUC, log-loss) remain comparable across all models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class _PersistenceBase(BaseEstimator, ClassifierMixin):
    """
    Shared logic for persistence baselines. Subclasses define which feature
    column carries the direction signal.

    The signal column must be a feature already in X (so that the same
    train/test split logic applies). For 1-day persistence we use
    log_return_1d; for 5-day persistence we use momentum_5 (cumulative
    log return over the past 5 days, mathematically equal in sign to
    close[t] - close[t-5]).
    """

    signal_column: str = "momentum_5"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_PersistenceBase":
        # Sklearn convention: classes_ must be set after fit.
        self.classes_ = np.array([0, 1])
        # Fallback: majority class if signal column happens to be missing.
        self._majority_class_ = int(round(float(np.asarray(y).mean())))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.signal_column in X.columns:
            signal = X[self.signal_column].values
            preds = (signal > 0).astype(int)
        else:
            preds = np.full(len(X), self._majority_class_, dtype=int)
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        # Pseudo-probabilities: 0.99 for the predicted class, 0.01 for the other.
        proba = np.zeros((len(preds), 2), dtype=float)
        proba[preds == 0, 0] = 0.99
        proba[preds == 0, 1] = 0.01
        proba[preds == 1, 0] = 0.01
        proba[preds == 1, 1] = 0.99
        return proba


class PersistenceClassifier1Day(_PersistenceBase):
    """
    Baseline: 'tomorrow looks like today'.

    Predicts the direction of the 5-day-ahead move based on the SIGN of the
    most recent 1-day return. Implements the supervisor's verbatim
    suggestion: "next day same as previous day".
    """

    def __init__(self) -> None:
        self.signal_column = "log_return_1d"


class PersistenceClassifier5Day(_PersistenceBase):
    """
    Baseline: 'next 5 days look like the previous 5 days'.

    Predicts the direction of the 5-day-ahead move based on the SIGN of the
    cumulative 5-day return. Horizon-matched persistence — the persistence
    rule operates on the same time scale as the prediction target, which is
    the methodologically tighter version of the same idea.
    """

    def __init__(self) -> None:
        self.signal_column = "momentum_5"


# -----------------------------------------------------------------------------
# Public API for the multi-model runner
# -----------------------------------------------------------------------------
def get_model_1d() -> tuple[str, BaseEstimator]:
    """1-day persistence (literal interpretation of supervisor's suggestion)."""
    return "Persistence1d", PersistenceClassifier1Day()


def get_model_5d() -> tuple[str, BaseEstimator]:
    """5-day persistence (horizon-matched version)."""
    return "Persistence5d", PersistenceClassifier5Day()


# Backward compatibility: previous code used `from models.persistence import get_model`.
def get_model() -> tuple[str, BaseEstimator]:
    """Default persistence baseline = 5-day, horizon-matched."""
    return get_model_5d()
