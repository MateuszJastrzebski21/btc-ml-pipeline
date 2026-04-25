"""
Random Forest classifier.

Non-linear baseline for comparison against the linear logistic regression.
Random forests are a natural next step when you suspect that interactions
between features (e.g. "high momentum AND high volatility") carry signal
that a linear model cannot capture.

Design notes:
    - No StandardScaler: tree-based models are scale-invariant by construction
      (they split on feature thresholds, not on distances), so scaling is
      pure overhead.
    - max_depth=5 is deliberately SHALLOW. Financial time series carry a very
      low signal-to-noise ratio; deep trees will happily memorize noise
      (Jansen Ch. 6, bias-variance discussion, pp. 164-166). A shallow forest
      with many trees trades variance for bias — the right side of that
      trade-off for our use case.
    - min_samples_leaf=20 further blocks the forest from creating leaves
      driven by single observations.
    - n_estimators=300 is on the higher side; random forests rarely overfit
      by adding more trees, only by growing deeper trees.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int = 5,
    min_samples_leaf: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Build a regularized Random Forest classifier suitable for noisy
    financial time series.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_model() -> tuple[str, BaseEstimator]:
    """Return (name, estimator) pair used by the multi-model runner."""
    return "RandomForest", build_random_forest()
