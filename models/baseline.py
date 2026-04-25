"""
Logistic Regression baseline model.

This is the "point of reference" model that all other models must beat.
Jansen (Ch. 7, pp. 212-219) uses logistic regression as the canonical
baseline for predicting the direction of price movements — we follow the
same convention.

Design notes:
    - Wrapped in an sklearn Pipeline with StandardScaler. This is NOT a
      cosmetic choice: logistic regression optimizes a gradient-based
      objective and requires scaled features to converge reliably.
    - The Pipeline is crucial for walk-forward: the scaler's fit()
      runs on training data INSIDE each fold, so no scaling statistics
      leak from future into past (see Jansen Ch. 6, pp. 168-170).
    - L2 regularization (the sklearn default) guards against overfitting
      on correlated features such as momentum_5 / momentum_10 / momentum_20.
      In sklearn >= 1.8 the `penalty` argument is deprecated — L2 is the
      default behaviour, controlled implicitly via the C parameter.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logreg_pipeline(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a Logistic Regression pipeline with standardization.

    Args:
        C: Inverse of regularization strength. Smaller C = stronger L2 penalty.
        max_iter: Maximum iterations for the lbfgs solver. 1000 is generous
            for a 10-feature problem and avoids ConvergenceWarning noise.
        random_state: Seed for reproducibility.

    Returns:
        An sklearn Pipeline that behaves like a single classifier:
        .fit(X, y) scales on train only, .predict_proba(X) scales using
        the train-fitted statistics.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs",
                    # penalty defaults to "l2" in sklearn 1.8+ — passing it
                    # explicitly triggers a deprecation FutureWarning.
                ),
            ),
        ]
    )


def get_model() -> tuple[str, BaseEstimator]:
    """Return (name, estimator) pair used by the multi-model runner."""
    return "LogReg", build_logreg_pipeline()
