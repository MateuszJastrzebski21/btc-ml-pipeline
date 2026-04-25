"""
XGBoost gradient-boosted classifier.

XGBoost is the de-facto standard advanced baseline for tabular problems in
quantitative finance. Unlike random forests (which average many independent
trees), gradient boosting builds trees sequentially — each new tree
corrects the errors of the previous ensemble. This sequential correction
makes boosting more powerful but also more prone to overfitting, so strong
regularization is essential.

Design notes:
    - Shallow trees (max_depth=4) and a slow learning rate (0.05) are the
      canonical "safe" boosting configuration for noisy data.
    - subsample=0.8 and colsample_bytree=0.8 add stochastic regularization
      by training each tree on 80% of rows / columns.
    - reg_alpha (L1) and reg_lambda (L2) penalize tree complexity in
      addition to the depth cap.
    - eval_metric is set to logloss so that XGBoost's internal monitoring
      matches the metric we report in the walk-forward results.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from xgboost import XGBClassifier


def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> XGBClassifier:
    """Build an XGBoost classifier with conservative regularization."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        # Modern XGBoost (>=2.0) auto-detects objective; being explicit helps clarity.
        objective="binary:logistic",
        eval_metric="logloss",
        # Silence deprecation warnings in newer versions.
        tree_method="hist",
    )


def get_model() -> tuple[str, BaseEstimator]:
    """Return (name, estimator) pair used by the multi-model runner."""
    return "XGBoost", build_xgboost()
