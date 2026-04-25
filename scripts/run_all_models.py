"""
Run all models end-to-end and persist results.

Models, in order from weakest baseline to strongest:
    1. Persistence1d  - "tomorrow same as today" (supervisor's suggestion).
    2. Persistence5d  - horizon-matched persistence (5d back -> 5d forward).
    3. LogReg         - linear baseline.
    4. RandomForest   - non-linear ensemble.
    5. XGBoost        - gradient-boosted trees.

Outputs (under evaluation/results/):
    - {model}_per_fold_metrics.csv
    - {model}_predictions.parquet
    - summary_all_models.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.walk_forward import WalkForwardResult, walk_forward_evaluate
from models.baseline import get_model as get_logreg
from models.persistence import get_model_1d, get_model_5d
from models.random_forest import get_model as get_random_forest
from models.xgboost_model import get_model as get_xgboost

# --- config ---
FEATURES_PATH = Path("data/processed/btc_features.parquet")
RESULTS_DIR = Path("evaluation/results")

FEATURE_COLUMNS = [
    "log_return_1d",
    "momentum_5", "momentum_10", "momentum_20",
    "rolling_vol_10", "rolling_vol_20",
    "rsi_14",
    "sma_20", "sma_50", "sma_20_over_50_ratio",
]

N_SPLITS = 5


def _load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    assert FEATURES_PATH.exists(), (
        f"Features file not found: {FEATURES_PATH}\n"
        "Run `python -m features.engineer` first."
    )
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["y"]
    regime = df["regime"]
    print(f"Loaded dataset: {X.shape}, positive rate: {y.mean():.3f}")
    print(f"Date range: {X.index.min().date()} -> {X.index.max().date()}")
    return X, y, regime


def _save_result(result: WalkForwardResult, regime: pd.Series) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / f"{result.model_name}_per_fold_metrics.csv"
    result.per_fold_metrics.to_csv(metrics_path, index=False)
    print(f"  saved -> {metrics_path}")

    preds = result.predictions.join(regime)
    preds_path = RESULTS_DIR / f"{result.model_name}_predictions.parquet"
    preds.to_parquet(preds_path)
    print(f"  saved -> {preds_path}")


def main() -> None:
    X, y, regime = _load_data()

    models = [
        get_model_1d(),
        get_model_5d(),
        get_logreg(),
        get_random_forest(),
        get_xgboost(),
    ]

    summaries: list[pd.Series] = []

    for name, estimator in models:
        print(f"\n{'=' * 72}")
        print(f"Running walk-forward for: {name}")
        print("=" * 72)

        result = walk_forward_evaluate(
            estimator=estimator,
            X=X,
            y=y,
            n_splits=N_SPLITS,
            model_name=name,
        )

        _save_result(result, regime)
        summaries.append(result.summary())

    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "summary_all_models.csv"
    summary_df.to_csv(summary_path)

    print(f"\n{'=' * 72}")
    print("SUMMARY - all models, mean +/- std across folds")
    print("=" * 72)
    metric_order = []
    for metric in ("accuracy", "f1", "roc_auc", "log_loss"):
        metric_order.extend([f"{metric}_mean", f"{metric}_std"])
    print(summary_df[metric_order].round(4).to_string())
    print(f"\nsaved -> {summary_path}")


if __name__ == "__main__":
    main()
