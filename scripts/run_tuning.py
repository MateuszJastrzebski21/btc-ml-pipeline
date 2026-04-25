"""
Small-scale hyperparameter tuning with MLflow logging.

Per Jansen (Ch. 6, pp. 170-172), the tuning philosophy for noisy financial
time series is "small and sensible": a narrow grid over a few meaningful
hyperparameters, evaluated through the same walk-forward CV as the main
experiment. The goal is NOT to squeeze out the best possible metric — it
is to understand how sensitive each model is to its knobs.

Design:
    - Grids are deliberately small (3-9 configurations per model).
    - Each configuration runs through the full 5-fold walk-forward.
    - Everything lands in a separate MLflow experiment: btc_ml_tuning.
    - The best configuration per model (by mean ROC-AUC) is printed at the
      end for manual inspection.

Choice of metric:
    ROC-AUC is preferred over accuracy/F1 for model selection because:
        - it is insensitive to the 54% class imbalance in the target,
        - it uses the full probability score, not a thresholded prediction,
        - it is what Jansen recommends for ranking classifiers (Ch. 6, p. 159).

Expected wall time:
    3 grids x ~5-9 configs x 5 folds x (fast model) ~ 3-8 minutes.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from evaluation.walk_forward import walk_forward_evaluate
from models.baseline import build_logreg_pipeline

# --- config ---
FEATURES_PATH = Path("data/processed/btc_features.parquet")
EXPERIMENT_NAME = "btc_ml_tuning"

FEATURE_COLUMNS = [
    "log_return_1d",
    "momentum_5", "momentum_10", "momentum_20",
    "rolling_vol_10", "rolling_vol_20",
    "rsi_14",
    "sma_20", "sma_50", "sma_20_over_50_ratio",
]

N_SPLITS = 5

# --- grids ---
# LogReg: regularization strength. Smaller C = stronger penalty.
LOGREG_GRID = {"C": [0.1, 1.0, 10.0]}

# RandomForest: depth and leaf size (the two knobs that really control
# bias-variance for trees on noisy data; n_estimators=300 is fixed).
RF_GRID = {
    "max_depth": [3, 5, 8],
    "min_samples_leaf": [10, 20, 50],
}

# XGBoost: depth x learning rate (the canonical boosting trade-off).
XGB_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _grid_configs(grid: dict) -> list[dict]:
    """Cartesian product of a parameter grid as a list of dicts."""
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in product(*[grid[k] for k in keys])]


def _evaluate(estimator, name: str, X, y) -> dict:
    """Run walk-forward and return summary metrics dict."""
    result = walk_forward_evaluate(
        estimator=estimator,
        X=X,
        y=y,
        n_splits=N_SPLITS,
        model_name=name,
        verbose=False,
    )
    return {
        "accuracy_mean": result.per_fold_metrics["accuracy"].mean(),
        "f1_mean": result.per_fold_metrics["f1"].mean(),
        "roc_auc_mean": result.per_fold_metrics["roc_auc"].mean(),
        "log_loss_mean": result.per_fold_metrics["log_loss"].mean(),
        "roc_auc_std": result.per_fold_metrics["roc_auc"].std(),
    }


def _tune_logreg(experiment_id: str, X, y) -> list[dict]:
    """Grid search over LogReg regularization."""
    records = []
    for config in _grid_configs(LOGREG_GRID):
        run_name = f"LogReg_C{config['C']}"
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.set_tag("model_type", "LogReg")
            mlflow.set_tag("run_type", "tuning")
            mlflow.log_params(config)

            estimator = build_logreg_pipeline(**config)
            metrics = _evaluate(estimator, "LogReg", X, y)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            records.append({"model": "LogReg", **config, **metrics})
            print(
                f"  LogReg C={config['C']:<6} -> "
                f"AUC={metrics['roc_auc_mean']:.4f} "
                f"(±{metrics['roc_auc_std']:.4f}) "
                f"F1={metrics['f1_mean']:.3f}"
            )
    return records


def _tune_random_forest(experiment_id: str, X, y) -> list[dict]:
    """Grid search over RF depth and leaf size."""
    records = []
    for config in _grid_configs(RF_GRID):
        run_name = (
            f"RF_d{config['max_depth']}_leaf{config['min_samples_leaf']}"
        )
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("run_type", "tuning")
            mlflow.log_params(config)

            estimator = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                **config,
            )
            metrics = _evaluate(estimator, "RandomForest", X, y)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            records.append({"model": "RandomForest", **config, **metrics})
            print(
                f"  RF depth={config['max_depth']} "
                f"leaf={config['min_samples_leaf']:<3} -> "
                f"AUC={metrics['roc_auc_mean']:.4f} "
                f"(±{metrics['roc_auc_std']:.4f}) "
                f"F1={metrics['f1_mean']:.3f}"
            )
    return records


def _tune_xgboost(experiment_id: str, X, y) -> list[dict]:
    """Grid search over XGBoost depth and learning rate."""
    records = []
    for config in _grid_configs(XGB_GRID):
        run_name = (
            f"XGB_d{config['max_depth']}_lr{config['learning_rate']}"
        )
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("run_type", "tuning")
            mlflow.log_params(config)

            estimator = XGBClassifier(
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                **config,
            )
            metrics = _evaluate(estimator, "XGBoost", X, y)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            records.append({"model": "XGBoost", **config, **metrics})
            print(
                f"  XGB depth={config['max_depth']} "
                f"lr={config['learning_rate']:<5} -> "
                f"AUC={metrics['roc_auc_mean']:.4f} "
                f"(±{metrics['roc_auc_std']:.4f}) "
                f"F1={metrics['f1_mean']:.3f}"
            )
    return records


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # Force file-based tracking — see scripts/run_with_mlflow.py for rationale.
    tracking_uri = f"file:///{Path('mlruns').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
    
    assert FEATURES_PATH.exists(), (
        f"Features file not found: {FEATURES_PATH}\n"
        "Run `python -m features.engineer` first."
    )

    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["y"]

    print(f"Loaded dataset: {X.shape}, positive rate: {y.mean():.3f}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    all_records: list[dict] = []

    print(f"\n{'=' * 72}")
    print("Tuning LogReg (regularization strength)")
    print("=" * 72)
    all_records.extend(_tune_logreg(experiment_id, X, y))

    print(f"\n{'=' * 72}")
    print("Tuning RandomForest (max_depth x min_samples_leaf)")
    print("=" * 72)
    all_records.extend(_tune_random_forest(experiment_id, X, y))

    print(f"\n{'=' * 72}")
    print("Tuning XGBoost (max_depth x learning_rate)")
    print("=" * 72)
    all_records.extend(_tune_xgboost(experiment_id, X, y))

    # Save tuning summary to disk — this table goes into section 3.6 of the thesis.
    tuning_df = pd.DataFrame(all_records)
    results_path = Path("evaluation/results/tuning_summary.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(results_path, index=False)

    # Pick the best config per model by mean ROC-AUC.
    print(f"\n{'=' * 72}")
    print("BEST CONFIGURATION PER MODEL (by mean ROC-AUC)")
    print("=" * 72)
    for model in ("LogReg", "RandomForest", "XGBoost"):
        sub = tuning_df[tuning_df["model"] == model]
        if sub.empty:
            continue
        best = sub.loc[sub["roc_auc_mean"].idxmax()]
        param_cols = [
            c for c in best.index
            if c not in {"model", "accuracy_mean", "f1_mean",
                         "roc_auc_mean", "log_loss_mean", "roc_auc_std"}
        ]
        params_str = ", ".join(f"{k}={best[k]}" for k in param_cols)
        print(
            f"  {model:<14} {params_str} -> "
            f"AUC={best['roc_auc_mean']:.4f} "
            f"(±{best['roc_auc_std']:.4f})"
        )

    print(f"\nSaved tuning summary -> {results_path}")
    print("\nStart MLflow UI with:")
    print("    mlflow ui")
    print(f"Then select experiment: '{EXPERIMENT_NAME}'")


if __name__ == "__main__":
    main()
