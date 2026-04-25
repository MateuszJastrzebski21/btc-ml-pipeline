"""
Run all models end-to-end and log everything to MLflow.

Models include two naive persistence baselines (1-day and 5-day),
LogReg, RandomForest and XGBoost.

The script forces a file-based tracking backend so that any leftover
SQLite configuration cannot redirect runs into a database that the UI
later fails to discover.

In addition to the per-model artifacts, this script writes a master
`summary_all_models.csv` that the plotting code (evaluation/plots.py)
expects as input.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from evaluation.walk_forward import WalkForwardResult, walk_forward_evaluate
from models.baseline import get_model as get_logreg
from models.persistence import get_model_1d, get_model_5d
from models.random_forest import get_model as get_random_forest
from models.xgboost_model import get_model as get_xgboost

# --- config ---
FEATURES_PATH = Path("data/processed/btc_features.parquet")
RESULTS_DIR = Path("evaluation/results")
EXPERIMENT_NAME = "btc_ml_pipeline"

FEATURE_COLUMNS = [
    "log_return_1d",
    "momentum_5", "momentum_10", "momentum_20",
    "rolling_vol_10", "rolling_vol_20",
    "rsi_14",
    "sma_20", "sma_50", "sma_20_over_50_ratio",
]

N_SPLITS = 5


def _extract_estimator_params(estimator) -> dict:
    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        clf = estimator.named_steps["clf"]
    else:
        clf = estimator

    raw = clf.get_params(deep=False)
    clean = {}
    for k, v in raw.items():
        if v is None:
            clean[k] = "None"
        elif isinstance(v, (int, float, bool, str)):
            clean[k] = v
        else:
            clean[k] = str(v)[:250]
    return clean


def _log_fold_run(experiment_id, parent_run_id, model_name, fold_row, params):
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{model_name}_fold_{int(fold_row['fold'])}",
        nested=True,
    ):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("run_type", "fold")
        mlflow.set_tag("fold", int(fold_row["fold"]))
        mlflow.set_tag("parent_run_id", parent_run_id)

        mlflow.log_params(params)
        mlflow.log_params(
            {
                "n_train": int(fold_row["n_train"]),
                "n_test": int(fold_row["n_test"]),
                "train_start": str(fold_row["train_start"].date()),
                "train_end": str(fold_row["train_end"].date()),
                "test_start": str(fold_row["test_start"].date()),
                "test_end": str(fold_row["test_end"].date()),
            }
        )

        for metric in ("accuracy", "f1", "roc_auc", "log_loss"):
            val = fold_row[metric]
            if pd.notna(val):
                mlflow.log_metric(metric, float(val))


def _log_parent_run(experiment_id, model_name, result, regime, params):
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{model_name}_wf_summary",
    ) as parent:
        parent_run_id = parent.info.run_id

        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("run_type", "summary")
        mlflow.set_tag("n_splits", N_SPLITS)

        mlflow.log_params(params)

        summary = result.summary()
        for metric_name, value in summary.items():
            if pd.notna(value):
                mlflow.log_metric(metric_name, float(value))

        for _, row in result.per_fold_metrics.iterrows():
            step = int(row["fold"])
            for m in ("accuracy", "f1", "roc_auc", "log_loss"):
                if pd.notna(row[m]):
                    mlflow.log_metric(f"{m}_per_fold", float(row[m]), step=step)

        tmp_dir = RESULTS_DIR
        tmp_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = tmp_dir / f"{model_name}_per_fold_metrics.csv"
        result.per_fold_metrics.to_csv(metrics_path, index=False)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        predictions = result.predictions.join(regime)
        preds_path = tmp_dir / f"{model_name}_predictions.parquet"
        predictions.to_parquet(preds_path)
        mlflow.log_artifact(str(preds_path), artifact_path="predictions")

        schema_path = tmp_dir / "feature_schema.txt"
        schema_path.write_text("\n".join(FEATURE_COLUMNS))
        mlflow.log_artifact(str(schema_path), artifact_path="schema")

    return parent_run_id


def run(with_figures: bool = True) -> None:
    # Force file-based tracking. Whatever sets the global default to sqlite
    # (VS Code extension, leftover config, anything) - we override it here.
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
    regime = df["regime"]

    print(f"Loaded dataset: {X.shape}, positive rate: {y.mean():.3f}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

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

        params = _extract_estimator_params(estimator)

        result = walk_forward_evaluate(
            estimator=estimator,
            X=X,
            y=y,
            n_splits=N_SPLITS,
            model_name=name,
        )

        parent_run_id = _log_parent_run(
            experiment_id=experiment_id,
            model_name=name,
            result=result,
            regime=regime,
            params=params,
        )
        print(f"  parent run: {parent_run_id}")

        for _, fold_row in result.per_fold_metrics.iterrows():
            _log_fold_run(
                experiment_id=experiment_id,
                parent_run_id=parent_run_id,
                model_name=name,
                fold_row=fold_row,
                params=params,
            )

        print(f"  logged {len(result.per_fold_metrics)} fold runs")
        summaries.append(result.summary())

    # ------------------------------------------------------------------
    # Master summary CSV — required by evaluation/plots.py.
    # Without this, plot_summary_bars() and the comparison notebook
    # would crash with FileNotFoundError.
    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_DIR / "summary_all_models.csv"
    summary_df.to_csv(summary_path)
    print(f"\nsaved -> {summary_path}")

    # ------------------------------------------------------------------
    if with_figures:
        from evaluation.plots import main as regenerate_plots

        print(f"\n{'=' * 72}")
        print("Regenerating comparison figures...")
        print("=" * 72)
        regenerate_plots()

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="comparison_figures",
        ):
            mlflow.set_tag("run_type", "figures")
            figures_dir = Path("evaluation/figures")
            for png in sorted(figures_dir.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="figures")
                print(f"  attached -> {png.name}")

    print(f"\n{'=' * 72}")
    print("Done. Start MLflow UI with:")
    print(f"    mlflow ui --backend-store-uri \"{tracking_uri}\"")
    print(f"Then open: http://127.0.0.1:5000 and select experiment '{EXPERIMENT_NAME}'")
    print("=" * 72)


if __name__ == "__main__":
    run()
