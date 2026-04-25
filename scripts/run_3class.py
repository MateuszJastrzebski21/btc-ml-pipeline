"""
Three-class classification experiment (v2 — XGBoost-compatible).

Same idea as v1 but with two fixes:

    1. XGBoost 3.x requires class labels in {0, 1, 2}, not {-1, 0, 1}.
       We encode labels 0/1/2 internally and decode for confusion
       matrices and reporting only.

    2. Probability columns are renormalised to sum to 1 after the
       per-model column reordering. This is a no-op for well-behaved
       models but silences the sklearn warning when columns have been
       moved around.

Outputs:
    evaluation/results/3class_per_fold_metrics.csv
    evaluation/results/3class_summary.csv
    evaluation/figures/figure_3class_confusion_matrices.png
    MLflow runs in experiment 'btc_ml_pipeline_3class'
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- config ---
FEATURES_PATH = Path("data/processed/btc_features_3class.parquet")
RESULTS_DIR = Path("evaluation/results")
FIGURES_DIR = Path("evaluation/figures")
EXPERIMENT_NAME = "btc_ml_pipeline_3class"

FEATURE_COLUMNS = [
    "log_return_1d",
    "momentum_5", "momentum_10", "momentum_20",
    "rolling_vol_10", "rolling_vol_20",
    "rsi_14",
    "sma_20", "sma_50", "sma_20_over_50_ratio",
]

# External (human-readable) labels and internal (XGBoost-friendly) encoding.
EXTERNAL_CLASSES = [-1, 0, 1]
INTERNAL_CLASSES = [0, 1, 2]
CLASS_NAMES = ["strong_down", "weak_neutral", "strong_up"]
EXT_TO_INT = dict(zip(EXTERNAL_CLASSES, INTERNAL_CLASSES))
INT_TO_EXT = dict(zip(INTERNAL_CLASSES, EXTERNAL_CLASSES))

N_SPLITS = 5


# -----------------------------------------------------------------------------
def _build_models() -> list[tuple[str, object]]:
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=2000, random_state=42, solver="lbfgs",
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", tree_method="hist",
    )

    return [
        ("LogReg_3class", logreg),
        ("RandomForest_3class", rf),
        ("XGBoost_3class", xgb),
    ]


# -----------------------------------------------------------------------------
def _reorder_proba(proba_raw: np.ndarray, model_classes: np.ndarray,
                   target_order: list[int]) -> np.ndarray:
    """
    Reorder predict_proba columns so they correspond to target_order in that order.
    Then renormalise to sum to 1 along axis=1 (silences sklearn warning even if
    upstream probabilities were already exact).
    """
    target_arr = np.asarray(target_order)
    if list(model_classes) == list(target_arr):
        out = proba_raw
    else:
        idx = [list(model_classes).index(c) for c in target_arr]
        out = proba_raw[:, idx]

    # Defensive renormalisation
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def _multiclass_metrics(y_true_int: np.ndarray, y_pred_int: np.ndarray,
                        y_proba_int: np.ndarray) -> dict:
    """All inputs use INTERNAL labels {0, 1, 2}."""
    out = {
        "accuracy": accuracy_score(y_true_int, y_pred_int),
        "f1_macro": f1_score(y_true_int, y_pred_int, average="macro", zero_division=0),
    }
    try:
        out["roc_auc_ovr"] = roc_auc_score(
            y_true_int, y_proba_int, multi_class="ovr", average="weighted",
            labels=INTERNAL_CLASSES,
        )
    except (ValueError, IndexError):
        out["roc_auc_ovr"] = float("nan")

    try:
        proba_clipped = np.clip(y_proba_int, 1e-7, 1 - 1e-7)
        # Renormalise after clipping
        proba_clipped = proba_clipped / proba_clipped.sum(axis=1, keepdims=True)
        out["log_loss"] = log_loss(y_true_int, proba_clipped, labels=INTERNAL_CLASSES)
    except ValueError:
        out["log_loss"] = float("nan")

    return out


# -----------------------------------------------------------------------------
def main() -> None:
    tracking_uri = f"file:///{Path('mlruns').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    assert FEATURES_PATH.exists(), (
        f"{FEATURES_PATH} not found. Run `python -m features.engineer_3class` first."
    )

    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLUMNS]

    # Encode target to internal {0, 1, 2} for fitting.
    y_external = df["y3"]
    y_internal = y_external.map(EXT_TO_INT).astype(int)
    print(f"Loaded dataset: {X.shape}")
    print(
        f"External class distribution: "
        f"{y_external.value_counts(normalize=True).round(3).to_dict()}"
    )
    print(
        f"Internal (encoded) class distribution: "
        f"{y_internal.value_counts(normalize=True).round(3).to_dict()}"
    )

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    summary_rows: list[dict] = []
    all_fold_rows: list[dict] = []
    confusion_matrices: dict[str, np.ndarray] = {}

    for name, base_estimator in _build_models():
        print(f"\n{'=' * 72}")
        print(f"Running 3-class walk-forward for: {name}")
        print("=" * 72)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"{name}_wf_summary") as parent:
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("run_type", "summary_3class")

            fold_metrics: list[dict] = []
            all_y_true_int: list[np.ndarray] = []
            all_y_pred_int: list[np.ndarray] = []

            for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr_int = y_internal.iloc[train_idx].values
                y_te_int = y_internal.iloc[test_idx].values

                model = clone(base_estimator)
                model.fit(X_tr, y_tr_int)

                y_pred_int = model.predict(X_te)
                proba_raw = model.predict_proba(X_te)

                # Recover the model's own class order
                if hasattr(model, "classes_"):
                    model_classes = model.classes_
                elif hasattr(model, "named_steps") and hasattr(
                    model.named_steps["clf"], "classes_"
                ):
                    model_classes = model.named_steps["clf"].classes_
                else:
                    model_classes = np.array(INTERNAL_CLASSES)
                proba = _reorder_proba(proba_raw, model_classes, INTERNAL_CLASSES)

                m = _multiclass_metrics(y_te_int, y_pred_int, proba)
                m["fold"] = fold
                m["n_train"] = len(train_idx)
                m["n_test"] = len(test_idx)
                fold_metrics.append(m)
                all_y_true_int.append(y_te_int)
                all_y_pred_int.append(y_pred_int)

                print(
                    f"  [{name}] fold {fold}/{N_SPLITS}  "
                    f"acc={m['accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}  "
                    f"auc_ovr={m['roc_auc_ovr']:.3f}  logloss={m['log_loss']:.3f}"
                )

                with mlflow.start_run(experiment_id=experiment_id,
                                      run_name=f"{name}_fold_{fold}",
                                      nested=True):
                    mlflow.set_tag("model_type", name)
                    mlflow.set_tag("run_type", "fold_3class")
                    mlflow.set_tag("fold", fold)
                    for k, v in m.items():
                        if isinstance(v, (int, float)) and pd.notna(v):
                            mlflow.log_metric(k, float(v))

            metrics_df = pd.DataFrame(fold_metrics)
            for _, row in metrics_df.iterrows():
                all_fold_rows.append({"model": name, **row.to_dict()})

            summary = {
                "model": name,
                "accuracy_mean": metrics_df["accuracy"].mean(),
                "accuracy_std": metrics_df["accuracy"].std(),
                "f1_macro_mean": metrics_df["f1_macro"].mean(),
                "f1_macro_std": metrics_df["f1_macro"].std(),
                "roc_auc_ovr_mean": metrics_df["roc_auc_ovr"].mean(),
                "roc_auc_ovr_std": metrics_df["roc_auc_ovr"].std(),
                "log_loss_mean": metrics_df["log_loss"].mean(),
                "log_loss_std": metrics_df["log_loss"].std(),
            }
            summary_rows.append(summary)

            for k, v in summary.items():
                if isinstance(v, (int, float)) and pd.notna(v):
                    mlflow.log_metric(k, float(v))

            # Decode predictions back to external labels for the confusion matrix
            y_true_ext = np.array([INT_TO_EXT[v] for v in np.concatenate(all_y_true_int)])
            y_pred_ext = np.array([INT_TO_EXT[v] for v in np.concatenate(all_y_pred_int)])
            cm = confusion_matrix(y_true_ext, y_pred_ext, labels=EXTERNAL_CLASSES)
            confusion_matrices[name] = cm

    # -------------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_fold_rows).to_csv(
        RESULTS_DIR / "3class_per_fold_metrics.csv", index=False
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "3class_summary.csv", index=False)
    print(f"\nsaved -> {RESULTS_DIR / '3class_summary.csv'}")
    print(summary_df.round(4).to_string(index=False))

    # -------------------------------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n = len(confusion_matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=20)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
        for i in range(len(EXTERNAL_CLASSES)):
            for j in range(len(EXTERNAL_CLASSES)):
                txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                        ha="center", va="center", color=txt_color, fontsize=9)
    fig.suptitle("3-class confusion matrices (row-normalised, all folds combined)",
                 y=1.02)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "figure_3class_confusion_matrices.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fig_path}")


if __name__ == "__main__":
    main()
