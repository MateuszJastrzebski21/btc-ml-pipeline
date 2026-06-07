"""
Probability calibration and decision-threshold tuning.

This module implements section 3.8 of the thesis. Two complementary
post-hoc analyses are performed on the predictions already produced by
walk_forward_evaluate (see evaluation/walk_forward.py):

    1. Isotonic calibration. Each model's per-fold predicted probabilities
       are re-fitted using sklearn's CalibratedClassifierCV with
       method="isotonic" and cv=3 inside each walk-forward fold. This
       approximates monotonic-but-flexible mapping from raw probabilities
       to actual frequencies of the positive class.

    2. Decision-threshold tuning WITHOUT test-set leakage. Inside every
       walk-forward fold we carve an internal validation slice from the last
       INNER_VAL_FRACTION (20%) of that fold's TRAINING window, fit a separate
       model on the remaining 80%, and pick the threshold in [0.30, 0.70]
       (step 0.01) that maximises F1 on that internal validation slice only.
       The selected threshold is then applied to the fold's TEST predictions.
       Per-fold test predictions are aggregated before scoring; the test set
       is NEVER used to choose the threshold. The `threshold` column in the
       summary holds the mean of the five per-fold thresholds.

Output:
    evaluation/results/calibration_summary.csv   — before/after metrics per model
    evaluation/results/calibration_thresholds_per_fold.csv — per-fold thresholds (raw & cal)
    evaluation/figures/figure_calibration_curves.png   — reliability diagrams
    evaluation/figures/figure_threshold_tuning.png     — F1 vs threshold curves

References:
    Jansen (2020), Ch. 6, pp. 159–170. Threshold and confusion-matrix discussion.
    scikit-learn User Guide, "Probability calibration":
        https://scikit-learn.org/stable/modules/calibration.html
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from models.baseline import build_logreg_pipeline
from models.random_forest import build_random_forest
from models.xgboost_model import build_xgboost

# --- config ---
FEATURES_PATH = Path("data/processed/btc_features.parquet")
RESULTS_DIR = Path("evaluation/results")
FIGURES_DIR = Path("evaluation/figures")

FEATURE_COLUMNS = [
    "log_return_1d",
    "momentum_5", "momentum_10", "momentum_20",
    "rolling_vol_10", "rolling_vol_20",
    "rsi_14",
    "sma_20", "sma_50", "sma_20_over_50_ratio",
]

N_SPLITS = 5
THRESHOLDS = np.linspace(0.30, 0.70, 41)  # step 0.01
INNER_VAL_FRACTION = 0.2  # last 20% of each fold's TRAIN window = internal validation

# Models to calibrate. Persistence baselines are skipped — they return
# degenerate pseudo-probabilities and calibration would be meaningless.
MODELS_TO_CALIBRATE: list[tuple[str, callable]] = [
    ("LogReg", build_logreg_pipeline),
    ("RandomForest", build_random_forest),
    ("XGBoost", build_xgboost),
]


# -----------------------------------------------------------------------------
def _calibrated_walk_forward(
    base_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    inner_val_fraction: float = INNER_VAL_FRACTION,
):
    """
    Walk-forward with isotonic calibration applied INSIDE each fold, plus
    per-fold decision-threshold selection that NEVER touches the test set.

    Per fold:
      1. Fit raw and isotonic-calibrated models on the FULL training window and
         score the test set -> proba_raw / proba_cal (the out-of-fold
         probabilities used everywhere downstream; unchanged from before).
      2. Carve an internal validation slice = last `inner_val_fraction` of the
         TRAIN rows (chronological). Fit a SEPARATE model on the earlier part,
         score the validation slice, and pick the F1-optimal threshold there.
      3. Apply that fold's validation threshold to this fold's TEST scores.

    Returns a dict of chronologically-concatenated TEST arrays plus the list of
    per-fold thresholds (raw and calibrated):
        y_true, proba_raw, proba_cal, y_pred_raw_tuned, y_pred_cal_tuned,
        thr_raw_per_fold, thr_cal_per_fold.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)

    y_true_parts, proba_raw_parts, proba_cal_parts = [], [], []
    pred_raw_tuned_parts, pred_cal_tuned_parts = [], []
    thr_raw_per_fold, thr_cal_per_fold = [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # (1) Models trained on the FULL training window -> test scores.
        # IMPORTANT: pass a fresh clone to the calibrator, otherwise
        # CalibratedClassifierCV refits state of an already-fitted pipeline.
        raw_model = base_factory()
        raw_model.fit(X_tr, y_tr)
        proba_raw = raw_model.predict_proba(X_te)[:, 1]

        cal_base = base_factory()
        cal_model = CalibratedClassifierCV(cal_base, method="isotonic", cv=3)
        cal_model.fit(X_tr, y_tr)
        proba_cal = cal_model.predict_proba(X_te)[:, 1]

        # (2) Internal validation = last fraction of TRAIN only (chronological).
        # Earliest rows train the threshold-selection model, latest rows are the
        # held-out validation slice on which the threshold is chosen.
        n_tr = len(X_tr)
        cut = int(round(n_tr * (1.0 - inner_val_fraction)))
        X_in_tr, X_in_val = X_tr.iloc[:cut], X_tr.iloc[cut:]
        y_in_tr, y_in_val = y_tr.iloc[:cut], y_tr.iloc[cut:]

        thr_raw = _tune_threshold_on_validation(
            base_factory, X_in_tr, y_in_tr, X_in_val, y_in_val, calibrated=False
        )
        thr_cal = _tune_threshold_on_validation(
            base_factory, X_in_tr, y_in_tr, X_in_val, y_in_val, calibrated=True
        )

        # (3) Apply the validation-selected thresholds to this fold's TEST scores.
        pred_raw_tuned_parts.append((proba_raw > thr_raw).astype(int))
        pred_cal_tuned_parts.append((proba_cal > thr_cal).astype(int))
        thr_raw_per_fold.append(thr_raw)
        thr_cal_per_fold.append(thr_cal)

        y_true_parts.append(y_te.values)
        proba_raw_parts.append(proba_raw)
        proba_cal_parts.append(proba_cal)

    return {
        "y_true": np.concatenate(y_true_parts),
        "proba_raw": np.concatenate(proba_raw_parts),
        "proba_cal": np.concatenate(proba_cal_parts),
        "y_pred_raw_tuned": np.concatenate(pred_raw_tuned_parts),
        "y_pred_cal_tuned": np.concatenate(pred_cal_tuned_parts),
        "thr_raw_per_fold": thr_raw_per_fold,
        "thr_cal_per_fold": thr_cal_per_fold,
    }


def _tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Find threshold in THRESHOLDS that maximises F1; return (threshold, f1)."""
    f1_scores = [
        f1_score(y_true, (y_proba > t).astype(int), zero_division=0)
        for t in THRESHOLDS
    ]
    best_idx = int(np.argmax(f1_scores))
    return float(THRESHOLDS[best_idx]), float(f1_scores[best_idx])


def _safe_metrics(y_true, y_proba, threshold) -> dict:
    """Return a dict of headline metrics for a given threshold."""
    y_pred = (y_proba > threshold).astype(int)
    proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
    return {
        "threshold": threshold,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, proba_clipped, labels=[0, 1]),
    }


def _tune_threshold_on_validation(
    base_factory, X_in_tr, y_in_tr, X_in_val, y_in_val, calibrated: bool,
) -> float:
    """
    Select a threshold WITHOUT seeing the test set: fit a model on the
    inner-training rows only, score the inner-validation rows, and return the
    F1-optimal threshold in THRESHOLDS. For the calibrated variant the same
    isotonic wrapper as the main pipeline is used, so the threshold is chosen
    on calibrated scores. Falls back to 0.5 when the validation slice is too
    small or single-class (an F1-optimal threshold would be meaningless).
    """
    y_val = np.asarray(y_in_val)
    if len(y_val) < 10 or len(np.unique(y_val)) < 2:
        return 0.5
    model = base_factory()
    if calibrated:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
    model.fit(X_in_tr, y_in_tr)
    proba_val = model.predict_proba(X_in_val)[:, 1]
    thr, _ = _tune_threshold(y_val, proba_val)
    return thr


def _tuned_metrics(y_true, y_proba, y_pred, reported_threshold) -> dict:
    """
    Headline metrics for a tuned variant. `y_pred` was produced per fold using
    each fold's validation-selected threshold; `reported_threshold` is the mean
    of those per-fold thresholds, stored for the summary only. ROC-AUC and
    log-loss are threshold-independent, hence identical to the default variant.
    Keys match _safe_metrics so the CSV schema is unchanged.
    """
    proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
    return {
        "threshold": reported_threshold,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, proba_clipped, labels=[0, 1]),
    }


# -----------------------------------------------------------------------------
def main() -> None:
    assert FEATURES_PATH.exists(), (
        f"Features file not found: {FEATURES_PATH}\n"
        "Run `python -m features.engineer` first."
    )

    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["y"]
    print(f"Loaded dataset: {X.shape}, positive rate: {y.mean():.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    calibration_data: dict[str, dict] = {}

    for name, factory in MODELS_TO_CALIBRATE:
        print(f"\n{'=' * 72}")
        print(f"Calibration + threshold tuning: {name}")
        print("=" * 72)

        wf = _calibrated_walk_forward(factory, X, y)
        y_true = wf["y_true"]
        proba_raw = wf["proba_raw"]
        proba_cal = wf["proba_cal"]

        # Default threshold (0.5) — unchanged baseline.
        m_raw_default = _safe_metrics(y_true, proba_raw, threshold=0.5)
        m_cal_default = _safe_metrics(y_true, proba_cal, threshold=0.5)

        # Tuned: thresholds were chosen per fold on that fold's internal
        # validation slice (never the test set) and already applied to the
        # fold's test predictions inside _calibrated_walk_forward. We report the
        # mean per-fold threshold; F1 is computed on the aggregated test
        # predictions. ROC-AUC / log-loss are threshold-free -> equal default.
        thr_raw = round(float(np.mean(wf["thr_raw_per_fold"])), 3)
        thr_cal = round(float(np.mean(wf["thr_cal_per_fold"])), 3)
        m_raw_tuned = _tuned_metrics(y_true, proba_raw, wf["y_pred_raw_tuned"], thr_raw)
        m_cal_tuned = _tuned_metrics(y_true, proba_cal, wf["y_pred_cal_tuned"], thr_cal)

        # Console output
        print(f"  RAW (uncalibrated):")
        print(f"    default thr=0.50  F1={m_raw_default['f1']:.4f}  "
              f"AUC={m_raw_default['roc_auc']:.4f}  "
              f"logloss={m_raw_default['log_loss']:.4f}")
        print(f"    tuned   thr={thr_raw:.2f}  F1={m_raw_tuned['f1']:.4f}  "
              f"AUC={m_raw_tuned['roc_auc']:.4f}  "
              f"logloss={m_raw_tuned['log_loss']:.4f}")
        print(f"  CALIBRATED (isotonic):")
        print(f"    default thr=0.50  F1={m_cal_default['f1']:.4f}  "
              f"AUC={m_cal_default['roc_auc']:.4f}  "
              f"logloss={m_cal_default['log_loss']:.4f}")
        print(f"    tuned   thr={thr_cal:.2f}  F1={m_cal_tuned['f1']:.4f}  "
              f"AUC={m_cal_tuned['roc_auc']:.4f}  "
              f"logloss={m_cal_tuned['log_loss']:.4f}")
        print(
            f"    per-fold thresholds  "
            f"raw={[round(t, 3) for t in wf['thr_raw_per_fold']]}  "
            f"cal={[round(t, 3) for t in wf['thr_cal_per_fold']]}"
        )

        for label, m in (
            ("raw_default", m_raw_default),
            ("raw_tuned", m_raw_tuned),
            ("cal_default", m_cal_default),
            ("cal_tuned", m_cal_tuned),
        ):
            summary_rows.append({"model": name, "variant": label, **m})

        calibration_data[name] = {
            "y_true": y_true,
            "proba_raw": proba_raw,
            "proba_cal": proba_cal,
            "thr_raw": thr_raw,
            "thr_cal": thr_cal,
            "thr_raw_per_fold": wf["thr_raw_per_fold"],
            "thr_cal_per_fold": wf["thr_cal_per_fold"],
        }

    # -------------------------------------------------------------------------
    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nsaved -> {summary_path}")
    print(summary_df.round(4).to_string(index=False))

    # -------------------------------------------------------------------------
    # Per-fold thresholds (transparency). Each threshold was selected on that
    # fold's internal validation split, never the test set. One row per
    # (model, source, fold); does not alter the calibration_summary schema.
    per_fold_rows: list[dict] = []
    for name, d in calibration_data.items():
        for fold_i, t in enumerate(d["thr_raw_per_fold"], start=1):
            per_fold_rows.append({"model": name, "source": "raw",
                                  "fold": fold_i, "threshold": round(float(t), 2)})
        for fold_i, t in enumerate(d["thr_cal_per_fold"], start=1):
            per_fold_rows.append({"model": name, "source": "cal",
                                  "fold": fold_i, "threshold": round(float(t), 2)})
    per_fold_path = RESULTS_DIR / "calibration_thresholds_per_fold.csv"
    pd.DataFrame(per_fold_rows).to_csv(per_fold_path, index=False)
    print(f"saved -> {per_fold_path}")

    # -------------------------------------------------------------------------
    # Figure: reliability diagrams (calibration curves) before/after
    n = len(MODELS_TO_CALIBRATE)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, _) in zip(axes, MODELS_TO_CALIBRATE):
        d = calibration_data[name]
        # 10-bin reliability curve
        frac_raw, mean_raw = calibration_curve(d["y_true"], d["proba_raw"], n_bins=10)
        frac_cal, mean_cal = calibration_curve(d["y_true"], d["proba_cal"], n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="perfect")
        ax.plot(mean_raw, frac_raw, "o-", label="raw", linewidth=2)
        ax.plot(mean_cal, frac_cal, "s-", label="isotonic", linewidth=2)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Empirical fraction of positives")
    fig.suptitle("Reliability diagrams — before vs after isotonic calibration",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_calibration_curves.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {FIGURES_DIR / 'figure_calibration_curves.png'}")

    # -------------------------------------------------------------------------
    # Figure: F1 vs threshold for each model (calibrated probabilities)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, _) in zip(axes, MODELS_TO_CALIBRATE):
        d = calibration_data[name]
        f1_raw = [f1_score(d["y_true"], (d["proba_raw"] > t).astype(int),
                           zero_division=0) for t in THRESHOLDS]
        f1_cal = [f1_score(d["y_true"], (d["proba_cal"] > t).astype(int),
                           zero_division=0) for t in THRESHOLDS]
        ax.plot(THRESHOLDS, f1_raw, label="raw", linewidth=2)
        ax.plot(THRESHOLDS, f1_cal, label="isotonic", linewidth=2)
        ax.axvline(d["thr_raw"], color="C0", linestyle=":", alpha=0.6)
        ax.axvline(d["thr_cal"], color="C1", linestyle=":", alpha=0.6)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4, label="default 0.5")
        ax.set_title(name)
        ax.set_xlabel("Decision threshold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("F1")
    fig.suptitle(
        "F1 vs decision threshold on out-of-fold TEST predictions (diagnostic);\n"
        "dotted lines = mean per-fold threshold selected on internal validation",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_threshold_tuning.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {FIGURES_DIR / 'figure_threshold_tuning.png'}")

    # -------------------------------------------------------------------------
    # Save calibrated predictions for downstream use (trading metrics, regimes)
    for name, d in calibration_data.items():
        out = pd.DataFrame(
            {
                "y_true": d["y_true"],
                "proba_raw": d["proba_raw"],
                "proba_cal": d["proba_cal"],
            },
            index=df.index[: len(d["y_true"])]  # placeholder, see note below
            if len(d["y_true"]) == len(df) else None,
        )
        # Re-index using the actual test-set timestamps from the existing
        # predictions parquet, so the ordering is reliable.
        existing = pd.read_parquet(RESULTS_DIR / f"{name}_predictions.parquet")
        if len(existing) == len(out):
            out.index = existing.index
        out_path = RESULTS_DIR / f"{name}_calibrated_predictions.parquet"
        out.to_parquet(out_path)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
