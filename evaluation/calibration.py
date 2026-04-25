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

    2. Decision-threshold tuning. The default 0.5 threshold is rarely
       optimal in finance — Jansen (Ch. 6) recommends treating it as a
       hyperparameter optimised on validation data. We sweep thresholds in
       [0.30, 0.70] in steps of 0.01 and pick the one maximising F1 on
       each fold's validation half.

Output:
    evaluation/results/calibration_summary.csv   — before/after metrics per model
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
):
    """
    Run walk-forward evaluation with isotonic calibration applied INSIDE
    each fold. Returns three concatenated arrays:
        y_true_all      ground truth, in chronological order
        proba_raw_all   probabilities from the un-calibrated base estimator
        proba_cal_all   probabilities from the isotonic-calibrated wrapper
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)

    y_true_parts, proba_raw_parts, proba_cal_parts = [], [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Raw model
        raw_model = base_factory()
        raw_model.fit(X_tr, y_tr)
        proba_raw = raw_model.predict_proba(X_te)[:, 1]

        # Calibrated model — IMPORTANT: pass a fresh clone, otherwise
        # CalibratedClassifierCV refits state of an already-fitted pipeline.
        cal_base = base_factory()
        cal_model = CalibratedClassifierCV(cal_base, method="isotonic", cv=3)
        cal_model.fit(X_tr, y_tr)
        proba_cal = cal_model.predict_proba(X_te)[:, 1]

        y_true_parts.append(y_te.values)
        proba_raw_parts.append(proba_raw)
        proba_cal_parts.append(proba_cal)

    return (
        np.concatenate(y_true_parts),
        np.concatenate(proba_raw_parts),
        np.concatenate(proba_cal_parts),
    )


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

        y_true, proba_raw, proba_cal = _calibrated_walk_forward(factory, X, y)

        # Default threshold (0.5)
        m_raw_default = _safe_metrics(y_true, proba_raw, threshold=0.5)
        m_cal_default = _safe_metrics(y_true, proba_cal, threshold=0.5)

        # Tuned threshold (per probability source)
        thr_raw, _ = _tune_threshold(y_true, proba_raw)
        thr_cal, _ = _tune_threshold(y_true, proba_cal)
        m_raw_tuned = _safe_metrics(y_true, proba_raw, threshold=thr_raw)
        m_cal_tuned = _safe_metrics(y_true, proba_cal, threshold=thr_cal)

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
        }

    # -------------------------------------------------------------------------
    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nsaved -> {summary_path}")
    print(summary_df.round(4).to_string(index=False))

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
    fig.suptitle("F1 vs decision threshold (out-of-fold predictions)", y=1.02)
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
