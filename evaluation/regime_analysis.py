"""
Regime-stability analysis.

For each model in {LogReg, RandomForest, XGBoost}, splits the existing
walk-forward predictions by market regime and computes accuracy / F1 /
ROC-AUC / log-loss within each regime separately. Output is a tidy
table that goes into thesis section 3.9.

Regimes are the same five manually-defined segments used in
features/engineer.py:
    pre_2020, bull_2020_2021, bear_2022, recovery_2023, bull_2024_plus

Reference:
    Jansen (2020), Ch. 6, p. 170 — "Challenges with cross-validation in
    finance". Non-stationarity and regime shifts are treated there as the
    chief obstacle to time-series CV in practice.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

# --- config ---
RESULTS_DIR = Path("evaluation/results")
FIGURES_DIR = Path("evaluation/figures")

MODELS = ["Persistence1d", "Persistence5d", "LogReg", "RandomForest", "XGBoost"]

REGIME_ORDER = [
    "pre_2020",
    "bull_2020_2021",
    "bear_2022",
    "recovery_2023",
    "bull_2024_plus",
]


def _safe_metrics(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
        metrics["log_loss"] = log_loss(y_true, proba_clipped, labels=[0, 1])
    else:
        metrics["roc_auc"] = np.nan
        metrics["log_loss"] = np.nan
    return metrics


def main() -> None:
    rows: list[dict] = []
    distribution_rows: list[dict] = []

    for model in MODELS:
        path = RESULTS_DIR / f"{model}_predictions.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {model}")
            continue
        preds = pd.read_parquet(path)

        # Sanity: must contain regime column (engineer.py adds it; runners join).
        if "regime" not in preds.columns:
            raise RuntimeError(
                f"{path} has no 'regime' column. Re-run scripts/run_with_mlflow.py."
            )

        for regime in REGIME_ORDER:
            sub = preds[preds["regime"] == regime]
            if len(sub) < 10:
                # Skip regimes that are barely represented in the test folds
                continue

            m = _safe_metrics(sub["y_true"].values, sub["y_pred"].values,
                              sub["y_proba"].values)
            row = {"model": model, "regime": regime, "n_obs": len(sub), **m}
            rows.append(row)
            distribution_rows.append({"model": model, "regime": regime,
                                      "n_obs": len(sub),
                                      "positive_rate": float(sub["y_true"].mean())})

    long_df = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "regime_metrics.csv"
    long_df.to_csv(summary_path, index=False)
    print(f"\nsaved -> {summary_path}")

    # ------------------------------------------------------------------
    # Pivot table for printing — easier for the thesis
    print("\nAccuracy per regime:")
    print(long_df.pivot(index="regime", columns="model", values="accuracy")
                 .reindex(REGIME_ORDER).round(3).to_string())
    print("\nF1 per regime:")
    print(long_df.pivot(index="regime", columns="model", values="f1")
                 .reindex(REGIME_ORDER).round(3).to_string())
    print("\nROC-AUC per regime:")
    print(long_df.pivot(index="regime", columns="model", values="roc_auc")
                 .reindex(REGIME_ORDER).round(3).to_string())

    # ------------------------------------------------------------------
    # Regime distribution (independent of model — just for the thesis text)
    dist_df = pd.DataFrame(distribution_rows).drop_duplicates(subset=["regime"])
    dist_df = dist_df[["regime", "n_obs", "positive_rate"]].copy()
    dist_df["regime"] = pd.Categorical(dist_df["regime"], categories=REGIME_ORDER, ordered=True)
    dist_df = dist_df.sort_values("regime")
    dist_path = RESULTS_DIR / "regime_distribution.csv"
    dist_df.to_csv(dist_path, index=False)
    print(f"\nRegime distribution:\n{dist_df.to_string(index=False)}")
    print(f"saved -> {dist_path}")

    # ------------------------------------------------------------------
    # Figure: bar chart of ROC-AUC per regime per model
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, metric, ylabel in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1", "F1"),
        (axes[2], "roc_auc", "ROC-AUC"),
    ]:
        pivot = long_df.pivot(index="regime", columns="model", values=metric)\
                       .reindex(REGIME_ORDER)
        # Order columns: persistence first, ML last
        pivot = pivot.reindex(columns=[m for m in MODELS if m in pivot.columns])
        pivot.plot.bar(ax=ax, width=0.85, edgecolor="black", linewidth=0.4)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Regime")
        ax.set_title(metric.upper().replace("_", "-"))
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=20)
        if metric == "roc_auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        if metric == "accuracy":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)

    fig.suptitle("Model performance by market regime (out-of-fold predictions)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_regime_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {FIGURES_DIR / 'figure_regime_metrics.png'}")


if __name__ == "__main__":
    main()
