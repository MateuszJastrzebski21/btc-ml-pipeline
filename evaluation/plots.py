"""
Publication-ready comparison plots for the thesis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("evaluation/results")
FIGURES_DIR = Path("evaluation/figures")

# Order: weakest baselines first.
MODELS = ["Persistence1d", "Persistence5d", "LogReg", "RandomForest", "XGBoost"]

MODEL_COLORS = {
    "Persistence1d": "#bdbdbd",   # light gray
    "Persistence5d": "#7f7f7f",   # darker gray
    "LogReg":        "#1f77b4",
    "RandomForest":  "#2ca02c",
    "XGBoost":       "#d62728",
}

# Persistence baselines are visually de-emphasized via dashed line style.
DASHED = {"Persistence1d", "Persistence5d"}


def _load_per_fold() -> pd.DataFrame:
    frames = []
    for m in MODELS:
        path = RESULTS_DIR / f"{m}_per_fold_metrics.csv"
        if not path.exists():
            print(f"  warning: {path} not found - skipping {m}")
            continue
        df = pd.read_csv(path)
        df["model"] = m
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_predictions(model_name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{model_name}_predictions.parquet")


def _load_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "summary_all_models.csv", index_col=0)


def _models_present() -> list[str]:
    return [m for m in MODELS if (RESULTS_DIR / f"{m}_per_fold_metrics.csv").exists()]


# -----------------------------------------------------------------------------
def plot_metrics_per_fold(save_path: Path) -> None:
    df = _load_per_fold()
    metrics = ["accuracy", "f1", "roc_auc", "log_loss"]
    models = _models_present()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for model in models:
            sub = df[df["model"] == model].sort_values("fold")
            linestyle = "--" if model in DASHED else "-"
            ax.plot(sub["fold"], sub[metric], marker="o", label=model,
                    color=MODEL_COLORS[model], linewidth=2, linestyle=linestyle)
        ax.set_title(metric.upper().replace("_", "-"), fontsize=11)
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 5])

        if metric == "roc_auc":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        if metric == "log_loss":
            ax.axhline(0.693, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Model metrics across walk-forward folds", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def plot_summary_bars(save_path: Path) -> None:
    summary = _load_summary()
    metrics = ["accuracy", "f1", "roc_auc", "log_loss"]
    models = _models_present()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    x = np.arange(len(models))
    for ax, metric in zip(axes, metrics):
        means = summary[f"{metric}_mean"].reindex(models).values
        stds = summary[f"{metric}_std"].reindex(models).values
        colors = [MODEL_COLORS[m] for m in models]
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_title(metric.upper().replace("_", "-"))
        ax.grid(True, axis="y", alpha=0.3)

        if metric == "roc_auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        if metric == "log_loss":
            ax.axhline(0.693, color="gray", linestyle="--", linewidth=1)

    fig.suptitle("Walk-forward metric summary (mean +/- std across 5 folds)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def plot_probability_distributions(save_path: Path) -> None:
    """
    Persistence baselines are excluded - they return degenerate
    pseudo-probabilities (0.99 / 0.01) that would visually dominate
    the histogram and convey nothing about calibration.
    """
    models = [m for m in _models_present() if m not in DASHED]
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        preds = _load_predictions(model)
        ax.hist(preds["y_proba"], bins=30, color=MODEL_COLORS[model],
                alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"{model}\n(predicted P(up-move))")
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Count of predictions")
    fig.suptitle(
        "Distribution of predicted probabilities (out-of-fold, all 5 folds combined)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")
    plot_metrics_per_fold(FIGURES_DIR / "figure_metrics_per_fold.png")
    plot_summary_bars(FIGURES_DIR / "figure_summary_bars.png")
    plot_probability_distributions(FIGURES_DIR / "figure_probability_distributions.png")
    print(f"Done. Figures are in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
