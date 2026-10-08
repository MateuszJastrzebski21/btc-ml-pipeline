"""
Post-thesis audit: stress-testing the thesis results.

The thesis pipeline (scripts/run_all_models.py, evaluation/calibration.py,
evaluation/trading_metrics.py) is left untouched so that every number in the
thesis stays reproducible. This script re-asks the questions a sceptical
quant reviewer would ask, and answers them with numbers:

    1. Trivial baselines. Does the model beat "always predict up" (the
       training-window majority class), not just the persistence rules?
    2. Purging. The 5-day label overlaps the train/test boundary. Does
       purging the last 5 training rows of each fold (TimeSeriesSplit
       gap=5, Lopez de Prado 2018, Ch. 7) change the result?
    3. Stationarity. sma_20 and sma_50 are raw price levels. Does the
       result survive when only scale-free features are used?
    4. Significance. Is the mean per-fold ROC-AUC distinguishable from 0.5
       once overlapping labels are accounted for? (moving-block bootstrap)
    5. P&L accounting. The thesis equity curve spreads each 5-day forward
       return evenly over 5 days. Recompute it with plain next-day returns
       and compare Sharpe to buy-and-hold over the same out-of-sample window.

Outputs:
    evaluation/audit/audit_models.csv
    evaluation/audit/audit_backtest.csv
    evaluation/figures/figure_audit_auc_ci.png

Run (after features.engineer and evaluation.calibration):
    python -m scripts.run_audit
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from features.engineer import FEATURE_COLUMNS, FORWARD_HORIZON
from models.baseline import build_logreg_pipeline
from models.random_forest import build_random_forest
from models.xgboost_model import build_xgboost

# --- config ---
FEATURES_PATH = Path("data/processed/btc_features.parquet")
RESULTS_DIR = Path("evaluation/results")
AUDIT_DIR = Path("evaluation/audit")
FIGURES_DIR = Path("evaluation/figures")

N_SPLITS = 5
PURGE_GAP = FORWARD_HORIZON  # label at t uses close[t+5] -> drop 5 train rows
STATIONARY_FEATURES = [c for c in FEATURE_COLUMNS if c not in ("sma_20", "sma_50")]

N_BOOT = 2000
BLOCK_LEN = 20  # trading days; comfortably longer than the 5-day label overlap
COST_BPS = 10
SEED = 42

MODELS = {
    "LogReg": build_logreg_pipeline,
    "RandomForest": build_random_forest,
    "XGBoost": build_xgboost,
}


# -----------------------------------------------------------------------------
# Walk-forward that returns raw out-of-fold scores (optionally purged)
# -----------------------------------------------------------------------------
def _oof_scores(factory, X: pd.DataFrame, y: pd.Series, gap: int) -> pd.DataFrame:
    """Out-of-fold P(up) for every test row, with the fold id attached."""
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=gap)
    parts = []
    for fold, (tr, te) in enumerate(splitter.split(X), start=1):
        model = clone(factory())
        model.fit(X.iloc[tr], y.iloc[tr])
        parts.append(pd.DataFrame(
            {
                "fold": fold,
                "y_true": y.iloc[te].values,
                "proba": model.predict_proba(X.iloc[te])[:, 1],
                # prior of the TRAIN window: what "always predict the
                # majority class" would have known at that point in time
                "train_prior": float(y.iloc[tr].mean()),
            },
            index=X.index[te],
        ))
    return pd.concat(parts)


def _majority_baseline(y: pd.Series) -> pd.DataFrame:
    """Always predict the training-window majority class (here: always up)."""
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    parts = []
    for fold, (tr, te) in enumerate(splitter.split(y), start=1):
        majority = int(y.iloc[tr].mean() >= 0.5)
        parts.append(pd.DataFrame(
            {"fold": fold, "y_true": y.iloc[te].values, "y_pred": majority},
            index=y.index[te],
        ))
    return pd.concat(parts)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _per_fold_mean(oof: pd.DataFrame, fn) -> float:
    return float(np.mean([fn(g) for _, g in oof.groupby("fold")]))


def _block_bootstrap_auc(oof: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Moving-block bootstrap of the mean per-fold ROC-AUC.

    Each fold's test set is resampled in contiguous blocks of BLOCK_LEN days,
    which keeps the autocorrelation induced by overlapping 5-day labels
    inside a block instead of pretending every day is an independent draw.
    """
    folds = [(g["y_true"].to_numpy(), g["proba"].to_numpy()) for _, g in oof.groupby("fold")]
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        aucs = []
        for yt, pr in folds:
            n = len(yt)
            starts = rng.integers(0, n - BLOCK_LEN + 1, size=int(np.ceil(n / BLOCK_LEN)))
            idx = (starts[:, None] + np.arange(BLOCK_LEN)).ravel()[:n]
            if len(np.unique(yt[idx])) == 2:
                aucs.append(roc_auc_score(yt[idx], pr[idx]))
        out[b] = np.mean(aucs)
    return out


def _model_row(name: str, variant: str, oof: pd.DataFrame, rng) -> dict:
    boot = _block_bootstrap_auc(oof, rng)
    return {
        "model": name,
        "variant": variant,
        "auc_mean_per_fold": _per_fold_mean(oof, lambda g: roc_auc_score(g["y_true"], g["proba"])),
        "auc_ci_low": float(np.quantile(boot, 0.025)),
        "auc_ci_high": float(np.quantile(boot, 0.975)),
        "p_auc_le_0.5": float((boot <= 0.5).mean()),
        "auc_pooled": float(roc_auc_score(oof["y_true"], oof["proba"])),
        "accuracy_thr_0.5": float(accuracy_score(oof["y_true"], oof["proba"] > 0.5)),
    }


# -----------------------------------------------------------------------------
# Backtest with plain next-day returns
# -----------------------------------------------------------------------------
def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else float("nan")


def _max_dd(log_ret: pd.Series) -> float:
    eq = np.exp(log_ret.cumsum())
    return float((eq / eq.cummax() - 1).min())


def _backtest(close: pd.Series, preds: pd.DataFrame, thresholds: pd.Series) -> dict:
    """
    Long/flat. Signal formed at the close of day t (features use data up to
    and including t) is held over (t, t+1]. Costs on every position change.
    Same per-fold calibrated probabilities and validation-selected thresholds
    as the thesis; only the return accounting differs.
    """
    r_next = np.log(close.shift(-1) / close).reindex(preds.index)
    thr = preds["fold"].map(thresholds)
    pos = (preds["proba_cal"] > thr).astype(int)
    cost = pos.diff().abs().fillna(pos.iloc[0]) * COST_BPS / 1e4
    strat = (pos * r_next - cost).dropna()
    bh = r_next.dropna()
    return {
        "exposure": float(pos.mean()),
        "n_trades": int(pos.diff().abs().sum()),
        "sharpe_ann": _sharpe(strat),
        "max_drawdown": _max_dd(strat),
        "final_equity": float(np.exp(strat.sum())),
        "bh_sharpe_ann": _sharpe(bh),
        "bh_max_drawdown": _max_dd(bh),
        "bh_final_equity": float(np.exp(bh.sum())),
    }


# -----------------------------------------------------------------------------
def main() -> None:
    assert FEATURES_PATH.exists(), "Run `python -m features.engineer` first."
    df = pd.read_parquet(FEATURES_PATH)
    y = df["y"]
    rng = np.random.default_rng(SEED)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. trivial baseline -------------------------------------------------
    maj = _majority_baseline(y)
    print("Always-up baseline (training-window majority class):")
    print(f"  accuracy={_per_fold_mean(maj, lambda g: accuracy_score(g['y_true'], g['y_pred'])):.3f}  "
          f"F1={_per_fold_mean(maj, lambda g: f1_score(g['y_true'], g['y_pred'])):.3f}  "
          f"(pooled F1={f1_score(maj['y_true'], maj['y_pred']):.3f})  ROC-AUC=0.500 by construction")

    # --- 2-4. thesis setup vs purged vs purged + stationary features ----------
    variants = {
        "thesis (no purge, 10 features)": (FEATURE_COLUMNS, 0),
        "purged gap=5": (FEATURE_COLUMNS, PURGE_GAP),
        "purged + stationary features": (STATIONARY_FEATURES, PURGE_GAP),
    }
    rows = [{
        "model": "AlwaysUp", "variant": "baseline",
        "auc_mean_per_fold": 0.5, "auc_ci_low": 0.5, "auc_ci_high": 0.5,
        "p_auc_le_0.5": 1.0, "auc_pooled": 0.5,
        "accuracy_thr_0.5": float(accuracy_score(maj["y_true"], maj["y_pred"])),
    }]
    for variant, (cols, gap) in variants.items():
        for name, factory in MODELS.items():
            oof = _oof_scores(factory, df[cols], y, gap=gap)
            row = _model_row(name, variant, oof, rng)
            rows.append(row)
            print(f"  {variant:32s} {name:12s} AUC={row['auc_mean_per_fold']:.3f} "
                  f"[{row['auc_ci_low']:.3f}, {row['auc_ci_high']:.3f}]  "
                  f"P(AUC<=0.5)={row['p_auc_le_0.5']:.2f}  pooled={row['auc_pooled']:.3f}")
    models_df = pd.DataFrame(rows)
    models_df.to_csv(AUDIT_DIR / "audit_models.csv", index=False)

    # --- 5. backtest with next-day returns ------------------------------------
    thr_df = pd.read_csv(RESULTS_DIR / "calibration_thresholds_per_fold.csv")
    old = pd.read_csv(RESULTS_DIR / "trading_metrics_summary.csv").set_index("model")
    bt_rows = []
    for name in MODELS:
        preds = pd.read_parquet(RESULTS_DIR / f"{name}_calibrated_predictions.parquet")
        wf = pd.read_parquet(RESULTS_DIR / f"{name}_predictions.parquet")
        preds["fold"] = wf["fold"].reindex(preds.index)
        thresholds = (thr_df[(thr_df["model"] == name) & (thr_df["source"] == "cal")]
                      .set_index("fold")["threshold"])
        res = _backtest(df["close"], preds, thresholds)
        res = {"model": name, "thesis_sharpe_ann": float(old.loc[name, "sharpe_ann"]), **res}
        bt_rows.append(res)
        print(f"  backtest {name:12s} Sharpe {res['thesis_sharpe_ann']:.2f} (thesis) -> "
              f"{res['sharpe_ann']:.2f} (next-day)  vs B&H {res['bh_sharpe_ann']:.2f}  "
              f"exposure={res['exposure']:.0%}")
    pd.DataFrame(bt_rows).to_csv(AUDIT_DIR / "audit_backtest.csv", index=False)

    # --- figure ----------------------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = models_df[models_df["model"] != "AlwaysUp"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    offsets = dict(zip(variants, (-0.22, 0.0, 0.22)))
    colors = dict(zip(variants, ("#1f77b4", "#ff7f0e", "#2ca02c")))
    for variant in variants:
        sub = plot_df[plot_df["variant"] == variant]
        xs = np.arange(len(sub)) + offsets[variant]
        ax.errorbar(
            xs, sub["auc_mean_per_fold"],
            yerr=[sub["auc_mean_per_fold"] - sub["auc_ci_low"],
                  sub["auc_ci_high"] - sub["auc_mean_per_fold"]],
            fmt="o", capsize=4, color=colors[variant], label=variant,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="no skill (0.5)")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(list(MODELS))
    ax.set_ylabel("Mean per-fold ROC-AUC")
    ax.set_title(f"ROC-AUC with 95% moving-block bootstrap CI (block={BLOCK_LEN}d, {N_BOOT} resamples)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_audit_auc_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {AUDIT_DIR}/audit_models.csv, audit_backtest.csv, figure_audit_auc_ci.png")


if __name__ == "__main__":
    main()
