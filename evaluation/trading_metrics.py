"""
Simple trading-aware evaluation: equity curve with transaction costs.

Builds a simplified directional strategy from each model's calibrated
probabilities and compares it against a buy-and-hold baseline. This is
intentionally a TOY backtest — it ignores order-book dynamics, slippage,
position sizing, intraday timing and many other realities. Its purpose
in the thesis (section 3.8.5) is to illustrate the gap between
classification metrics and economically meaningful outcomes, NOT to make
profitability claims.

Strategy rules:
    - Long only (no shorting). When predicted probability of up-move
      exceeds the model's tuned threshold, hold a long position the next
      day; otherwise be in cash.
    - Position changes incur a flat transaction cost (default 10 bps,
      conservative for crypto exchanges).
    - Returns are computed over the SAME 5-day forward horizon as the
      target (so the comparison to actual y is consistent).

Outputs:
    evaluation/results/trading_metrics_summary.csv
    evaluation/figures/figure_equity_curves.png

References:
    Jansen (2020), Ch. 8 — backtest engines and simulation pitfalls.
    The discussion in this thesis section explicitly acknowledges that
    a single-asset, no-slippage simulation is illustrative only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- config ---
FEATURES_PATH = Path("data/processed/btc_features.parquet")
RESULTS_DIR = Path("evaluation/results")
FIGURES_DIR = Path("evaluation/figures")

MODELS = ["LogReg", "RandomForest", "XGBoost"]
COST_BPS = 10  # 10 basis points = 0.10% per transaction

FORWARD_HORIZON = 5  # match the target horizon


# -----------------------------------------------------------------------------
def _load_inputs(model_name: str) -> tuple[pd.DataFrame, pd.Series, float]:
    """Load calibrated predictions for one model + tuned threshold."""
    preds = pd.read_parquet(RESULTS_DIR / f"{model_name}_calibrated_predictions.parquet")
    summary = pd.read_csv(RESULTS_DIR / "calibration_summary.csv")
    thr_row = summary[(summary["model"] == model_name) & (summary["variant"] == "cal_tuned")]
    threshold = float(thr_row["threshold"].iloc[0])
    return preds, threshold


def _compute_equity_curve(
    proba: pd.Series,
    forward_returns: pd.Series,
    threshold: float,
    cost_bps: int = COST_BPS,
) -> pd.DataFrame:
    """
    Compute strategy equity curve from probabilities + forward returns.

    Args:
        proba: predicted P(up) at time t (from calibrated model)
        forward_returns: realised log-return over the next 5 days at time t
        threshold: decision threshold (long if proba > threshold)
        cost_bps: transaction cost per change of position, in bps

    Returns:
        DataFrame indexed by date with columns:
            signal       (0 / 1)
            strat_ret    daily strategy log-return
            equity       cumulative wealth, starting at 1.0
            bh_equity    buy-and-hold benchmark
    """
    # Signal must shift by 1: today's prediction is acted on for tomorrow's return
    signal = (proba > threshold).astype(int)

    # Position changes -> transaction cost
    position_change = signal.diff().abs().fillna(signal.iloc[0])
    cost = position_change * (cost_bps / 10000.0)

    # Strategy return: scaled to daily-equivalent so we don't double-count
    # (forward_returns are 5-day log returns, we approximate daily exposure
    # by dividing by horizon). This is illustrative — see module docstring.
    daily_strategy_ret = signal.shift(1).fillna(0) * (forward_returns / FORWARD_HORIZON) - cost
    daily_bh_ret = forward_returns / FORWARD_HORIZON

    equity = np.exp(daily_strategy_ret.cumsum())
    bh_equity = np.exp(daily_bh_ret.cumsum())

    return pd.DataFrame(
        {
            "signal": signal,
            "strat_ret": daily_strategy_ret,
            "equity": equity,
            "bh_equity": bh_equity,
        },
        index=proba.index,
    )


def _trading_metrics(eq: pd.DataFrame) -> dict:
    """Compute hit-rate, mean strategy return, Sharpe (rf=0), max drawdown."""
    strat = eq["strat_ret"].dropna()
    n_trades = int((eq["signal"].diff().abs() > 0).sum())
    hit_rate = float((strat[eq["signal"].shift(1) == 1] > 0).mean()) if (
        eq["signal"].sum() > 0
    ) else float("nan")

    annualisation = np.sqrt(252)
    sharpe = float(strat.mean() / strat.std() * annualisation) if strat.std() > 0 else float("nan")

    rolling_peak = eq["equity"].cummax()
    drawdown = eq["equity"] / rolling_peak - 1.0
    max_dd = float(drawdown.min())

    final_equity = float(eq["equity"].iloc[-1])
    final_bh = float(eq["bh_equity"].iloc[-1])

    return {
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "mean_daily_ret": float(strat.mean()),
        "sharpe_ann": sharpe,
        "max_drawdown": max_dd,
        "final_equity": final_equity,
        "buy_hold_equity": final_bh,
    }


# -----------------------------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)

    # Forward 5-day log return aligned with predictions.
    fwd_ret = (df["close"].shift(-FORWARD_HORIZON) / df["close"]).apply(np.log)

    summary_rows: list[dict] = []
    eq_curves: dict[str, pd.DataFrame] = {}

    for model_name in MODELS:
        print(f"\n{'=' * 72}")
        print(f"Trading metrics: {model_name}")
        print("=" * 72)

        preds, threshold = _load_inputs(model_name)
        # Align with forward returns by index
        common_idx = preds.index.intersection(fwd_ret.index)
        proba = preds.loc[common_idx, "proba_cal"]
        ret = fwd_ret.loc[common_idx].dropna()
        proba = proba.loc[ret.index]

        eq = _compute_equity_curve(proba, ret, threshold=threshold, cost_bps=COST_BPS)
        eq_curves[model_name] = eq

        metrics = _trading_metrics(eq)
        metrics["model"] = model_name
        metrics["threshold"] = threshold
        summary_rows.append(metrics)

        print(
            f"  threshold={threshold:.2f}  "
            f"trades={metrics['n_trades']}  "
            f"hit-rate={metrics['hit_rate']:.3f}  "
            f"Sharpe={metrics['sharpe_ann']:.2f}  "
            f"MaxDD={metrics['max_drawdown']:.2%}  "
            f"final equity={metrics['final_equity']:.2f}x  "
            f"BH={metrics['buy_hold_equity']:.2f}x"
        )

    # -------------------------------------------------------------------------
    # CSV summary
    summary_df = pd.DataFrame(summary_rows)
    cols_first = ["model", "threshold", "n_trades", "hit_rate", "sharpe_ann",
                  "max_drawdown", "final_equity", "buy_hold_equity"]
    summary_df = summary_df[cols_first + [c for c in summary_df.columns if c not in cols_first]]
    summary_path = RESULTS_DIR / "trading_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nsaved -> {summary_path}")

    # -------------------------------------------------------------------------
    # Equity curves figure
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"LogReg": "#1f77b4", "RandomForest": "#2ca02c", "XGBoost": "#d62728"}
    for name, eq in eq_curves.items():
        ax.plot(eq.index, eq["equity"], label=f"{name} strategy",
                color=colors[name], linewidth=1.6)
    # Buy & hold uses the last model's bh_equity (all identical)
    last_eq = list(eq_curves.values())[-1]
    ax.plot(last_eq.index, last_eq["bh_equity"], label="Buy & hold",
            color="black", linewidth=1.6, linestyle="--")

    ax.set_title(
        f"Illustrative equity curves "
        f"(5-day directional strategies, {COST_BPS} bps transaction cost)"
    )
    ax.set_ylabel("Cumulative wealth (start = 1.0)")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {FIGURES_DIR / 'figure_equity_curves.png'}")


if __name__ == "__main__":
    main()
