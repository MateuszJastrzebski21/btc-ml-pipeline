"""
End-to-end reproducibility check.

Runs every step of the pipeline in order and reports OK / FAIL for each.
Useful before the supervisor demo: exec this once on a clean clone, see
the full green list, and you know nothing is broken.

Usage:
    python -m scripts.run_full_pipeline

Each step is a subprocess call so a failure in one stage does not kill
later checks (we still want to know what else might be wrong).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


# Each item: (description, command_as_list, list_of_expected_output_files)
STEPS: list[tuple[str, list[str], list[str]]] = [
    (
        "1. Fetch raw OHLCV from Binance",
        [sys.executable, "data/fetch_binance.py"],
        ["data/raw/btc_daily.parquet"],
    ),
    (
        "2. Engineer binary features and target",
        [sys.executable, "-m", "features.engineer"],
        ["data/processed/btc_features.parquet"],
    ),
    (
        "3. Smoke-test walk-forward on DummyClassifier",
        [sys.executable, "-m", "evaluation.walk_forward"],
        [],
    ),
    (
        "4. Run main pipeline (5 models, MLflow tracking)",
        [sys.executable, "-m", "scripts.run_with_mlflow"],
        [
            "evaluation/results/summary_all_models.csv",
            "evaluation/results/Persistence1d_per_fold_metrics.csv",
            "evaluation/results/Persistence5d_per_fold_metrics.csv",
            "evaluation/results/LogReg_per_fold_metrics.csv",
            "evaluation/results/RandomForest_per_fold_metrics.csv",
            "evaluation/results/XGBoost_per_fold_metrics.csv",
            "evaluation/figures/figure_metrics_per_fold.png",
            "evaluation/figures/figure_summary_bars.png",
            "evaluation/figures/figure_probability_distributions.png",
        ],
    ),
    (
        "5. Hyperparameter tuning (21 configurations)",
        [sys.executable, "-m", "scripts.run_tuning"],
        ["evaluation/results/tuning_summary.csv"],
    ),
    (
        "6. Per-regime stability analysis",
        [sys.executable, "-m", "evaluation.regime_analysis"],
        [
            "evaluation/results/regime_metrics.csv",
            "evaluation/results/regime_distribution.csv",
            "evaluation/figures/figure_regime_metrics.png",
        ],
    ),
    (
        "7. Probability calibration + threshold tuning",
        [sys.executable, "-m", "evaluation.calibration"],
        [
            "evaluation/results/calibration_summary.csv",
            "evaluation/figures/figure_calibration_curves.png",
            "evaluation/figures/figure_threshold_tuning.png",
        ],
    ),
    (
        "8. Trading metrics + equity curves",
        [sys.executable, "-m", "evaluation.trading_metrics"],
        [
            "evaluation/results/trading_metrics_summary.csv",
            "evaluation/figures/figure_equity_curves.png",
        ],
    ),
    (
        "9. Engineer 3-class target",
        [sys.executable, "-m", "features.engineer_3class"],
        ["data/processed/btc_features_3class.parquet"],
    ),
    (
        "10. Run 3-class classification",
        [sys.executable, "-m", "scripts.run_3class"],
        [
            "evaluation/results/3class_summary.csv",
            "evaluation/results/3class_per_fold_metrics.csv",
            "evaluation/figures/figure_3class_confusion_matrices.png",
        ],
    ),
]


def run_step(description: str, command: list[str], expected_files: list[str]) -> dict:
    """Run a single command, return a dict with status info."""
    start = time.time()

    print(f"\n{'=' * 72}")
    print(description)
    print(f"$ {' '.join(command)}")
    print("=" * 72)

    result = subprocess.run(command, capture_output=False)
    duration = time.time() - start

    missing = [f for f in expected_files if not Path(f).exists()]

    return {
        "description": description,
        "exit_code": result.returncode,
        "duration_sec": duration,
        "missing_files": missing,
    }


def main() -> int:
    print(f"{'#' * 72}")
    print("END-TO-END REPRODUCIBILITY CHECK")
    print(f"Working dir: {Path.cwd()}")
    print(f"{'#' * 72}")

    results: list[dict] = []
    for description, command, expected in STEPS:
        results.append(run_step(description, command, expected))

    # ------------------------------------------------------------------
    # Final report
    print(f"\n\n{'#' * 72}")
    print("FINAL REPORT")
    print(f"{'#' * 72}\n")

    n_ok = 0
    n_fail = 0
    total_time = 0.0

    for r in results:
        ok = (r["exit_code"] == 0) and not r["missing_files"]
        status = "[ OK ]" if ok else "[FAIL]"
        print(f"{status}  {r['description']:<55s}  {r['duration_sec']:>6.1f}s")
        if not ok:
            if r["exit_code"] != 0:
                print(f"        exit code: {r['exit_code']}")
            if r["missing_files"]:
                for f in r["missing_files"]:
                    print(f"        missing:   {f}")
            n_fail += 1
        else:
            n_ok += 1
        total_time += r["duration_sec"]

    print(f"\n{'-' * 72}")
    print(f"Steps OK:     {n_ok}/{len(results)}")
    print(f"Steps FAIL:   {n_fail}/{len(results)}")
    print(f"Total time:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print("-" * 72)

    if n_fail == 0:
        print("\nALL GREEN - pipeline is reproducible end-to-end.")
        return 0
    else:
        print(f"\n{n_fail} step(s) failed. Inspect output above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
