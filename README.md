# BTC ML Pipeline — directional prediction with walk-forward validation

> Engineering thesis project, PJATK Warsaw, 2026.
> Topic: *Application of machine learning methods for forecasting market signals on BTC data.*
> Author: Mateusz Jastrzębski (s27397). Supervisor: dr Adam Szmigielski.

End-to-end machine-learning pipeline for predicting the direction of 5-day-ahead returns
of BTC/USDT, with rigorous time-series validation, probability calibration, regime-stability
analysis, and full experiment tracking via MLflow.

---

## Highlights

- **5 models** compared head-to-head: two naive persistence baselines (1-day and 5-day),
  Logistic Regression, Random Forest, XGBoost.
- **Walk-forward validation** with `TimeSeriesSplit(n_splits=5)` — no information leakage
  from the future.
- **Probability calibration** (isotonic) and **decision-threshold tuning** — boosts
  calibrated F1 by ~39-57% over the default 0.5 threshold (threshold selected per fold
  on an internal validation split, never the test set).
- **Regime-stability analysis** across 5 manually defined market phases
  (pre-2020, bull 2020-21, bear 2022, recovery 2023, bull 2024+).
- **Three-class classification** extension (strong-down / weak-neutral / strong-up).
- **MLflow tracking** with parent-child run hierarchy: ~70 runs across 3 experiments.

## Headline results

| Model | Accuracy | F1 | ROC-AUC | Log-loss |
| --- | --- | --- | --- | --- |
| Persistence1d (naive) | 0.47 | 0.49 | 0.47 | 2.42 |
| Persistence5d (naive) | 0.48 | 0.51 | 0.48 | 2.38 |
| LogReg | 0.49 | 0.34 | 0.53 | 1.27 |
| **RandomForest** | **0.50** | 0.41 | **0.55** | **0.72** |
| XGBoost | 0.50 | 0.38 | 0.54 | 1.01 |

After threshold tuning, calibrated F1 of the ML models reaches **0.61-0.63**.

## Project structure

```
btc-ml-pipeline/
├── data/
│   ├── fetch_binance.py           # OHLCV downloader (Binance public API)
│   ├── raw/                       # downloaded parquet files
│   └── processed/                 # engineered features + target
├── features/
│   ├── engineer.py                # binary target + 10 features (returns, momentum, vol, RSI, SMA)
│   └── engineer_3class.py         # three-class target generator
├── models/
│   ├── baseline.py                # Logistic Regression with StandardScaler in Pipeline
│   ├── random_forest.py           # Random Forest classifier (regularised)
│   ├── xgboost_model.py           # XGBoost classifier
│   └── persistence.py             # naive baselines (1d and 5d)
├── evaluation/
│   ├── walk_forward.py            # TimeSeriesSplit walk-forward evaluator
│   ├── calibration.py             # isotonic calibration + threshold tuning
│   ├── trading_metrics.py         # equity curve with transaction costs
│   ├── regime_analysis.py         # per-regime stability analysis
│   ├── plots.py                   # comparison figures
│   ├── results/                   # CSV + parquet outputs (.gitignore? see below)
│   └── figures/                   # PNG outputs
├── scripts/
│   ├── run_all_models.py          # offline runner without MLflow
│   ├── run_with_mlflow.py         # main pipeline with MLflow logging
│   ├── run_tuning.py              # 21-config hyperparameter grid
│   └── run_3class.py              # multi-class classification experiment
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA: distributions, target shift, regimes
│   └── 02_model_comparison.ipynb  # consolidated dashboard of all results
├── configs/                       # (placeholder for future YAML configs)
├── mlruns/                        # MLflow file store (auto-created, gitignored)
├── requirements.txt
└── README.md
```

## How to run from scratch

```powershell
# 1. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (~10 seconds)
python data/fetch_binance.py

# 4. Engineer binary features and target
python -m features.engineer

# 5. Run the main pipeline (5 models, walk-forward, MLflow tracking)
python -m scripts.run_with_mlflow

# 6. Hyperparameter tuning — 21 configurations
python -m scripts.run_tuning

# 7. Calibration + threshold tuning + equity curves
python -m evaluation.calibration
python -m evaluation.trading_metrics

# 8. Per-regime stability analysis
python -m evaluation.regime_analysis

# 9. Three-class extension
python -m features.engineer_3class
python -m scripts.run_3class

# 10. Generate comparison figures
python -m evaluation.plots
```

Total runtime on a modern laptop: ~10-15 minutes including all tuning runs.

## Inspect results

### MLflow UI

```powershell
mlflow ui --backend-store-uri "file:///$(Resolve-Path mlruns)"
```

Then open `http://127.0.0.1:5000` and pick one of the three experiments:
- `btc_ml_pipeline` — main run with all 5 models, 31 runs
- `btc_ml_tuning` — hyperparameter grid, 21 runs
- `btc_ml_pipeline_3class` — multi-class extension, ~18 runs

### Notebooks

```powershell
jupyter notebook notebooks/02_model_comparison.ipynb
```

`02_model_comparison.ipynb` consolidates every result table and figure produced
by the pipeline into a single readable narrative — designed as a "demo dashboard"
for thesis defence.

### Raw artefacts

- `evaluation/results/` — CSVs (per-fold metrics, calibration summary,
  regime metrics, trading metrics, 3-class summary, tuning summary)
- `evaluation/figures/` — PNG figures (model comparison, calibration curves,
  threshold tuning, equity curves, regime metrics, confusion matrices,
  probability distributions)

## Methodology highlights

### Walk-forward validation

Standard random K-fold leaks future information into training. We use
`TimeSeriesSplit(n_splits=5)` which expands the training window
chronologically: each fold's test set is strictly newer than its training set.
Preprocessing (scaling) is wrapped in an sklearn `Pipeline` so that
`StandardScaler.fit()` runs only on the training half of each fold. Reference:
Jansen (2020), *Machine Learning for Algorithmic Trading*, Ch. 6, pp. 168-170.

### Probability calibration

Tree-ensemble models in particular tend to return uncalibrated probabilities
that concentrate near 0 and 1 (visible in the probability distribution figure).
We apply isotonic calibration via `sklearn.calibration.CalibratedClassifierCV`
inside each walk-forward fold. This trades a small amount of global ROC-AUC
for substantially better log-loss — typically 24-38% reduction across models.

### Decision-threshold tuning

The default 0.5 threshold is rarely optimal in finance. Inside each walk-forward
fold we sweep thresholds in `[0.30, 0.70]` on an internal validation split carved
from the last 20% of that fold's training window — never the test set — and pick
the one that maximises F1. This step improves calibrated F1 by **~39-57%** across
the three ML models.

### Regime-stability analysis

We split the test predictions into five manually defined regimes based on
known BTC market phases and recompute all metrics within each. Shows that
model performance is **strongly regime-dependent** — a known challenge with
non-stationary financial time series (López de Prado 2018, Ch. 7).

## What's intentionally out of scope

This is a focused engineering thesis, not an exhaustive research paper.
The following are deliberately not included:

- Deep learning (LSTM, Transformer architectures)
- Sentiment / on-chain alternative data
- Portfolio context, position sizing, risk management
- Realistic backtest engine (Zipline, backtrader) — only an illustrative
  equity curve with a flat 10 bps transaction cost
- Multi-asset extension (ETH, SOL, ...)

These constitute natural directions for follow-up research and are listed
in the conclusions of the thesis.

## References

Primary literature:

- Jansen, S. (2020). *Machine Learning for Algorithmic Trading*, 2nd edition.
  Packt Publishing.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- scikit-learn User Guide, *Probability calibration*:
  https://scikit-learn.org/stable/modules/calibration.html
- MLflow documentation: https://mlflow.org/docs/latest/

## License

Educational use only. This project was prepared as the practical part of an
engineering thesis at Polish-Japanese Academy of Information Technology.
