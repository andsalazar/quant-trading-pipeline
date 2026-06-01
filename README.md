# Quantitative Trading Pipeline

**End-to-end machine learning system for equity market prediction and automated execution.**

Built as a solo project over 6+ months, this pipeline fetches multi-asset data from 7 sources, engineers 400+ features, trains gradient-boosted ensembles with walk-forward validation, and generates daily trading signals — with execution running on a headless, containerized cloud stack (GCP VM + IB Gateway) decoupled from the local signal-generation machine.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-4.5-blue)](https://optuna.org/)
[![IBKR](https://img.shields.io/badge/IBKR-TWS%20API-red)](https://interactivebrokers.com/)
[![GCP](https://img.shields.io/badge/Google%20Cloud-Compute%20%2B%20Storage-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY PIPELINE (Automated)                   │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Step 1-7 │→│  Step 8   │→│  Step 9   │→│  Step 10a/b/c │  │
│  │  Fetch    │  │  Database │  │  Feature  │  │  Track →      │  │
│  │  7 Sources│  │  Update   │  │  Engineer │  │  Predict →    │  │
│  │           │  │           │  │           │  │  Reconcile    │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│       │                           │               │             │
│  Market Data              400+ Features       BUY / CASH        │
│  Currencies               Regime Detection    Next-Day Signal   │
│  Treasuries               Cross-Asset Flows   + Slippage /      │
│  Futures                  Momentum Quality      Alpha-Capture   │
│  Options                  Sentiment Divergence  Reconciliation  │
│  Macro Events             Technical Indicators                  │
│  News Sentiment           Interaction Features                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              STOCK ENSEMBLE (Periodic Retrain)                  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  XGBoost  │  │ LightGBM │  │ CatBoost  │  │ Meta-Learner  │  │
│  │  (GPU)    │  │          │  │           │  │ (Stacking)    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       └──────────────┴─────────────┘                │           │
│              Per-Symbol Walk-Forward          Final Rankings     │
│              10-Day Market-Relative           ~500 Stocks       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Results

### SPY Timing Model (Next-Day Direction)
Walk-forward out-of-sample results on 5 yearly folds (2021–2025):

| Year | Ensemble AUC | Accuracy | Backtest Return | SPY Buy & Hold |
|------|:------------:|:--------:|:---------------:|:--------------:|
| 2021 |    0.785     |  70.9%   |    +35.1%       |    +27.3%      |
| 2022 |    0.794     |  71.2%   |  **+24.8%**     |  **-18.8%**    |
| 2023 |    0.793     |  70.9%   |    +32.9%       |    +23.6%      |
| 2024 |    0.638     |  61.5%   |    +23.4%       |    +23.2%      |
| 2025 |    0.588     |  62.3%   |    +25.4%       |    +16.0%      |
| **Avg** | **0.720** | **67.4%** |                |                |

- **5-year backtest**: $2,000 → $6,851 (Sharpe 2.35, max drawdown -14.3%)
- **Realistic execution**: next-open fills, IBKR commissions, 1 bps slippage
- **BUY precision**: 69.8% | **CASH precision**: 63.7%
- Profitable in every year, including 2022 bear market

### Stock-Picking Ensemble (10-Day, 500 Stocks)
Per-symbol walk-forward across ~500 equities:

| Model | Per-Symbol AUC | Walk-Forward Folds |
|-------|:--------------:|:------------------:|
| XGBoost (GPU) | 0.524 | 5 |
| LightGBM | 0.524 | 5 |
| CatBoost | 0.521 | 5 |
| Meta-Learner | 0.508 | Holdout |

> Cross-sectional stock ranking remains near the noise floor — an honest result
> that reflects the difficulty of the problem, not a lack of rigor.

### Deep Learning Experiments (112.5 GPU Hours)
Systematic double-descent investigation across 3 architectures:

| Architecture | Models Trained | Best Val AUC | GPU Hours |
|:------------:|:--------------:|:------------:|:---------:|
| MLP | 5 (32→4096 neurons) | 0.640 | 37.5 |
| 1D-CNN | 4 (8→128 filters) | 0.638 | 37.5 |
| LSTM | 4 (32→512 hidden) | 0.641 | 37.5 |

**Finding**: No double-descent observed on financial data. All architectures
converge to the same ~0.64 AUC ceiling regardless of model complexity — a strong
information-theoretic bound suggesting the features contain approximately 1 bit
of learnable signal per prediction.

---

## Project Structure

```
quant-trading-pipeline/
│
├── src/
│   ├── data_pipeline/           # Step 1-7: Multi-asset data ingestion
│   │   ├── fetch_market/        #   OHLCV for ~500 equities + ETFs
│   │   ├── fetch_currencies/    #   6 FX pairs → dollar strength, carry signals
│   │   ├── fetch_treasuries/    #   Yield curve (2Y, 5Y, 10Y, 30Y) from FRED
│   │   ├── fetch_futures/       #   ES, CL, GC, DX, etc. continuous contracts
│   │   ├── fetch_options/       #   ATM options, put/call ratios, Greeks
│   │   ├── fetch_events/        #   FOMC, BLS economic calendar
│   │   └── fetch_sentiment/     #   News headlines → GPT-3.5 sentiment scoring
│   │
│   ├── data_pipeline/database/  # Step 8: Normalized SQLite database layer
│   │
│   ├── feature_engineering/     # Step 9: 400+ ML features from raw data
│   │   └── feature_engineer.py  #   Technical, momentum, regime, cross-asset,
│   │                            #   sentiment divergence, interaction features
│   │
│   ├── model_training/
│   │   ├── spy_timing/          # Step 10: Next-day SPY direction model
│   │   │   ├── train.py         #   LGB + XGB ensemble, walk-forward backtest
│   │   │   ├── predict.py       #   Daily BUY/CASH signal generation
│   │   │   ├── track.py         #   Live accuracy tracking + degradation alerts
│   │   │   ├── promote.py       #   Atomic model swap + isotonic recalibration
│   │   │   │                    #   (AUC guardrail aborts on >3pt regression)
│   │   │   └── optimize.py      #   Optuna hyperparameter optimization
│   │   │
│   │   ├── stock_ensemble/      # 10-day stock-picking ensemble
│   │   │   ├── train_xgb.py     #   XGBoost (GPU) per-symbol walk-forward
│   │   │   ├── train_lgb.py     #   LightGBM per-symbol walk-forward
│   │   │   ├── train_catboost.py#   CatBoost per-symbol walk-forward
│   │   │   ├── meta_learner.py  #   Stacking ensemble with honest OOS eval
│   │   │   └── predict.py       #   Generate production rankings
│   │   │
│   │   ├── backtesting/         # Realistic backtest engine
│   │   └── hyperparameter_optimization/
│   │
│   ├── trade_execution/         # IBKR TWS API integration
│   │   ├── ibkr_trader.py       #   Confidence-weighted position sizing
│   │   ├── spy_timing_trader.py #   SPY fractional shares + email alerts
│   │   └── reconcile_executions.py  # Three-way join: prediction × trade × fill
│   │                            #   (slippage_bps, alpha_capture_pct, alerts)
│   │
│   ├── pipeline_orchestration/  # Daily automation + Task Scheduler
│   │
│   └── dl_experiments/          # Deep learning double-descent research
│       ├── data_loader.py       #   PyTorch DataLoader (flat + sequenced)
│       ├── trainer.py           #   MLP / 1D-CNN / LSTM trainer
│       ├── analyze_results.py   #   Cross-architecture comparison
│       └── runners/             #   Experiment launchers (overnight, sweep)
│
├── deploy/                      # Headless cloud execution stack (GCP)
│   ├── docker-compose.yml       #   IB Gateway + IBC + executor services
│   ├── Dockerfile               #   Containerized executor (Python 3.11)
│   ├── entrypoint.sh            #   In-container scheduler loop (09:31 America/New_York)
│   ├── gcs_pull.py              #   VM-side signal download from GCS
│   └── README.md                #   Full VM + bucket + service-account setup guide
│
├── docs/
│   ├── architecture.md          # Detailed system design
│   ├── feature_taxonomy.md      # All 400+ features documented
│   ├── spy_timing_results.md    # Full backtest results + equity curves
│   ├── execution_reconciliation.md  # Step 10c: model × trader × broker join
│   └── dl_experiments.md        # Double-descent findings
│
├── config_sample.py             # Configuration template (no secrets)
├── .gitignore                   # Excludes all data, models, logs, keys
├── LICENSE                      # MIT
└── README.md
```

## Feature Engineering (400+ Features)

Features are computed from 10 normalized database tables spanning 7 asset classes:

| Category | Count | Examples |
|----------|:-----:|---------|
| **Technical Indicators** | ~40 | SMA ratios, RSI, Bollinger position, ATR |
| **Momentum** | ~30 | Multi-period returns (1d–60d), momentum consistency |
| **Volatility** | ~25 | Realized vol, vol-of-vol, GARCH-style persistence |
| **Cross-Asset** | ~30 | ETF returns (SPY, QQQ, GLD, TLT), futures, FX |
| **Regime Detection** | 20 | VIX regime, yield curve state, risk-on/off, trend |
| **Cross-Asset Flows** | 16 | Bond-equity rotation, safe haven demand, commodity breadth |
| **Momentum Quality** | 15 | Risk-adjusted momentum, Sortino, consistency scores |
| **Sentiment Divergence** | 15 | Price-sentiment divergence, exhaustion, contrarian |
| **Options-Derived** | ~20 | Put/call ratios, IV skew, term structure slope |
| **Macro/Calendar** | ~15 | FOMC proximity, event density, month/quarter |
| **Interaction Features** | ~50 | Cross-products of top features |
| **Lag Features** | 36 | Explicit temporal lags for tree model memory |
| **Normalization** | ~80 | Z-scores, percentile ranks, cross-sectional ranks |

All features are computed using only information available at prediction time (no lookahead bias).
Forward-looking columns (e.g., `IWM_Return`, `Market_Return_Mean`) are explicitly excluded.

## Technical Highlights

### Walk-Forward Validation
Every model uses expanding-window walk-forward with proper embargo gaps:
```
Fold 1: Train [2015–2020] → Test [2021]  (10-day gap)
Fold 2: Train [2015–2021] → Test [2022]  (10-day gap)
Fold 3: Train [2015–2022] → Test [2023]  (10-day gap)
Fold 4: Train [2015–2023] → Test [2024]  (10-day gap)
Fold 5: Train [2015–2024] → Test [2025]  (10-day gap)
```

### Realistic Backtesting
- **Execution**: Signals generated from close data → executed at next day's open
- **Commissions**: IBKR tiered (min $0.35, $0.0035/share, capped at 1%)
- **Slippage**: 1 basis point per side (conservative for SPY)
- **Capital**: $2,000 starting — no unrealistic $1M assumptions

### Bayesian Hyperparameter Optimization
Optuna TPE sampler with walk-forward as the objective (not simple train/test):
- 100 trials per model, 30-minute timeout
- Optimized both LightGBM and XGBoost independently
- Ensemble evaluated against baseline on all 5 folds
- Result: +0.74 AUC improvement, +1.7% accuracy improvement

### GPU Acceleration
- XGBoost: `gpu_hist` tree method with CUDA (RTX 4060 Laptop 8GB)
- PyTorch: CUDA-accelerated DL experiments
- Training time: ~55 seconds for full SPY model (5-fold + production)

## What This Project Demonstrates

1. **Production ML Engineering** — Not a Jupyter notebook; a fully automated pipeline
   that runs daily, handles failures gracefully, and generates actionable signals.

2. **Domain Expertise** — 7 asset classes, regime-aware features, realistic execution
   modeling with actual broker commissions and slippage.

3. **Statistical Rigor** — Walk-forward validation, embargo gaps, proper OOS evaluation.
   No in-sample metrics presented as results.

4. **Intellectual Honesty** — The stock-picking ensemble (~0.50 AUC) is included
   alongside the strong SPY model (~0.72 AUC). Real research includes failures.

5. **Deep Learning Research** — 112.5 GPU hours systematically testing whether neural
   architectures could beat tree models on this data. They couldn't — and that result
   is documented rather than hidden.

6. **Production Discipline** — The model is not the system. The system includes the
   monitoring and safety machinery that lets a live trading pipeline survive contact
   with the market (see next section).

7. **Cloud & DevOps** — Execution runs as a headless, containerized stack on a GCP
   VM (Docker Compose, IB Gateway + IBC auto-login, a scheduler loop, Cloud Storage hand-off,
   least-privilege service-account IAM) — decoupled from signal generation so the
   trade fires at market open whether or not the workstation is awake.

## Production Discipline

A live trading system fails in ways a backtest never sees. This pipeline includes
the operational scaffolding that turns a one-shot model into an asset that survives
months of unattended running:

| Discipline                  | Where                                       | What it catches                                              |
| --------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **Daily accuracy tracking** | `spy_timing/track.py` (Step 10a)            | Live AUC / hit-rate drift vs OOS backtest                    |
| **Daily reconciliation**    | `trade_execution/reconcile_executions.py` (Step 10c) | Execution failures, slippage drift, alpha-capture decay |
| **AUC promotion guardrail** | `spy_timing/promote.py`                     | Aborts model swap if new eval AUC drops >3pts vs last promote |
| **Atomic model swap**       | `spy_timing/promote.py`                     | Backs up production .pkl + calibration before overwrite      |
| **Isotonic recalibration**  | `spy_timing/promote.py`                     | Re-fits probability calibration on each promotion            |
| **Threshold re-tuning**     | `spy_timing/promote.py`                     | Sweeps 0.40–0.70, picks calib-Sharpe optimum                 |
| **Email alerts**            | trader + reconciler                         | Same-day notification on execution failure / slippage / capture drop |
| **Pipeline soft-fail**      | `pipeline_orchestration/daily_pipeline.py`  | Reconciler exits 0 even on alerts — never blocks tomorrow's signal |

See [`docs/execution_reconciliation.md`](docs/execution_reconciliation.md) for the
three-way reconciliation methodology (model says × trader does × broker fills).

## Cloud Execution Infrastructure

Signal *generation* runs on a local GPU workstation; trade *execution* runs in the
cloud. The two are deliberately decoupled so a laptop being asleep, offline, or
mid-reboot can never miss a market-open trade.

**The problem it solves.** Running the broker API against a desktop trading
terminal is fragile: the terminal restarts on its own daily schedule, the API
socket silently disables itself, and the whole thing depends on one machine
staying awake. The fix was to move execution onto an always-on, headless,
containerized stack with no GUI in the loop.

```
  LOCAL WORKSTATION (GPU)                 GOOGLE CLOUD PLATFORM
  ┌────────────────────────┐              ┌──────────────────────────────────┐
  │ Daily pipeline (5:30pm) │              │  Compute Engine VM (always-on)   │
  │  → trains / predicts    │   signal     │  ┌────────────────────────────┐  │
  │  → BUY / CASH signal     │  ──CSV──►    │  │ Docker Compose             │  │
  │  → uploads to GCS        │   (GCS)      │  │  ┌──────────┐ ┌──────────┐ │  │
  └────────────────────────┘   bucket      │  │  │ IB Gateway│ │ Executor │ │  │
                                            │  │  │ + IBC     │◄│  (loop)  │ │  │
  ┌────────────────────────┐               │  │  │ (headless)│ │ pulls    │ │  │
  │ Cloud Storage bucket    │  ◄──pull──    │  │  └────┬─────┘ │ signal,  │ │  │
  │  versioned signal hand- │   09:31 ET    │  │   socat relay │ trades   │ │  │
  │  off, least-priv access │               │  │       └───────┴──────────┘ │  │
  └────────────────────────┘               │  └────────────────────────────┘  │
                                            │   restart:always · auto-relogin  │
                                            └──────────────────────────────────┘
```

| Concern | Approach |
|---------|----------|
| **Always-on execution** | GCP Compute Engine VM (Debian 12) — no dependence on the laptop being awake |
| **Headless broker login** | IB Gateway + IBC auto-login in a container — no GUI, no manual click-through on restart |
| **Service isolation** | Docker Compose: gateway and executor in separate containers, internal-only API ports (never published to the host) |
| **Self-healing** | `restart: always` + container healthchecks + IBC re-authentication on the broker's daily restart |
| **Scheduled trigger** | A self-contained scheduler loop in the executor fires at market open in `America/New_York`, timezone-correct year-round |
| **Decoupled hand-off** | Google Cloud Storage as the signal bus — generation and execution share nothing but a versioned CSV |
| **Least-privilege auth** | Dedicated service account scoped to a single bucket; key mounted as a secret, never committed |
| **Staleness guard** | Executor refuses signals older than N business days (weekend-aware) so a missed run can't fire a stale trade |

See [`deploy/README.md`](deploy/README.md) for the full provisioning guide
(VM, bucket, service account, container build). No credentials are stored in the
repository — all secrets live in untracked environment files on the VM.

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | Polygon.io API, FRED, web scraping, SQLite |
| **Features** | pandas, NumPy, custom engineering pipeline |
| **ML** | XGBoost (GPU), LightGBM, CatBoost, scikit-learn |
| **DL** | PyTorch + CUDA |
| **Optimization** | Optuna (TPE sampler) |
| **Execution** | Interactive Brokers Gateway API (ib_insync), IBC headless auto-login |
| **Cloud / DevOps** | Google Cloud Platform (Compute Engine, Cloud Storage), Docker, Docker Compose, Linux (Debian), service-account IAM |
| **Automation** | Windows Task Scheduler (signal gen), in-container scheduler loop (execution), batch scripts |
| **Sentiment** | OpenAI GPT-3.5 via API |

## Setup

```bash
# Clone
git clone https://github.com/andsalazar/quant-trading-pipeline.git
cd quant-trading-pipeline

# Environment
conda create -n quant python=3.10
conda activate quant
pip install xgboost lightgbm catboost scikit-learn optuna pandas numpy

# GPU (optional, for XGBoost GPU + PyTorch)
pip install xgboost --upgrade  # Built with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Configuration
cp config_sample.py config.py
# Edit config.py with your paths and API keys

# Data (not included — requires Polygon.io subscription)
export POLYGON_API_KEY="your_key_here"
```

> **Note**: Historical market data is not included in this repository. You'll need
> a [Polygon.io](https://polygon.io) subscription (or similar data provider) to
> run the full pipeline. The code and architecture are the focus of this repo.

## Related Projects

- **[Economic Research Paper](https://github.com/andsalazar)** — Companion academic
  project applying econometric methods to related market microstructure questions.

## Disclaimer

> **This project is for educational and portfolio demonstration purposes only — it is not financial advice.**
> Signal thresholds, position sizing, and capital allocation shown in the code are
> illustrative examples used for backtesting and development. They do not represent
> optimized or recommended trading parameters. Past backtest performance does not
> guarantee future results. Use at your own risk.

## License

MIT — See [LICENSE](LICENSE) for details.

---

*Built by [Andy Salazar](https://github.com/andsalazar) — 2025–2026*
