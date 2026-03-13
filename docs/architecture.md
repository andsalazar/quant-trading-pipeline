# System Architecture

## Overview

The system is a fully automated quantitative trading pipeline that runs daily via Windows Task Scheduler. It encompasses data ingestion from 7 asset classes, feature engineering (400+ features), ML model training with walk-forward validation, signal generation, and automated trade execution via Interactive Brokers.

## Data Flow

```
                         ┌──────────────────────┐
                         │   External Sources     │
                         │                        │
                         │  Polygon.io (Market)   │
                         │  FRED (Treasuries)     │
                         │  Web (Macro Events)    │
                         │  OpenAI (Sentiment)    │
                         └──────────┬─────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │   Step 1-7: Fetchers    │
                         │                        │
                         │  500 Equities + ETFs   │
                         │  6 FX Pairs            │
                         │  4 Treasury Maturities │
                         │  8 Futures Contracts   │
                         │  ATM Options + Greeks  │
                         │  FOMC + BLS Events     │
                         │  News Headlines        │
                         └──────────┬─────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │   Step 8: Database      │
                         │                        │
                         │  SQLite (10 tables)    │
                         │  Normalized schema     │
                         │  Incremental append    │
                         │  ~11 years of history  │
                         └──────────┬─────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │   Step 9: Features      │
                         │                        │
                         │  400+ ML features      │
                         │  LEFT JOIN all tables  │
                         │  Regime detection      │
                         │  Cross-asset flows     │
                         │  Momentum quality      │
                         │  Sentiment divergence  │
                         └──────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │                               │
         ┌──────────▼──────────┐         ┌──────────▼──────────┐
         │  Step 10: SPY       │         │  Stock Ensemble      │
         │  Timing Model       │         │  (Periodic Retrain)  │
         │                     │         │                      │
         │  LGB + XGB ensemble │         │  XGB + LGB + CB      │
         │  Next-day direction │         │  Per-symbol WF       │
         │  BUY / CASH signal  │         │  10-day relative     │
         │  Daily automated    │         │  Meta-learner        │
         └──────────┬──────────┘         └──────────┬──────────┘
                    │                               │
         ┌──────────▼──────────┐         ┌──────────▼──────────┐
         │  Trade Execution    │         │  Rankings Output     │
         │                     │         │                      │
         │  IBKR TWS API       │         │  500 stocks ranked   │
         │  Fractional shares  │         │  Confidence scored   │
         │  Email alerts       │         │  Export for content  │
         └─────────────────────┘         └─────────────────────┘
```

## Database Schema

The SQLite database (`quant_trading_v2.db`) contains 10 normalized tables:

| Table | Rows | Columns | Description |
|-------|------|---------|-------------|
| `market_data` | ~1.3M | 8 | Daily OHLCV for ~500 equities |
| `etf_features` | ~2.7K | 20+ | SPY, QQQ, GLD, TLT, VXX returns |
| `currency_data` | ~2.7K | 30+ | FX rates + dollar strength + carry |
| `treasury_data` | ~2.7K | 15+ | Yield curve + spread features |
| `futures_data` | ~2.7K | 25+ | ES, CL, GC, DX, NG returns |
| `options_data` | ~2.5K | 40+ | P/C ratios, IV, Greeks, term structure |
| `macro_events` | ~500 | 8 | FOMC, BLS releases + impact scores |
| `event_features` | ~2.7K | 15 | Event proximity + density features |
| `sentiment_data` | ~2.7K | 20+ | GPT-scored news sentiment aggregates |
| `market_enhanced` | ~1.3M | 15+ | Technical indicators per stock |

## Walk-Forward Validation

All models use expanding-window walk-forward to prevent lookahead bias:

```
Training window grows →

│████████████████████│     test 2021     │
│████████████████████████████│  test 2022│
│████████████████████████████████████│ 23│
│████████████████████████████████████████│
│████████████████████████████████████████│

← 10-day embargo gap prevents target leakage →
```

For the 10-day stock-picking model, embargo gaps equal the prediction horizon to prevent any overlap between training targets and test predictions.

## SPY Timing Model

The production model predicts next-day SPY direction using:
- **27 cross-asset features**: ETF returns, futures, FX, macro breadth, yield curve
- **20 SPY technicals**: Multi-period returns, vol ratios, SMA ratios, RSI, Bollinger, drawdown
- **20 regime features**: VIX regime, yield curve state, risk-on/off, trend, vol persistence
- **16 cross-asset flow features**: Bond-equity rotation, safe haven, commodity-currency alignment

Models: LightGBM + XGBoost simple average ensemble.

Hyperparameters optimized via Optuna TPE (100 trials per model) using walk-forward AUC as the objective — not simple cross-validation.

## Execution Model

```
Market closes 4:00 PM ET
    │
    ▼
Polygon daily bars available (~5 PM)
    │
    ▼
Pipeline runs: fetch → features → predict
    │
    ▼
Signal generated: BUY or CASH (with probability)
    │
    ▼
Execute at NEXT DAY'S OPEN
    │
    ▼
Track outcome → update accuracy log
```

This is fully realistic — no same-day execution with close data.
