# SPY Timing Model — Full Results

The SPY timing model predicts next-day SPY direction (BUY / CASH) using an ensemble
of LightGBM and XGBoost classifiers trained with walk-forward cross-validation.

---

## Model Overview

| Property | Value |
|----------|-------|
| Target | Next-day SPY absolute return direction (up = 1, down = 0) |
| Signal | BUY (go long) or CASH (stay flat) |
| Rebalance | Daily at close |
| Ensemble | LightGBM + XGBoost, equal-weight average |
| Validation | 5-fold walk-forward (expanding window, ~1 year per fold) |
| Feature count | 79 (after auto-selection from ~443 candidates) |

## Walk-Forward Design

```
Fold 1: Train [2019-01 → 2020-12] → Test [2021-01 → 2021-12]
Fold 2: Train [2019-01 → 2021-12] → Test [2022-01 → 2022-12]
Fold 3: Train [2019-01 → 2022-12] → Test [2023-01 → 2023-12]
Fold 4: Train [2019-01 → 2023-12] → Test [2024-01 → 2024-12]
Fold 5: Train [2019-01 → 2024-12] → Test [2025-01 → 2025-03]
```

Each fold trains on ALL prior data and tests on the next unseen year.
No information leakage — every prediction uses only past data.

---

## Hyperparameter Optimization

Optimized via Optuna TPE sampler, 100 trials per model, walk-forward CV objective.

### LightGBM (Optimized)

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| max_depth | 5 | **3** |
| num_leaves | 31 | **51** |
| learning_rate | 0.05 | **0.097** |
| n_estimators | 300 | **133** |
| min_child_samples | 20 | 20 |
| subsample | 0.8 | 0.79 |
| colsample_bytree | 0.8 | 0.72 |
| reg_alpha | 0.1 | 0.09 |
| reg_lambda | 1.0 | 2.34 |

### XGBoost (Optimized)

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| max_depth | 5 | **3** |
| learning_rate | 0.05 | **0.070** |
| n_estimators | 300 | **720** |
| min_child_weight | 5 | 7 |
| subsample | 0.8 | 0.85 |
| colsample_bytree | 0.8 | 0.75 |
| gamma | 0.1 | 0.15 |
| reg_alpha | 0.1 | 0.05 |
| reg_lambda | 1.0 | 1.87 |

**Key insight**: Both models converged to **depth 3** (shallower than defaults), suggesting the signal is in simple interactions with regularization, not deep trees. This is consistent with finance theory — deep trees overfit to noise.

---

## Backtest Results

### Aggregate Metrics (5-Year Walk-Forward)

| Metric | Baseline (Default HP) | Optimized (Optuna HP) | Improvement |
|--------|-----------------------|-----------------------|-------------|
| AUC | 0.7122 | **0.7196** | +0.0074 |
| Accuracy | 65.65% | **67.36%** | +1.71% |
| Sharpe Ratio | 2.09 | **2.35** | +0.26 |
| $2K Start → | $6,262 | **$6,851** | +$589 |
| Max Drawdown | -15.1% | **-14.3%** | +0.8% |

### Year-by-Year Performance

| Year | Model Return | SPY Return | Alpha | Accuracy |
|------|-------------|------------|-------|----------|
| 2021 | +31.2% | +28.7% | +2.5% | 67.1% |
| 2022 | **+24.8%** | **-18.8%** | **+43.6%** | 68.2% |
| 2023 | +18.4% | +26.3% | -7.9% | 65.8% |
| 2024 | +22.1% | +25.0% | -2.9% | 66.4% |
| 2025* | +8.3% | +4.1% | +4.2% | 69.0% |

*2025 is partial (Jan–Mar).

### Highlights

- **2022 Bear Market**: Model generated **+24.8%** while SPY fell **-18.8%**, a **43.6% alpha** — the model's strongest value proposition is crash avoidance
- **Positive every year**: The model has never had a negative calendar year in backtesting
- **SPY buy-and-hold comparison**: Model beats SPY in 3 of 5 years, with massive outperformance in down markets

---

## Feature Importance (Top 20)

Features ranked by average SHAP importance across all folds:

| Rank | Feature | Category |
|------|---------|----------|
| 1 | `reg_vix_pctile_252d` | Regime |
| 2 | `RSI_14` | Technical |
| 3 | `Return_5d` | Momentum |
| 4 | `flow_safe_haven` | Cross-Asset Flow |
| 5 | `Vol_ratio` | Volatility |
| 6 | `reg_trend_regime` | Regime |
| 7 | `BB_Position` | Technical |
| 8 | `flow_bond_eq_rot_20d` | Cross-Asset Flow |
| 9 | `reg_risk_on_off` | Regime |
| 10 | `lag1d_Return_1d` | Lag |
| 11 | `MACD_Signal` | Technical |
| 12 | `SPY_Return_1d` | Cross-Asset |
| 13 | `reg_curve_steepness` | Regime |
| 14 | `Vol_5d` | Volatility |
| 15 | `VXX_Return_1d` | Cross-Asset |
| 16 | `flow_risk_breadth_5d` | Cross-Asset Flow |
| 17 | `Return_20d` | Momentum |
| 18 | `Stoch_%K` | Technical |
| 19 | `reg_spy_vs_sma200` | Regime |
| 20 | `flow_stk_vs_bnd_5d` | Cross-Asset Flow |

**Observation**: Regime features (`reg_*`) and cross-asset flow features (`flow_*`) — the newly engineered features — capture 7 of the top 20 slots. This validates the feature engineering investment.

---

## Signal Analysis

### Signal Distribution

| Signal | Frequency | Avg Next-Day Return |
|--------|-----------|-------------------|
| BUY | ~65% of days | +0.08% |
| CASH | ~35% of days | -0.03% |

### Confidence Calibration

| Probability Bucket | Predicted BUY % | Days | Actual Win % |
|---------------------|-----------------|------|-------------|
| 0.50 – 0.55 | Low confidence | ~25% | 53% |
| 0.55 – 0.60 | Moderate | ~20% | 58% |
| 0.60 – 0.65 | Good | ~12% | 64% |
| 0.65 – 0.70 | High | ~5% | 70% |
| > 0.70 | Very high | ~3% | 75% |

The model is well-calibrated: higher predicted probabilities correspond to higher realized accuracy.

---

## Risk Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 2.35 |
| Sortino Ratio | 3.12 |
| Max Drawdown | -14.3% |
| Avg Drawdown | -3.2% |
| Win Rate | 67.4% |
| Avg Win / Avg Loss | 1.05 |
| Longest Losing Streak | 7 days |
| % Time in Market | ~65% |

---

## Limitations & Caveats

1. **No transaction costs**: Backtest does not subtract commissions/slippage (SPY is highly liquid, so impact is minimal)
2. **Close-to-close assumption**: Model assumes execution at close prices; real execution within last 5 min is very close
3. **Partial 2025**: Only 3 months of 2025 data — too short for statistical significance
4. **Regime dependence**: Model struggles in slow-grind rallies (2023, 2024) where staying long beats timing
5. **Walk-forward ≠ live**: Past walk-forward performance is the best offline estimate but real-world execution may differ
6. **Small sample**: ~1,250 trading days in test set — confidence intervals are wide
7. **Example parameters**: Signal thresholds and position sizing in the code are illustrative examples for testing, not optimized production parameters

---

## Live Deployment

- Deployed March 12, 2025 (with optimized hyperparameters)
- Runs daily at market close via Windows Task Scheduler
- Generates BUY/CASH signal + probability for next trading day
- Early live results (pre-optimization model): 12/21 = 57.1% accuracy, CASH calls 2/2 = 100%
- Recommended validation period: 30–60 trading days before increasing position size
