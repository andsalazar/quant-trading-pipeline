# Feature Taxonomy

Complete catalog of the 400+ features computed from 10 database tables across 7 asset classes.
All features use only information available at prediction time (no lookahead bias).

---

## 1. Technical Indicators (~40 features)

Computed per-stock from daily OHLCV data.

| Feature | Formula | Lookback |
|---------|---------|----------|
| `SMA_5_ratio` | Close / SMA(5) | 5d |
| `SMA_10_ratio` | Close / SMA(10) | 10d |
| `SMA_20_ratio` | Close / SMA(20) | 20d |
| `SMA_50_ratio` | Close / SMA(50) | 50d |
| `RSI_14` | Wilder RSI | 14d |
| `BB_Position` | (Close - BB_lower) / (BB_upper - BB_lower) | 20d |
| `ATR_14` | True Range rolling mean | 14d |
| `MACD_Signal` | EMA(12) - EMA(26) | 26d |
| `Stoch_%K` | Stochastic oscillator | 14d |
| `OBV_Change` | Volume direction momentum | 20d |
| ... | Additional technical overlays | Various |

## 2. Momentum Features (~30)

Multi-horizon return features and momentum quality metrics.

| Feature | Description |
|---------|-------------|
| `Return_1d` to `Return_60d` | Rolling returns (1, 5, 10, 20, 60 day) |
| `Return_rank_20d` | Cross-sectional percentile rank |
| `Momentum_consistency` | % of positive days in lookback |
| `Risk_adj_momentum` | Return / volatility |
| `Momentum_acceleration` | Short-term vs long-term momentum |

## 3. Volatility Features (~25)

| Feature | Description |
|---------|-------------|
| `Vol_5d` to `Vol_60d` | Realized volatility at multiple windows |
| `Vol_ratio` | Short/long vol (vol regime detection) |
| `Vol_of_vol` | Standard deviation of rolling volatility |
| `Vol_percentile_252d` | Yearly volatility percentile rank |
| `Idiosyncratic_vol` | Vol after removing market factor |

## 4. Cross-Asset Features (~30)

Same values for all stocks on a given date — market-level signals.

| Feature | Source |
|---------|--------|
| `SPY_Return_1d`, `QQQ_Return_1d` | Equity ETFs |
| `GLD_Return_1d`, `TLT_Return_1d` | Safe haven ETFs |
| `VXX_Return_1d` | Volatility ETF |
| `ES_Return_1d`, `CL_Return_1d` | Futures (E-mini, Crude) |
| `DX_Return_1d` | Dollar index futures |
| `EUR_USD_return`, `USD_JPY_return` | FX pairs |
| `Market_Return_Std` | Cross-stock return dispersion |
| `Stocks_Above_SMA20` | Market breadth indicator |
| `Yield_Curve_Momentum` | Treasury curve dynamics |

## 5. Regime Detection Features (20, prefix `reg_`)

Classify the current market environment into discrete or continuous states.

| Feature | Description |
|---------|-------------|
| `reg_vix_pctile_252d` | VIX percentile (0–1) over past year |
| `reg_vix_momentum_5d` | Short-term VIX change |
| `reg_vix_regime` | 0=Low, 1=Normal, 2=High volatility |
| `reg_curve_steepness` | 10Y - 2Y treasury spread |
| `reg_curve_steep_chg_20d` | 20-day change in curve steepness |
| `reg_yield_curve_regime` | 0=Inverted, 1=Flat, 2=Steep |
| `reg_risk_on_off` | Composite risk appetite signal |
| `reg_risk_regime` | 0=Risk-off, 1=Neutral, 2=Risk-on |
| `reg_dollar_str_pctile` | Dollar strength percentile |
| `reg_dollar_regime` | 0=Weak, 1=Neutral, 2=Strong dollar |
| `reg_spy_vs_sma50` | SPY distance from 50-day SMA (%) |
| `reg_spy_vs_sma200` | SPY distance from 200-day SMA (%) |
| `reg_spy_trend_str` | SMA50 / SMA200 ratio |
| `reg_trend_regime` | 0=Downtrend, 1=Range, 2=Uptrend |
| `reg_spy_rvol_20d` | SPY 20-day realized vol (annualized) |
| `reg_vol_persistence` | Autocorrelation of volatility |
| `reg_vol_corr_stk_bnd` | Stock-bond vol correlation (63d) |

## 6. Cross-Asset Flow Features (16, prefix `flow_`)

Capital rotation signals across asset classes.

| Feature | Description |
|---------|-------------|
| `flow_bond_eq_rot_5d` | Bond-equity relative momentum (5d) |
| `flow_bond_eq_rot_20d` | Bond-equity relative momentum (20d) |
| `flow_rot_str_5d` | Rotation strength (absolute) |
| `flow_rot_pctile_252d` | Rotation percentile (yearly context) |
| `flow_safe_haven` | Composite safe-haven demand (JPY + Gold + Bonds) |
| `flow_safe_haven_str` | Absolute safe haven signal strength |
| `flow_oil_dollar_align` | Oil-dollar relationship (20d avg) |
| `flow_gold_dollar_align` | Gold-dollar inverse relationship |
| `flow_commodity_breadth` | Fraction of commodities rising |
| `flow_credit_eq_div` | Credit-equity divergence signal |
| `flow_risk_breadth_1d` | Risky assets breadth (1 day) |
| `flow_risk_breadth_5d` | Risky assets breadth (5 day avg) |
| `flow_risk_brdth_mom` | Change in risk breadth |
| `flow_stk_vs_bnd_5d` | Stock vs bond relative strength |
| `flow_rot_strength` | Dispersion of cross-class rotation |
| `flow_rot_leader` | Which asset class is leading (0/1/2) |

## 7. Momentum Quality Features (15, prefix `mqual_`)

Not just "is it going up" but "how reliable is the momentum."

| Feature | Description |
|---------|-------------|
| `mqual_risk_adj_20d` | 20-day return / 20-day vol |
| `mqual_risk_adj_60d` | 60-day risk-adjusted momentum |
| `mqual_sortino_20d` | Sortino ratio (downside vol only) |
| `mqual_consistency_20d` | % of positive return days |
| `mqual_consistency_60d` | Longer-term consistency |
| `mqual_max_dd_20d` | Max drawdown in 20-day window |
| `mqual_up_capture` | Avg gain on up days / overall vol |
| `mqual_down_capture` | Avg loss on down days / overall vol |
| ... | Additional quality-adjusted momentum |

## 8. Sentiment Divergence Features (15, prefix `sdiv_`)

When price and sentiment disagree, something interesting is happening.

| Feature | Description |
|---------|-------------|
| `sdiv_price_sent_div_5d` | Price return - sentiment change (5d) |
| `sdiv_price_sent_div_20d` | Longer-term price-sentiment gap |
| `sdiv_exhaustion_up` | High price + falling sentiment |
| `sdiv_exhaustion_down` | Low price + improving sentiment |
| `sdiv_contrarian` | Extreme sentiment reversal signal |
| `sdiv_sent_momentum_5d` | 5-day sentiment trend |
| `sdiv_sent_momentum_20d` | 20-day sentiment trend |
| `sdiv_sent_dispersion` | Cross-stock sentiment disagreement |
| ... | Additional divergence metrics |

## 9. Options-Derived Features (~20)

| Feature | Description |
|---------|-------------|
| `put_call_ratio` | Daily P/C volume ratio |
| `iv_rank_252d` | IV percentile over past year |
| `iv_skew` | OTM put vs call IV difference |
| `term_structure_slope` | Near vs far IV ratio |
| `option_volume_ratio` | Options vs stock volume |
| `gamma_exposure` | Market maker gamma positioning |

## 10. Macro / Calendar Features (~15)

| Feature | Description |
|---------|-------------|
| `fomc_proximity` | Days to next FOMC meeting |
| `event_density` | Count of macro events this week |
| `bls_proximity` | Days to next BLS release |
| `month`, `quarter` | Calendar seasonality |
| `day_of_week` | Intraweek patterns |

## 11. Interaction Features (~50)

Cross-products of top features selected by importance:

```python
# Example: momentum × volatility interactions
Return_20d × Vol_ratio
RSI_14 × VXX_Return_1d
Stocks_Above_SMA20 × reg_risk_on_off
```

## 12. Lag Features (36, prefix `lag*d_`)

Explicit temporal lags for tree models (which have no built-in memory):

| Feature | Description |
|---------|-------------|
| `lag1d_Return_1d` | Yesterday's daily return |
| `lag5d_Return_1d` | Return from 5 days ago |
| `lag1d_Vol_5d` | Yesterday's short-term vol |
| `lag5d_RSI_14` | RSI from 5 days ago |
| ... | 1-day and 5-day lags of 18 key features |

## 13. Normalization Features (~80)

Applied as additional columns, not replacements:

| Type | Description |
|------|-------------|
| Z-score (time-series) | Rolling mean/std normalization per stock |
| Z-score (cross-sectional) | Rank relative to other stocks same day |
| Percentile rank | 0–1 bounded, distribution-free |
| Winsorized | Extreme values capped at ±3σ |

---

## Feature Selection

After computation, features pass through:
1. **Variance filter**: Drop features with < 0.001 variance (near-constant)
2. **NaN filter**: Drop features with > 50% missing values
3. **Typical result**: ~385 usable features from ~443 computed

The models handle remaining NaN values internally (XGBoost/LightGBM/CatBoost all support native missing value handling).
