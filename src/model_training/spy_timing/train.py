"""
================================================================================
06_01 SPY TIMING MODEL — TRAIN + BACKTEST
================================================================================
Purpose:  Predict next-day SPY direction using cross-asset/macro features.
          Strategy: Long SPY when model says "up", cash when "down."
          Realistic backtest at $2,000 with IBKR commissions.

Execution:  NEXT-OPEN
          Pipeline runs after market close → computes features from today's
          closing data → generates signal → execute at NEXT DAY'S OPEN.
          This is realistic because Polygon daily bars aren't available
          until after 4:00 PM ET.

Data:     Cross-asset features from ml_features_master.csv (one row per date).
          SPY prices from SPY_Close_fut (futures-derived, 2015-2026 coverage).
          SPY Open prices from futures_long.csv for execution simulation.
          SPY-specific technicals computed from SPY_Close_fut.

Target:   Binary — will SPY go up tomorrow? (close-to-close, SPY_fwd_1d > 0)
          Note: absolute return, NOT cross-sectional. Different from pooled model.
          The actual P&L is open-to-open since we execute at open.

Models:   LightGBM + XGBoost ensemble (simple average).
Walk-Forward: 5 yearly folds (test 2021-2025), expanding window.

Created:  2026-02-08
================================================================================
"""

import os, sys, warnings, json, time, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')


class SPYTimingModel:
    """SPY market timing model with integrated backtest."""

    def __init__(self, initial_capital=2000.0):
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.features_csv = os.path.join(self.base_path, '#_feature_engineering', 'ml_features_master.csv')
        self.futures_long_csv = os.path.join(self.base_path, '#_fetch_data', '#_04_fetch_futures', '00_04_futures_long.csv')
        self.model_dir = os.path.join(self.base_path, '#_model_training', 'trained_models', 'spy_timing')
        self.output_dir = os.path.join(self.base_path, '#_model_training')
        self.initial_capital = initial_capital

        os.makedirs(self.model_dir, exist_ok=True)

        # IBKR commissions
        self.comm_per_share = 0.0035
        self.comm_min = 0.35
        self.regulatory = 0.01

        # Slippage
        self.slippage_pct = 0.0001  # 1 bps per side (SPY is ultra-liquid)

        # Walk-forward folds
        self.folds = [
            {'train_end': '2020-12-31', 'test_year': 2021},
            {'train_end': '2021-12-31', 'test_year': 2022},
            {'train_end': '2022-12-31', 'test_year': 2023},
            {'train_end': '2023-12-31', 'test_year': 2024},
            {'train_end': '2024-12-31', 'test_year': 2025},
        ]

        # Conservative hyperparameters (defaults — overridden if optimized params exist)
        self.lgb_params = {
            'n_estimators': 300,
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'min_child_samples': 50,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        self.xgb_params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'min_child_weight': 50,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'tree_method': 'gpu_hist',
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1,
        }

        # Load optimized hyperparameters if available
        opt_path = os.path.join(self.model_dir, 'spy_optimized_hyperparameters.json')
        if os.path.exists(opt_path):
            with open(opt_path, 'r') as f:
                opt = json.load(f)
            if 'spy_timing_lgb' in opt:
                self.lgb_params = opt['spy_timing_lgb']
                print(f"  [INFO] Loaded optimized LGB params (CV AUC={opt.get('lgb_cv_auc', '?')})")
            if 'spy_timing_xgb' in opt:
                self.xgb_params = opt['spy_timing_xgb']
                print(f"  [INFO] Loaded optimized XGB params (CV AUC={opt.get('xgb_cv_auc', '?')})")

        # ---- Feature groups ----
        # Cross-asset returns (same for all symbols per date)
        # NOTE: IWM_Return, VTI_Return, Market_Return_Mean are EXCLUDED
        #       because they contain forward-looking (T+1) data in the CSV.
        self.cross_asset_features = [
            # ETF returns (safe: same-day aligned)
            'SPY_Return_1d', 'QQQ_Return_1d',
            'GLD_Return', 'TLT_Return', 'GLD_Return_1d', 'TLT_Return_1d',
            'VXX_Return_1d', 'USO_Return',
            # Futures returns (safe: _1d suffix = same-day)
            'ES_Return_1d', 'CL_Return_1d', 'NG_Return_1d', 'DX_Return_1d',
            'HG_Return_1d', 'TY_Return_1d', 'FV_Return_1d',
            # Currency returns (safe: same-day)
            'EUR_USD_return', 'GBP_USD_return', 'USD_JPY_return',
            'USD_CAD_return', 'AUD_USD_return',
            # Market breadth / macro (safe)
            'Market_Return_Std',
            'Stocks_Above_SMA20',
            # Yield / rates
            'Yield_Curve_Momentum',
            # Currency aggregate
            'Currency_Market_Momentum_Avg', 'Currency_Market_Momentum_Dispersion',
            # Calendar
            'month', 'quarter',
        ]

    def _compute_spy_technicals(self, daily_df):
        """Compute SPY-specific technical indicators from SPY_Close_fut."""
        df = daily_df.copy()
        price = df['SPY_Close_fut']

        # Returns
        df['spy_ret_1d'] = price.pct_change()
        df['spy_ret_5d'] = price / price.shift(5) - 1
        df['spy_ret_10d'] = price / price.shift(10) - 1
        df['spy_ret_20d'] = price / price.shift(20) - 1
        df['spy_ret_60d'] = price / price.shift(60) - 1

        # Volatility
        df['spy_vol_5d'] = df['spy_ret_1d'].rolling(5).std()
        df['spy_vol_20d'] = df['spy_ret_1d'].rolling(20).std()
        df['spy_vol_60d'] = df['spy_ret_1d'].rolling(60).std()
        df['spy_vol_ratio'] = df['spy_vol_5d'] / df['spy_vol_20d']

        # Moving average ratios
        df['spy_sma5_ratio'] = price / price.rolling(5).mean()
        df['spy_sma10_ratio'] = price / price.rolling(10).mean()
        df['spy_sma20_ratio'] = price / price.rolling(20).mean()
        df['spy_sma50_ratio'] = price / price.rolling(50).mean()

        # RSI-14
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['spy_rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger position
        sma20 = price.rolling(20).mean()
        std20 = price.rolling(20).std()
        df['spy_bb_position'] = (price - (sma20 - 2*std20)) / (4*std20)

        # High-low range (use ES data if available)
        df['spy_gap'] = df['spy_ret_1d'] - df.get('ES_Return_1d', df['spy_ret_1d'])

        # Consecutive up/down days
        up = (df['spy_ret_1d'] > 0).astype(int)
        df['spy_consec_up'] = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
        df['spy_consec_up'] = df['spy_consec_up'] * up
        down = (df['spy_ret_1d'] < 0).astype(int)
        df['spy_consec_down'] = down.groupby((down != down.shift()).cumsum()).cumcount() + 1
        df['spy_consec_down'] = df['spy_consec_down'] * down

        # Drawdown from peak
        peak = price.expanding().max()
        df['spy_drawdown'] = price / peak - 1

        # Day of week (0=Mon, 4=Fri)
        df['day_of_week'] = df['date'].dt.dayofweek

        return df

    def _ibkr_commission(self, shares, price):
        """IBKR tiered commission."""
        trade_value = shares * price
        comm = max(shares * self.comm_per_share, self.comm_min)
        comm = min(comm, trade_value * 0.01)
        comm += self.regulatory
        return comm

    def run(self):
        t0 = time.time()
        print("=" * 80)
        print("STEP 06_01: SPY TIMING MODEL — TRAIN + BACKTEST")
        print("=" * 80)
        print(f"Target: Will SPY go up tomorrow? (absolute, not relative)")
        print(f"Execution: NEXT-DAY OPEN (realistic — data available after close)")
        print(f"Strategy: Long SPY when bullish, cash when bearish")
        print(f"Capital: ${self.initial_capital:,.0f} | Broker: IBKR")
        print("=" * 80)

        # ============================================================
        # PART 1: DATA PREPARATION
        # ============================================================
        print("\n[1/8] Loading cross-asset features (one row per date)...")
        df = pd.read_csv(self.features_csv)
        df['date'] = pd.to_datetime(df['date'])

        # Get one row per date (cross-asset features are identical across symbols)
        daily = df.groupby('date').first().reset_index()
        daily = daily.sort_values('date').reset_index(drop=True)

        print(f"  OK {len(daily)} trading dates ({daily['date'].min().date()} to {daily['date'].max().date()})")

        # Filter to dates with SPY price
        daily = daily[daily['SPY_Close_fut'].notna()].copy()
        print(f"  OK {len(daily)} dates with SPY_Close_fut")

        # Load SPY Open prices for next-open execution
        print("  Loading SPY Open prices from futures_long.csv...")
        futures_long = pd.read_csv(self.futures_long_csv)
        spy_opens = futures_long[futures_long['Symbol'] == 'SPY'][['Date', 'Open']].copy()
        spy_opens['Date'] = pd.to_datetime(spy_opens['Date']).dt.normalize()
        spy_opens = spy_opens.rename(columns={'Open': 'SPY_Open', 'Date': 'date'})
        spy_opens = spy_opens.drop_duplicates('date').sort_values('date')
        daily = daily.merge(spy_opens, on='date', how='left')
        # Forward-fill a few missing opens, drop the rest
        daily['SPY_Open'] = daily['SPY_Open'].ffill(limit=2)
        n_open = daily['SPY_Open'].notna().sum()
        print(f"  OK {n_open} dates with SPY_Open")

        # ---- Compute SPY technicals ----
        print("\n[2/8] Computing SPY-specific technicals...")
        daily = self._compute_spy_technicals(daily)

        spy_tech_features = [
            'spy_ret_1d', 'spy_ret_5d', 'spy_ret_10d', 'spy_ret_20d', 'spy_ret_60d',
            'spy_vol_5d', 'spy_vol_20d', 'spy_vol_60d', 'spy_vol_ratio',
            'spy_sma5_ratio', 'spy_sma10_ratio', 'spy_sma20_ratio', 'spy_sma50_ratio',
            'spy_rsi_14', 'spy_bb_position', 'spy_gap',
            'spy_consec_up', 'spy_consec_down', 'spy_drawdown',
            'day_of_week',
        ]

        # Auto-discover regime (reg_*) and cross-asset flow (flow_*) features
        # These are cross-sectional (same for all symbols per date) — ideal for SPY timing
        regime_features = sorted([c for c in daily.columns if c.startswith('reg_')])
        flow_features = sorted([c for c in daily.columns if c.startswith('flow_')])
        print(f"  OK Auto-discovered: {len(regime_features)} regime + {len(flow_features)} flow features")

        # Full feature list
        all_features = self.cross_asset_features + spy_tech_features + regime_features + flow_features
        # Remove any that don't exist and deduplicate
        all_features = list(dict.fromkeys(f for f in all_features if f in daily.columns))
        print(f"  OK {len(all_features)} total features")

        # ---- Compute target ----
        print("\n[3/8] Computing target (SPY up tomorrow?)...")
        # Target: close-to-close direction (will tomorrow's close > today's close?)
        daily['spy_fwd_1d'] = daily['SPY_Close_fut'].shift(-1) / daily['SPY_Close_fut'] - 1
        daily['target'] = (daily['spy_fwd_1d'] > 0).astype(int)

        # Next-day open price (for execution simulation)
        daily['next_open'] = daily['SPY_Open'].shift(-1)
        # Next-next-day open (for exit price if we sell at the next signal's open)
        daily['next_next_open'] = daily['SPY_Open'].shift(-2)

        daily = daily.dropna(subset=['target', 'spy_fwd_1d', 'next_open'])

        # Drop initial rows with NaN technicals (need ~60 days warmup)
        daily = daily.iloc[60:].reset_index(drop=True)

        up_days = daily['target'].sum()
        total = len(daily)
        print(f"  OK {total} samples (after 60-day warmup)")
        print(f"  OK Up days: {up_days} ({100*up_days/total:.1f}%), Down: {total-up_days} ({100*(total-up_days)/total:.1f}%)")

        # Check for NaN in features
        nan_pct = daily[all_features].isna().mean()
        high_nan = nan_pct[nan_pct > 0.5]
        if len(high_nan) > 0:
            print(f"  WARNING: Dropping {len(high_nan)} features with >50% NaN:")
            for f in high_nan.index:
                print(f"    {f}: {nan_pct[f]*100:.0f}% missing")
            all_features = [f for f in all_features if f not in high_nan.index]
            print(f"  OK {len(all_features)} features after cleanup")

        # ============================================================
        # PART 2: WALK-FORWARD TRAINING
        # ============================================================
        print("\n[4/8] Walk-forward validation...")
        all_oos = []
        fold_metrics = []

        for fold_i, fold in enumerate(self.folds):
            train_end = pd.Timestamp(fold['train_end'])
            test_year = fold['test_year']
            test_start = pd.Timestamp(f'{test_year}-01-01')
            test_end = pd.Timestamp(f'{test_year}-12-31')

            train_data = daily[daily['date'] <= train_end]
            test_data = daily[(daily['date'] >= test_start) & (daily['date'] <= test_end)]

            if len(test_data) < 10:
                print(f"\n  Fold {fold_i+1}: Not enough test data for {test_year}, skipping")
                continue

            X_train = train_data[all_features]
            y_train = train_data['target']
            X_test = test_data[all_features]
            y_test = test_data['target']

            print(f"\n  Fold {fold_i+1}/5: Train ≤{fold['train_end']} ({len(train_data)} days), Test {test_year} ({len(test_data)} days)")

            # LightGBM
            lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            lgb_model.fit(X_train, y_train)
            lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
            lgb_auc = roc_auc_score(y_test, lgb_probs)

            # XGBoost
            xgb_model = xgb.XGBClassifier(**self.xgb_params)
            xgb_model.fit(X_train, y_train)
            xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
            xgb_auc = roc_auc_score(y_test, xgb_probs)

            # Ensemble
            ens_probs = (lgb_probs + xgb_probs) / 2
            ens_preds = (ens_probs > 0.5).astype(int)
            ens_auc = roc_auc_score(y_test, ens_probs)
            ens_acc = accuracy_score(y_test, ens_preds)

            up_pct = y_test.mean() * 100
            pred_up_pct = ens_preds.mean() * 100
            print(f"    LGB AUC={lgb_auc:.4f}, XGB AUC={xgb_auc:.4f}, Ens AUC={ens_auc:.4f}, Acc={ens_acc:.4f}")
            print(f"    Actual up: {up_pct:.1f}%, Predicted up: {pred_up_pct:.1f}%")

            fold_metrics.append({
                'fold': fold_i+1, 'test_year': test_year,
                'lgb_auc': lgb_auc, 'xgb_auc': xgb_auc,
                'ens_auc': ens_auc, 'ens_acc': ens_acc,
                'actual_up_pct': up_pct, 'pred_up_pct': pred_up_pct,
            })

            oos = test_data[['date', 'SPY_Close_fut', 'SPY_Open', 'next_open', 'next_next_open', 'spy_fwd_1d', 'target']].copy()
            oos['lgb_prob'] = lgb_probs
            oos['xgb_prob'] = xgb_probs
            oos['ens_prob'] = ens_probs
            oos['ens_pred'] = ens_preds
            oos['fold'] = fold_i + 1
            all_oos.append(oos)

        # ---- Feature importance ----
        print("\n[5/8] Feature importance (last fold)...")
        importances = pd.Series(lgb_model.feature_importances_, index=all_features)
        importances = importances.sort_values(ascending=False)
        print("  Top 15 features:")
        for i, (feat, imp) in enumerate(importances.head(15).items()):
            print(f"    {i+1:2d}. {feat:45s} {imp:.0f}")

        # ============================================================
        # PART 3: BACKTEST
        # ============================================================
        print("\n[6/8] Backtesting strategies...")
        all_oos_df = pd.concat(all_oos, ignore_index=True)
        all_oos_df = all_oos_df.sort_values('date').reset_index(drop=True)

        strategies = {}

        # --- Strategy A: Long when prob > 0.5, cash otherwise ---
        strategies['Model (p>0.50)'] = self._backtest_strategy(
            all_oos_df, threshold=0.50, label='Model (p>0.50)'
        )

        # --- Strategy B: Higher confidence threshold ---
        strategies['Model (p>0.55)'] = self._backtest_strategy(
            all_oos_df, threshold=0.55, label='Model (p>0.55)'
        )

        # --- Strategy C: High confidence ---
        strategies['Model (p>0.60)'] = self._backtest_strategy(
            all_oos_df, threshold=0.60, label='Model (p>0.60)'
        )

        # --- Benchmark: Buy and hold ---
        strategies['SPY Buy & Hold'] = self._backtest_buyhold(all_oos_df)

        # --- Benchmark: Always cash (0% return) ---
        strategies['Cash'] = {
            'final': self.initial_capital, 'total_return': 0, 'ann_return': 0,
            'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'trades': 0,
            'invested_pct': 0, 'total_comm': 0,
        }

        # ============================================================
        # PART 4: RESULTS
        # ============================================================
        print("\n[7/8] Results Summary")
        print("=" * 80)

        metrics_df = pd.DataFrame(fold_metrics)
        print(f"\n  Walk-Forward AUC by Year:")
        for _, row in metrics_df.iterrows():
            print(f"    {row['test_year']}: LGB={row['lgb_auc']:.4f}, XGB={row['xgb_auc']:.4f}, Ens={row['ens_auc']:.4f}, Acc={row['ens_acc']:.4f}")
        print(f"    AVG:  LGB={metrics_df['lgb_auc'].mean():.4f}, XGB={metrics_df['xgb_auc'].mean():.4f}, Ens={metrics_df['ens_auc'].mean():.4f}, Acc={metrics_df['ens_acc'].mean():.4f}")

        print(f"\n  Backtest Results (${self.initial_capital:,.0f} initial, NEXT-OPEN execution, 2021-2025):")
        print(f"  {'Strategy':<20s}  {'Final':>10s}  {'Return':>8s}  {'Ann.':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'WinR':>6s}  {'Trades':>6s}  {'Invested':>8s}  {'Comms':>8s}")
        for name, m in strategies.items():
            print(f"  {name:<20s}  ${m['final']:>9,.2f}  {m['total_return']:>+7.1f}%  {m['ann_return']:>+6.1f}%  {m['sharpe']:>7.2f}  {m['max_dd']:>6.1f}%  {m['win_rate']:>5.1f}%  {m['trades']:>6d}  {m['invested_pct']:>7.0f}%  ${m['total_comm']:>7.2f}")

        # Year-by-year for best strategy
        print(f"\n  Year-by-Year Returns (Model p>0.50):")
        if 'equity_curve' in strategies['Model (p>0.50)']:
            eq = strategies['Model (p>0.50)']['equity_curve']
            eq['year'] = eq['date'].dt.year
            spy_eq = strategies['SPY Buy & Hold'].get('equity_curve')
            for year in sorted(eq['year'].unique()):
                yr = eq[eq['year'] == year]
                yr_ret = (yr['equity'].iloc[-1] / yr['equity'].iloc[0] - 1) * 100
                spy_yr_ret = ''
                if spy_eq is not None:
                    spy_yr = spy_eq[spy_eq['date'].dt.year == year]
                    if len(spy_yr) > 1:
                        spy_yr_ret = f"  (SPY: {(spy_yr['equity'].iloc[-1] / spy_yr['equity'].iloc[0] - 1) * 100:+.1f}%)"
                print(f"    {year}: {yr_ret:+.1f}%{spy_yr_ret}")

        # Signal analysis
        print(f"\n  Signal Analysis:")
        correct_up = ((all_oos_df['ens_pred'] == 1) & (all_oos_df['target'] == 1)).sum()
        pred_up = (all_oos_df['ens_pred'] == 1).sum()
        correct_down = ((all_oos_df['ens_pred'] == 0) & (all_oos_df['target'] == 0)).sum()
        pred_down = (all_oos_df['ens_pred'] == 0).sum()
        print(f"    Predicted UP: {pred_up} days, correct: {correct_up} ({correct_up/pred_up*100:.1f}% precision)")
        print(f"    Predicted DOWN: {pred_down} days, correct: {correct_down} ({correct_down/pred_down*100:.1f}% precision)")
        print(f"    Avg return on UP days: {all_oos_df[all_oos_df['ens_pred']==1]['spy_fwd_1d'].mean()*100:.3f}%")
        print(f"    Avg return on DOWN days: {all_oos_df[all_oos_df['ens_pred']==0]['spy_fwd_1d'].mean()*100:.3f}%")

        # ---- Save production model ----
        print("\n[8/8] Training production model (all data)...")
        X_all = daily[all_features]
        y_all = daily['target']

        prod_lgb = lgb.LGBMClassifier(**self.lgb_params)
        prod_lgb.fit(X_all, y_all)

        prod_xgb = xgb.XGBClassifier(**self.xgb_params)
        prod_xgb.fit(X_all, y_all)

        model_path = os.path.join(self.model_dir, 'spy_timing_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'lgb_model': prod_lgb,
                'xgb_model': prod_xgb,
                'features': all_features,
                'cross_asset_features': self.cross_asset_features,
                'spy_tech_features': spy_tech_features,
                'regime_features': regime_features,
                'flow_features': flow_features,
            }, f)
        print(f"  OK Model saved: {model_path}")

        # Save OOS predictions
        oos_path = os.path.join(self.output_dir, 'spy_timing_predictions_oos.csv')
        all_oos_df.to_csv(oos_path, index=False)

        elapsed = time.time() - t0
        print(f"\nTIME: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print("=" * 80)

    def _backtest_strategy(self, oos_df, threshold, label):
        """Backtest: long SPY when ens_prob > threshold, cash otherwise.

        Execution model (NEXT-OPEN):
          - Signal computed from day T's close features (pipeline runs ~5 PM).
          - BUY/SELL executed at day T+1's OPEN ('next_open' column).
          - Mark-to-market at day T's close (SPY_Close_fut) for equity curve.
          - This is realistic: Polygon daily bars arrive after close.
        """
        capital = self.initial_capital
        position = 0  # shares held
        entry_price = 0.0
        in_position = False
        total_comm = 0
        equity_curve = []
        trades = 0
        invested_days = 0
        pending_buy = False
        pending_sell = False

        rows = oos_df.reset_index(drop=True)

        for i in range(len(rows)):
            row = rows.iloc[i]
            close_price = row['SPY_Close_fut']
            open_price = row.get('SPY_Open', np.nan)  # today's open

            # ---- Execute pending orders at today's open ----
            if pending_buy and not np.isnan(open_price):
                shares = int(capital / open_price)
                if shares > 0:
                    cost = shares * open_price
                    comm = self._ibkr_commission(shares, open_price)
                    slip = cost * self.slippage_pct
                    capital -= (cost + comm + slip)
                    total_comm += comm
                    position = shares
                    entry_price = open_price
                    in_position = True
                    trades += 1
                pending_buy = False

            elif pending_sell and in_position and not np.isnan(open_price):
                proceeds = position * open_price
                comm = self._ibkr_commission(position, open_price)
                slip = proceeds * self.slippage_pct
                capital += (proceeds - comm - slip)
                total_comm += comm
                position = 0
                in_position = False
                trades += 1
                pending_sell = False

            # ---- Generate signal for TOMORROW's open ----
            signal = row['ens_prob'] > threshold

            if signal and not in_position and not pending_buy:
                pending_buy = True
                pending_sell = False
            elif not signal and in_position and not pending_sell:
                pending_sell = True
                pending_buy = False

            # ---- Mark-to-market at today's close ----
            if in_position:
                equity = capital + position * close_price
                invested_days += 1
            else:
                equity = capital

            equity_curve.append({'date': row['date'], 'equity': equity})

        # Close any remaining position at last close
        if in_position and len(rows) > 0:
            last_price = rows.iloc[-1]['SPY_Close_fut']
            proceeds = position * last_price
            comm = self._ibkr_commission(position, last_price)
            capital += (proceeds - comm)
            total_comm += comm
            equity_curve[-1]['equity'] = capital

        eq_df = pd.DataFrame(equity_curve)
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        final = eq_df['equity'].iloc[-1]
        total_return = (final / self.initial_capital - 1) * 100
        n_years = len(eq_df) / 252
        ann_return = ((final / self.initial_capital) ** (1/n_years) - 1) * 100 if n_years > 0 and final > 0 else 0

        daily_rets = eq_df['equity'].pct_change().dropna()
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

        peak = eq_df['equity'].cummax()
        dd = ((eq_df['equity'] - peak) / peak).min() * 100

        # Win rate: count days where we were invested and market went up
        win_rate = 0
        if invested_days > 0:
            invested_mask = eq_df['equity'].diff() != 0
            if invested_mask.sum() > 0:
                wins = (eq_df['equity'].diff() > 0).sum()
                win_rate = wins / max(invested_mask.sum(), 1) * 100

        return {
            'final': final,
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe': sharpe,
            'max_dd': dd,
            'win_rate': win_rate,
            'trades': trades,
            'invested_pct': invested_days / len(oos_df) * 100,
            'total_comm': total_comm,
            'equity_curve': eq_df,
        }

    def _backtest_buyhold(self, oos_df):
        """Benchmark: buy SPY at first day's open, hold."""
        first_open = oos_df.iloc[0]['SPY_Open']
        if pd.isna(first_open):
            first_open = oos_df.iloc[0]['SPY_Close_fut']  # fallback
        shares = int(self.initial_capital / first_open)
        if shares < 1:
            return {'final': self.initial_capital, 'total_return': 0, 'ann_return': 0,
                    'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'trades': 1,
                    'invested_pct': 100, 'total_comm': 0}

        comm = self._ibkr_commission(shares, first_open)
        cash_left = self.initial_capital - (shares * first_open + comm + shares * first_open * self.slippage_pct)
        equity_curve = []
        for _, row in oos_df.iterrows():
            equity = cash_left + shares * row['SPY_Close_fut']
            equity_curve.append({'date': row['date'], 'equity': equity})

        eq_df = pd.DataFrame(equity_curve)
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        final = eq_df['equity'].iloc[-1]
        total_return = (final / self.initial_capital - 1) * 100
        n_years = len(eq_df) / 252
        ann_return = ((final / self.initial_capital) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

        daily_rets = eq_df['equity'].pct_change().dropna()
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

        peak = eq_df['equity'].cummax()
        dd = ((eq_df['equity'] - peak) / peak).min() * 100

        wins = (daily_rets > 0).sum()
        win_rate = wins / len(daily_rets) * 100 if len(daily_rets) > 0 else 0

        return {
            'final': final,
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe': sharpe,
            'max_dd': dd,
            'win_rate': win_rate,
            'trades': 1,
            'invested_pct': 100,
            'total_comm': comm,
            'equity_curve': eq_df,
        }


if __name__ == '__main__':
    SPYTimingModel(initial_capital=2000.0).run()
