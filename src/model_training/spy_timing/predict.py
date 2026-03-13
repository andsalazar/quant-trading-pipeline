"""
================================================================================
06_02 SPY TIMING — DAILY PREDICTION
================================================================================
Purpose:  Load the trained SPY timing model and generate today's signal.
          Outputs a clear BUY / CASH recommendation for tomorrow's open.

Workflow:  Run AFTER the daily pipeline (Steps 1-8b) completes.
           → Loads today's features from ml_features_master.csv
           → Computes SPY technicals from SPY_Close_fut
           → Runs LGB + XGB ensemble
           → Prints signal + appends to prediction log

Output:   1. Console: BUY or CASH with probability
          2. spy_timing_signal_latest.csv — single-row file for easy reading
          3. spy_timing_prediction_log.csv — append-only log of all predictions
             (used by 06_03_track_spy_timing.py for accuracy tracking)

Created:  2026-02-08
================================================================================
"""

import os, sys, warnings, pickle, time
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class SPYTimingPredictor:
    """Generate daily SPY timing signal from trained model."""

    def __init__(self):
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.features_csv = os.path.join(self.base_path, '#_feature_engineering', 'ml_features_master.csv')
        self.futures_long_csv = os.path.join(self.base_path, '#_fetch_data', '#_04_fetch_futures', '00_04_futures_long.csv')
        self.model_path = os.path.join(self.base_path, '#_model_training', 'trained_models', 'spy_timing', 'spy_timing_model.pkl')
        self.output_dir = os.path.join(self.base_path, '#_model_training')

        self.signal_file = os.path.join(self.output_dir, 'spy_timing_signal_latest.csv')
        self.log_file = os.path.join(self.output_dir, 'spy_timing_prediction_log.csv')

    def _compute_spy_technicals(self, daily_df):
        """Compute SPY-specific technical indicators from SPY_Close_fut.
        Must match 06_01 exactly."""
        df = daily_df.copy()
        price = df['SPY_Close_fut']

        df['spy_ret_1d'] = price.pct_change()
        df['spy_ret_5d'] = price / price.shift(5) - 1
        df['spy_ret_10d'] = price / price.shift(10) - 1
        df['spy_ret_20d'] = price / price.shift(20) - 1
        df['spy_ret_60d'] = price / price.shift(60) - 1

        df['spy_vol_5d'] = df['spy_ret_1d'].rolling(5).std()
        df['spy_vol_20d'] = df['spy_ret_1d'].rolling(20).std()
        df['spy_vol_60d'] = df['spy_ret_1d'].rolling(60).std()
        df['spy_vol_ratio'] = df['spy_vol_5d'] / df['spy_vol_20d']

        df['spy_sma5_ratio'] = price / price.rolling(5).mean()
        df['spy_sma10_ratio'] = price / price.rolling(10).mean()
        df['spy_sma20_ratio'] = price / price.rolling(20).mean()
        df['spy_sma50_ratio'] = price / price.rolling(50).mean()

        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['spy_rsi_14'] = 100 - (100 / (1 + rs))

        sma20 = price.rolling(20).mean()
        std20 = price.rolling(20).std()
        df['spy_bb_position'] = (price - (sma20 - 2*std20)) / (4*std20)

        df['spy_gap'] = df['spy_ret_1d'] - df.get('ES_Return_1d', df['spy_ret_1d'])

        up = (df['spy_ret_1d'] > 0).astype(int)
        df['spy_consec_up'] = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
        df['spy_consec_up'] = df['spy_consec_up'] * up
        down = (df['spy_ret_1d'] < 0).astype(int)
        df['spy_consec_down'] = down.groupby((down != down.shift()).cumsum()).cumcount() + 1
        df['spy_consec_down'] = df['spy_consec_down'] * down

        peak = price.expanding().max()
        df['spy_drawdown'] = price / peak - 1

        df['day_of_week'] = df['date'].dt.dayofweek

        return df

    def run(self):
        t0 = time.time()
        print("=" * 80)
        print("STEP 06_02: SPY TIMING — DAILY PREDICTION")
        print("=" * 80)

        # ---- Load model ----
        print("\n[1/4] Loading trained model...")
        if not os.path.exists(self.model_path):
            print(f"  ERROR: Model not found: {self.model_path}")
            print("  Run 06_01_spy_timing_model.py first to train.")
            return False

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        lgb_model = model_data['lgb_model']
        xgb_model = model_data['xgb_model']
        features = model_data['features']
        cross_asset_features = model_data['cross_asset_features']
        regime_features = model_data.get('regime_features', [])
        flow_features = model_data.get('flow_features', [])
        print(f"  OK Model loaded ({len(features)} features incl {len(regime_features)} regime + {len(flow_features)} flow)")

        # ---- Load latest features ----
        print("\n[2/4] Loading latest features...")
        df = pd.read_csv(self.features_csv)
        df['date'] = pd.to_datetime(df['date'])

        # One row per date
        daily = df.groupby('date').first().reset_index()
        daily = daily.sort_values('date').reset_index(drop=True)

        # Filter to dates with SPY price
        daily = daily[daily['SPY_Close_fut'].notna()].copy()

        # Compute SPY technicals (need history for rolling windows)
        daily = self._compute_spy_technicals(daily)

        # Get the latest row
        latest = daily.iloc[-1]
        signal_date = latest['date']
        spy_close = latest['SPY_Close_fut']

        print(f"  OK Latest date: {signal_date.date()}")
        print(f"  OK SPY Close:   ${spy_close:.2f}")

        # ---- Generate prediction ----
        print("\n[3/4] Generating prediction...")

        # Build feature vector
        X = daily[features].iloc[[-1]]  # last row as DataFrame

        # Check for missing features
        missing = [f for f in features if f not in X.columns]
        if missing:
            print(f"  WARNING: Missing features: {missing}")
            for m in missing:
                X[m] = 0

        nan_feats = X.isna().sum(axis=1).iloc[0]
        if nan_feats > 0:
            nan_cols = X.columns[X.isna().iloc[0]].tolist()
            print(f"  WARNING: {nan_feats} NaN features (filling with 0): {nan_cols[:5]}...")
            X = X.fillna(0)

        # Predict
        lgb_prob = lgb_model.predict_proba(X)[:, 1][0]
        xgb_prob = xgb_model.predict_proba(X)[:, 1][0]
        ens_prob = (lgb_prob + xgb_prob) / 2

        signal = "BUY" if ens_prob > 0.50 else "CASH"
        confidence = abs(ens_prob - 0.50) * 200  # 0-100 scale

        # ---- Output ----
        print("\n[4/4] Signal Output")
        print("=" * 80)
        print(f"\n  Date:         {signal_date.date()} (after close)")
        print(f"  SPY Close:    ${spy_close:.2f}")
        print(f"  LGB Prob:     {lgb_prob:.4f}")
        print(f"  XGB Prob:     {xgb_prob:.4f}")
        print(f"  Ensemble:     {ens_prob:.4f}")
        print()

        if signal == "BUY":
            print(f"  ╔══════════════════════════════════════╗")
            print(f"  ║  SIGNAL:  BUY SPY at tomorrow's OPEN ║")
            print(f"  ║  Confidence: {confidence:.0f}%                     ║")
            print(f"  ╚══════════════════════════════════════╝")
        else:
            print(f"  ╔══════════════════════════════════════╗")
            print(f"  ║  SIGNAL:  CASH (stay out / sell)     ║")
            print(f"  ║  Confidence: {confidence:.0f}%                     ║")
            print(f"  ╚══════════════════════════════════════╝")

        print()

        # Key features for context
        print("  Key feature values:")
        key_feats = ['Stocks_Above_SMA20', 'spy_rsi_14', 'spy_sma20_ratio',
                     'spy_ret_1d', 'spy_vol_20d', 'spy_drawdown']
        for feat in key_feats:
            if feat in daily.columns:
                val = latest.get(feat, np.nan)
                if pd.notna(val):
                    if 'ratio' in feat or 'drawdown' in feat:
                        print(f"    {feat:30s}  {val:.4f}")
                    elif 'ret' in feat or 'vol' in feat:
                        print(f"    {feat:30s}  {val*100:.2f}%")
                    else:
                        print(f"    {feat:30s}  {val:.2f}")
        print()

        # ---- Save signal file (latest only) ----
        signal_row = {
            'signal_date': signal_date.date(),
            'execute_at': 'NEXT_OPEN',
            'spy_close': spy_close,
            'signal': signal,
            'ensemble_prob': round(ens_prob, 4),
            'lgb_prob': round(lgb_prob, 4),
            'xgb_prob': round(xgb_prob, 4),
            'confidence_pct': round(confidence, 1),
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        pd.DataFrame([signal_row]).to_csv(self.signal_file, index=False)
        print(f"  OK Signal saved: {self.signal_file}")

        # ---- Append to prediction log ----
        log_row = {
            'signal_date': str(signal_date.date()),
            'spy_close': spy_close,
            'signal': signal,
            'ensemble_prob': round(ens_prob, 4),
            'lgb_prob': round(lgb_prob, 4),
            'xgb_prob': round(xgb_prob, 4),
            'confidence_pct': round(confidence, 1),
            'actual_next_day_return': np.nan,
            'actual_direction': '',
            'correct': '',
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filled_at': '',
        }

        if os.path.exists(self.log_file):
            log_df = pd.read_csv(self.log_file)
            # Don't duplicate if already ran today
            if str(signal_date.date()) in log_df['signal_date'].astype(str).values:
                print(f"  INFO: Prediction for {signal_date.date()} already in log — updating")
                log_df = log_df[log_df['signal_date'].astype(str) != str(signal_date.date())]
            log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_row])

        log_df.to_csv(self.log_file, index=False)
        print(f"  OK Log updated: {self.log_file} ({len(log_df)} entries)")

        elapsed = time.time() - t0
        print(f"\nTIME: {elapsed:.1f} seconds")
        print("=" * 80)
        return True


if __name__ == '__main__':
    SPYTimingPredictor().run()
