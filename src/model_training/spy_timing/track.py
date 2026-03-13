"""
================================================================================
06_03 SPY TIMING — TRACK PREDICTIONS & ACCURACY
================================================================================
Purpose:  Backfill actual outcomes into the prediction log and compute rolling
          accuracy metrics. Alerts if model accuracy is degrading.

Workflow:  Run DAILY, BEFORE generating today's new prediction (06_02).
           → Reads spy_timing_prediction_log.csv
           → For any row missing 'actual_next_day_return', looks up the real
             SPY return for that date and fills it in
           → Computes rolling 20-day / 60-day accuracy, calibration, Brier score
           → Prints a dashboard and saves updated log
           → Alerts if rolling accuracy drops below random (50%)

Why:      Models degrade over time. This script lets you:
           1. See if the model is still working
           2. Detect regime changes early
           3. Build a record for future retraining decisions

Created:  2026-02-08
================================================================================
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class SPYTimingTracker:
    """Track SPY timing predictions and compute accuracy metrics."""

    def __init__(self):
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.features_csv = os.path.join(self.base_path, '#_feature_engineering', 'ml_features_master.csv')
        self.futures_long_csv = os.path.join(self.base_path, '#_fetch_data', '#_04_fetch_futures', '00_04_futures_long.csv')
        self.log_file = os.path.join(self.base_path, '#_model_training', 'spy_timing_prediction_log.csv')

        # Alert thresholds
        self.alert_window = 20  # rolling window for accuracy check
        self.alert_threshold = 0.45  # alert if accuracy drops below this

    def run(self):
        t0 = time.time()
        print("=" * 80)
        print("STEP 06_03: SPY TIMING — TRACK & BACKFILL ACTUALS")
        print("=" * 80)

        # ---- Load prediction log ----
        print("\n[1/5] Loading prediction log...")
        if not os.path.exists(self.log_file):
            print(f"  No prediction log found: {self.log_file}")
            print("  Run 06_02_predict_spy_timing.py first to generate predictions.")
            return False

        log = pd.read_csv(self.log_file)
        log['signal_date'] = pd.to_datetime(log['signal_date'])
        print(f"  OK {len(log)} predictions loaded")
        print(f"  OK Date range: {log['signal_date'].min().date()} to {log['signal_date'].max().date()}")

        unfilled = log['actual_next_day_return'].isna().sum()
        print(f"  OK {unfilled} predictions need actual outcomes filled")

        if unfilled == 0:
            print("  All predictions already have actuals filled.")
        else:
            # ---- Load SPY price data for backfilling ----
            print("\n[2/5] Loading SPY price data for backfilling...")

            # Get SPY close from features CSV
            df = pd.read_csv(self.features_csv, usecols=['date', 'SPY_Close_fut'])
            df['date'] = pd.to_datetime(df['date'])
            spy_prices = df.groupby('date')['SPY_Close_fut'].first().reset_index()
            spy_prices = spy_prices.dropna().sort_values('date').reset_index(drop=True)

            # Build date → close lookup
            price_map = dict(zip(spy_prices['date'], spy_prices['SPY_Close_fut']))

            # Also load from futures_long for more complete coverage
            if os.path.exists(self.futures_long_csv):
                fl = pd.read_csv(self.futures_long_csv)
                spy_fl = fl[fl['Symbol'] == 'SPY'][['Date', 'Close']].copy()
                spy_fl['Date'] = pd.to_datetime(spy_fl['Date']).dt.normalize()
                for _, r in spy_fl.iterrows():
                    if r['Date'] not in price_map:
                        price_map[r['Date']] = r['Close']

            all_dates_sorted = sorted(price_map.keys())
            print(f"  OK {len(price_map)} SPY price dates available")

            # ---- Backfill actuals ----
            print("\n[3/5] Backfilling actual outcomes...")
            filled_count = 0

            for idx in log.index:
                if pd.notna(log.loc[idx, 'actual_next_day_return']):
                    continue  # already filled

                signal_date = log.loc[idx, 'signal_date']
                # We need the NEXT trading day's close
                # Find signal_date in the sorted list, get the next date
                close_today = price_map.get(signal_date)
                if close_today is None:
                    continue  # signal date not in price data

                # Find next trading day
                next_date = None
                for d in all_dates_sorted:
                    if d > signal_date:
                        next_date = d
                        break

                if next_date is None:
                    continue  # no future data yet (most recent prediction)

                close_next = price_map.get(next_date)
                if close_next is None:
                    continue

                actual_ret = (close_next / close_today) - 1
                actual_dir = 'UP' if actual_ret > 0 else 'DOWN'
                predicted_dir = 'UP' if log.loc[idx, 'signal'] == 'BUY' else 'DOWN'
                correct = actual_dir == predicted_dir

                log.loc[idx, 'actual_next_day_return'] = round(actual_ret, 6)
                log.loc[idx, 'actual_direction'] = actual_dir
                log.loc[idx, 'correct'] = correct
                log.loc[idx, 'filled_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                filled_count += 1

            print(f"  OK Filled {filled_count} predictions with actual outcomes")

            # Save updated log
            log.to_csv(self.log_file, index=False)
            print(f"  OK Log saved: {self.log_file}")

        # ---- Compute accuracy metrics ----
        print("\n[4/5] Accuracy Dashboard")
        print("=" * 80)

        filled_log = log[log['actual_next_day_return'].notna()].copy()
        filled_log = filled_log.sort_values('signal_date').reset_index(drop=True)
        filled_log['correct'] = filled_log['correct'].astype(bool)

        n_filled = len(filled_log)
        if n_filled == 0:
            print("  No filled predictions yet — need at least 1 day of history.")
            print("  Run tomorrow after the pipeline to get the first actual result.")
            print("=" * 80)
            return True

        # Overall accuracy
        overall_acc = filled_log['correct'].mean()
        overall_n = len(filled_log)

        # By signal type
        buys = filled_log[filled_log['signal'] == 'BUY']
        cash = filled_log[filled_log['signal'] == 'CASH']

        buy_acc = buys['correct'].mean() if len(buys) > 0 else np.nan
        cash_acc = cash['correct'].mean() if len(cash) > 0 else np.nan

        # Avg return when BUY vs CASH
        buy_avg_ret = buys['actual_next_day_return'].mean() * 100 if len(buys) > 0 else np.nan
        cash_avg_ret = cash['actual_next_day_return'].mean() * 100 if len(cash) > 0 else np.nan

        # Brier score (lower is better, 0.25 = random)
        probs = filled_log['ensemble_prob'].values
        actuals = (filled_log['actual_direction'] == 'UP').astype(float).values
        brier = np.mean((probs - actuals) ** 2)

        print(f"\n  Overall Accuracy: {overall_acc*100:.1f}%  ({overall_n} predictions)")
        print(f"  Brier Score:      {brier:.4f}  (0.25 = random, lower = better)")
        print()
        print(f"  BUY signals:  {len(buys):>4d}  accuracy={buy_acc*100:.1f}%  avg SPY return={buy_avg_ret:+.3f}%")
        print(f"  CASH signals: {len(cash):>4d}  accuracy={cash_acc*100:.1f}%  avg SPY return={cash_avg_ret:+.3f}%")
        print()

        # Simulated P&L (simple: get SPY return on BUY days, 0 on CASH days)
        filled_log['strategy_return'] = np.where(
            filled_log['signal'] == 'BUY',
            filled_log['actual_next_day_return'],
            0.0
        )
        cumulative = (1 + filled_log['strategy_return']).cumprod()
        spy_cumulative = (1 + filled_log['actual_next_day_return']).cumprod()

        strat_total = (cumulative.iloc[-1] - 1) * 100
        spy_total = (spy_cumulative.iloc[-1] - 1) * 100

        print(f"  Cumulative return (strategy): {strat_total:+.2f}%")
        print(f"  Cumulative return (SPY B&H):  {spy_total:+.2f}%")
        print(f"  Alpha:                        {strat_total - spy_total:+.2f}%")

        # Rolling accuracy
        if n_filled >= self.alert_window:
            rolling_acc = filled_log['correct'].rolling(self.alert_window).mean()
            latest_rolling = rolling_acc.iloc[-1]
            best_rolling = rolling_acc.max()
            worst_rolling = rolling_acc.min()

            print(f"\n  Rolling {self.alert_window}-day accuracy:")
            print(f"    Current: {latest_rolling*100:.1f}%")
            print(f"    Best:    {best_rolling*100:.1f}%")
            print(f"    Worst:   {worst_rolling*100:.1f}%")

        if n_filled >= 60:
            rolling60 = filled_log['correct'].rolling(60).mean()
            print(f"\n  Rolling 60-day accuracy:")
            print(f"    Current: {rolling60.iloc[-1]*100:.1f}%")

        # Recent predictions table
        print(f"\n  Last 10 Predictions:")
        print(f"  {'Date':>12s}  {'Signal':>6s}  {'Prob':>5s}  {'Actual':>6s}  {'Return':>8s}  {'Correct':>7s}")
        for _, row in filled_log.tail(10).iterrows():
            correct_str = 'YES' if row['correct'] else 'NO'
            print(f"  {str(row['signal_date'].date()):>12s}  {row['signal']:>6s}  {row['ensemble_prob']:.3f}  {row['actual_direction']:>6s}  {row['actual_next_day_return']*100:>+7.3f}%  {correct_str:>7s}")

        # ---- Alerts ----
        print(f"\n[5/5] Alerts")
        print("-" * 40)

        alerts = []

        if n_filled >= self.alert_window:
            if latest_rolling < self.alert_threshold:
                alerts.append(
                    f"DEGRADATION: Rolling {self.alert_window}-day accuracy = "
                    f"{latest_rolling*100:.1f}% (below {self.alert_threshold*100:.0f}% threshold). "
                    f"Consider retraining the model."
                )

        if n_filled >= 60:
            r60 = rolling60.iloc[-1]
            if r60 < 0.50:
                alerts.append(
                    f"LONG-TERM DEGRADATION: Rolling 60-day accuracy = "
                    f"{r60*100:.1f}% (below 50%). Model may have lost predictive power."
                )

        # Calibration check: is the model over-predicting BUY?
        if len(buys) > 0 and len(cash) > 0:
            buy_pct = len(buys) / n_filled * 100
            if buy_pct > 80:
                alerts.append(
                    f"BIAS: Model predicting BUY {buy_pct:.0f}% of the time. "
                    f"May be over-bullish."
                )
            elif buy_pct < 20:
                alerts.append(
                    f"BIAS: Model predicting CASH {100-buy_pct:.0f}% of the time. "
                    f"May be over-bearish."
                )

        # Consecutive wrong predictions
        if n_filled >= 5:
            recent_correct = filled_log['correct'].tail(10).values
            consec_wrong = 0
            for c in reversed(recent_correct):
                if not c:
                    consec_wrong += 1
                else:
                    break
            if consec_wrong >= 5:
                alerts.append(
                    f"STREAK: {consec_wrong} consecutive wrong predictions. "
                    f"Model may be in a bad regime."
                )

        if alerts:
            for alert in alerts:
                print(f"  *** {alert}")
        else:
            print(f"  No alerts — model performing within normal parameters.")

        print()
        elapsed = time.time() - t0
        print(f"TIME: {elapsed:.1f} seconds")
        print("=" * 80)
        return True


if __name__ == '__main__':
    SPYTimingTracker().run()
