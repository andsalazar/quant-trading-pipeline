"""
================================================================================
06_04 SPY TIMING — EXECUTION RECONCILIATION
================================================================================
Purpose:  Close the loop between (a) what the MODEL predicted, (b) what the
          TRADER did, and (c) how it actually FILLED at IBKR. Quantifies
          slippage, alpha capture, and execution failures.

Workflow:  Run DAILY in the evening pipeline (after 06_03 tracker and 06_02
           next-day predictor — i.e. as Step 10c).

           For each prediction in spy_timing_prediction_log.csv:
             1. Find the matching trade in spy_timing_trade_log.csv
                (trade fires on the next trading day after signal_date)
             2. Optionally enrich with live IBKR ib.reqExecutions() if TWS is
                reachable — captures commission + precise VWAP fill price
             3. Compute reconciliation metrics:
                   - executed_as_intended  (did model say BUY → did we buy?)
                   - slippage_vs_signal_bps (fill price vs signal close)
                   - same_day_intraday_pnl  (close on trade-day vs fill)
                   - alpha_capture_pct      (realized / theoretical)
             4. Append to spy_reconciliation_log.csv
             5. Print dashboard + alert on:
                   - signal=BUY but no fill (execution failure)
                   - |slippage| > 30 bps (3× backtest assumption)
                   - 20d rolling alpha_capture < 70%
             6. Optional Gmail alert (only when alerts present)

Why:      Model accuracy ≠ realized $ PnL. This is the only place that proves
          whether the live stack is capturing the edge the backtest promised.

Requirements:
    pip install pandas numpy ib_insync python-dotenv

Created:  2026-05-25
================================================================================
"""

import os, sys, time, warnings, smtplib
import numpy as np
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Force UTF-8 output (matches the rest of the pipeline)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# IBKR is OPTIONAL — evening pipeline often runs without TWS open
try:
    from ib_insync import IB, ExecutionFilter, Stock
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

# .env loader (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
EMAIL_TO   = os.getenv('EMAIL_TO')


# ============================================================================
# CONFIG
# ============================================================================

BASE_PATH = CONFIG_BASE_PATH  # Set in config.py

CONFIG = {
    'prediction_log':     os.path.join(BASE_PATH, '#_model_training', 'spy_timing_prediction_log.csv'),
    'trade_log':          os.path.join(BASE_PATH, '#_model_training', 'spy_timing_trade_log.csv'),
    'reconciliation_log': os.path.join(BASE_PATH, '#_model_training', 'spy_reconciliation_log.csv'),
    'features_csv':       os.path.join(BASE_PATH, '#_feature_engineering', 'ml_features_master.csv'),
    'futures_long_csv':   os.path.join(BASE_PATH, '#_fetch_data', '#_04_fetch_futures', '00_04_futures_long.csv'),

    # IBKR (optional enrichment — captures commission + precise VWAP fill)
    'try_ibkr':           True,
    'ibkr_host':          '127.0.0.1',
    'ibkr_port':          7497,         # 7497=paper, 7496=live
    'ibkr_client_id':     7,            # Distinct from trader to avoid client_id conflicts
    'ibkr_lookback_days': 7,            # IBKR executions API limit (~7 days)

    # Alerts
    'slippage_alert_bps':   30,         # Alert if |slippage| > 30 bps
    'alpha_capture_window': 20,         # Rolling window for alpha capture
    'alpha_capture_alert':  0.70,       # Alert if 20d rolling capture < 70%

    # Email (only fires when there are alerts; requires .env with EMAIL_USER/PASS/TO)
    'send_email_on_alert':  True,
}


# ============================================================================
# EMAIL
# ============================================================================

def send_alert_email(alerts, dashboard_lines):
    """Send Gmail alert. Returns True on success, False otherwise."""
    if not alerts:
        return False
    if not CONFIG.get('send_email_on_alert'):
        return False
    if not (EMAIL_USER and EMAIL_PASS and EMAIL_TO):
        print("  [SKIP email] EMAIL_USER/EMAIL_PASS/EMAIL_TO not set in .env")
        return False

    subject = f"[SPY Reconciliation] {len(alerts)} alert(s) — {datetime.now().strftime('%Y-%m-%d')}"
    body_lines = []
    body_lines.append("Execution reconciliation produced the following alerts:\n")
    body_lines.extend(alerts)
    body_lines.append("\n\n--- Dashboard ---\n")
    body_lines.extend(dashboard_lines)
    body_lines.append(f"\n\nFull log: {CONFIG['reconciliation_log']}")

    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_TO
        msg.attach(MIMEText('\n'.join(body_lines), 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15) as s:
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        print(f"  [OK] Alert email sent")
        return True
    except Exception as e:
        print(f"  [!] Email send failed: {type(e).__name__}: {e}")
        return False


# ============================================================================
# RECONCILER
# ============================================================================

class ExecutionReconciler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictions = None
        self.trades = None
        self.price_map = {}
        self.ibkr_fills = pd.DataFrame()

    # ----------------------------------------------------------------------
    # Load CSV logs
    # ----------------------------------------------------------------------
    def load_logs(self):
        print("\n[1/6] Loading prediction & trade logs...")

        if not os.path.exists(self.cfg['prediction_log']):
            print(f"  [SKIP] No prediction log: {self.cfg['prediction_log']}")
            return False

        if not os.path.exists(self.cfg['trade_log']):
            print(f"  [SKIP] No trade log yet: {self.cfg['trade_log']}")
            print(f"         (this is expected before the first live trade)")
            return False

        self.predictions = pd.read_csv(self.cfg['prediction_log'])
        self.predictions['signal_date'] = pd.to_datetime(self.predictions['signal_date'])

        self.trades = pd.read_csv(self.cfg['trade_log'])
        self.trades['date'] = pd.to_datetime(self.trades['date'])
        self.trades['signal_date'] = pd.to_datetime(self.trades['signal_date'])

        print(f"  [OK] {len(self.predictions)} predictions, {len(self.trades)} trades")
        return True

    # ----------------------------------------------------------------------
    # Load SPY price map (for trade-day close + theoretical return)
    # ----------------------------------------------------------------------
    def load_prices(self):
        print("\n[2/6] Loading SPY price map...")

        if os.path.exists(self.cfg['features_csv']):
            df = pd.read_csv(self.cfg['features_csv'], usecols=['date', 'SPY_Close_fut'])
            df['date'] = pd.to_datetime(df['date'])
            sp = df.groupby('date')['SPY_Close_fut'].first().dropna()
            self.price_map = dict(zip(sp.index, sp.values))

        # Supplement from futures_long for the most recent days
        if os.path.exists(self.cfg['futures_long_csv']):
            fl = pd.read_csv(self.cfg['futures_long_csv'])
            spy = fl[fl['Symbol'] == 'SPY'][['Date', 'Close']].copy()
            spy['Date'] = pd.to_datetime(spy['Date']).dt.normalize()
            for _, r in spy.iterrows():
                if r['Date'] not in self.price_map:
                    self.price_map[r['Date']] = r['Close']

        print(f"  [OK] {len(self.price_map)} SPY price dates")

    # ----------------------------------------------------------------------
    # Optional: enrich from IBKR live executions
    # ----------------------------------------------------------------------
    def fetch_ibkr_fills(self):
        print("\n[3/6] Optional IBKR fill enrichment...")

        if not self.cfg['try_ibkr']:
            print("  [SKIP] Disabled in config")
            return
        if not IBKR_AVAILABLE:
            print("  [SKIP] ib_insync not installed in this env")
            return

        ib = IB()
        try:
            ib.connect(self.cfg['ibkr_host'], self.cfg['ibkr_port'],
                       clientId=self.cfg['ibkr_client_id'], timeout=4)
        except Exception as e:
            print(f"  [SKIP] TWS not reachable ({type(e).__name__}: {e})")
            print(f"         Reconciliation will proceed using CSV logs only.")
            return

        try:
            # IBKR limits executions to ~7 days back
            since = (datetime.now() - pd.Timedelta(days=self.cfg['ibkr_lookback_days'])).strftime('%Y%m%d-%H:%M:%S')
            flt = ExecutionFilter(symbol='SPY', time=since)
            fills = ib.reqExecutions(flt)

            rows = []
            for f in fills:
                rows.append({
                    'exec_id':    f.execution.execId,
                    'fill_time':  pd.to_datetime(f.execution.time),
                    'side':       f.execution.side,           # 'BOT' / 'SLD'
                    'shares':     f.execution.shares,
                    'fill_price': f.execution.price,
                    'commission': getattr(f.commissionReport, 'commission', np.nan),
                    'symbol':     f.contract.symbol,
                })
            self.ibkr_fills = pd.DataFrame(rows)
            if not self.ibkr_fills.empty:
                self.ibkr_fills['fill_date'] = self.ibkr_fills['fill_time'].dt.normalize()
                print(f"  [OK] {len(self.ibkr_fills)} SPY fills from IBKR (last "
                      f"{self.cfg['ibkr_lookback_days']}d)")
            else:
                print("  [OK] No SPY fills in IBKR lookback window")

        except Exception as e:
            print(f"  [!] IBKR enrichment failed: {e}")
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # Build the reconciliation table
    # ----------------------------------------------------------------------
    def reconcile(self):
        print("\n[4/6] Reconciling predictions × trades × fills...")

        rows = []
        for _, pred in self.predictions.iterrows():
            sig_date    = pred['signal_date']
            sig         = pred['signal']
            sig_close   = pred.get('spy_close', np.nan)
            cal_prob    = pred.get('ensemble_prob', np.nan)
            actual_ret  = pred.get('actual_next_day_return', np.nan)

            # The trade fires on the FIRST trade row where signal_date == sig_date
            matched = self.trades[self.trades['signal_date'] == sig_date]

            if matched.empty:
                # No live trade attempted for this prediction
                rows.append({
                    'signal_date': sig_date,
                    'signal': sig,
                    'signal_spy_close': sig_close,
                    'ens_prob': cal_prob,
                    'trade_date': pd.NaT,
                    'action_taken': 'NO_TRADE',
                    'fill_status': 'NO_TRADE',
                    'shares': 0,
                    'fill_price': np.nan,
                    'commission': np.nan,
                    'slippage_bps': np.nan,
                    'trade_day_close': np.nan,
                    'realized_intraday_pnl': np.nan,
                    'theoretical_next_day_return': actual_ret,
                    'alpha_capture_pct': np.nan,
                    'execution_ok': (sig == 'CASH'),  # CASH + no-trade = fine (already in cash)
                    'note': '',
                })
                continue

            t = matched.iloc[0]
            trade_date = t['date']
            action     = t.get('action_taken', '')
            status     = t.get('status', '')
            shares     = float(t.get('shares', 0) or 0)
            fill_px    = float(t.get('price', 0) or 0)
            commission = np.nan

            # IBKR enrichment for commission + exact fill price (last 7d only)
            if not self.ibkr_fills.empty:
                same_day = self.ibkr_fills[self.ibkr_fills['fill_date'] == trade_date.normalize()]
                if not same_day.empty:
                    if pd.notna(same_day['commission']).any():
                        commission = float(same_day['commission'].sum())
                    # Re-derive a VWAP fill price across partials
                    nz = same_day[same_day['shares'] > 0]
                    if len(nz):
                        vwap = (nz['fill_price'] * nz['shares']).sum() / nz['shares'].sum()
                        if abs(vwap - fill_px) / max(fill_px, 1) < 0.02:  # sanity
                            fill_px = vwap

            # Trade-day close (for same-day P&L)
            trade_day_close = self.price_map.get(trade_date.normalize())

            # ---- Metrics ----
            slippage_bps = np.nan
            if sig == 'BUY' and pd.notna(sig_close) and fill_px > 0:
                # Positive bps = paid more than signal close (bad for BUY)
                slippage_bps = (fill_px - sig_close) / sig_close * 10_000

            realized_intraday = np.nan
            if pd.notna(trade_day_close) and fill_px > 0 and shares > 0:
                # Mark-to-close P&L on the fill (intraday)
                realized_intraday = shares * (trade_day_close - fill_px)

            # Alpha capture: realized vs theoretical (close→close move the model bet on)
            alpha_capture = np.nan
            if (sig == 'BUY' and pd.notna(actual_ret) and pd.notna(sig_close)
                    and fill_px > 0 and abs(actual_ret) > 1e-6):
                if pd.notna(trade_day_close):
                    realized_ret = (trade_day_close - fill_px) / fill_px
                    alpha_capture = realized_ret / actual_ret

            execution_ok = (
                (sig == 'BUY'  and action == 'BUY_SPY'  and status == 'Filled') or
                (sig == 'CASH' and action in ('CASH', 'SELL_SPY', 'NO_TRADE'))
            )

            note = ''
            if sig == 'BUY' and status != 'Filled':
                note = f'EXECUTION_FAIL: {status}'
            elif sig == 'BUY' and shares == 0:
                note = 'EXECUTION_FAIL: zero shares'

            rows.append({
                'signal_date': sig_date,
                'signal': sig,
                'signal_spy_close': sig_close,
                'ens_prob': cal_prob,
                'trade_date': trade_date,
                'action_taken': action,
                'fill_status': status,
                'shares': shares,
                'fill_price': fill_px,
                'commission': commission,
                'slippage_bps': slippage_bps,
                'trade_day_close': trade_day_close,
                'realized_intraday_pnl': realized_intraday,
                'theoretical_next_day_return': actual_ret,
                'alpha_capture_pct': alpha_capture,
                'execution_ok': execution_ok,
                'note': note,
            })

        recon = pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True)
        print(f"  [OK] {len(recon)} reconciled rows")
        return recon

    # ----------------------------------------------------------------------
    # Dashboard + alerts
    # ----------------------------------------------------------------------
    def dashboard(self, recon):
        print("\n[5/6] Reconciliation dashboard")
        print("-" * 78)

        lines = []  # captured for email body
        def emit(s):
            print(s)
            lines.append(s)

        # Limit to BUY signals with an attempted trade, last 30 days, for stats
        live = recon[(recon['action_taken'].isin(['BUY_SPY', 'SELL_SPY']))].copy()

        if live.empty:
            print("  No live trade rows yet — dashboard will populate after first fill.")
            return [], lines

        recent = live.tail(30)
        n_total = len(recent)
        n_ok    = int(recent['execution_ok'].sum())
        n_fail  = n_total - n_ok

        buys = recent[recent['signal'] == 'BUY'].copy()
        slip_mean = buys['slippage_bps'].dropna().mean()
        slip_med  = buys['slippage_bps'].dropna().median()
        cap_mean  = buys['alpha_capture_pct'].dropna().mean()
        comm_sum  = recent['commission'].dropna().sum()

        emit(f"  Rows (last 30 trades):       {n_total}")
        emit(f"  Executed as intended:        {n_ok}/{n_total}  ({n_ok/max(n_total,1):.0%})")
        emit(f"  Execution failures:          {n_fail}")
        emit(f"  Avg slippage vs signal:      "
             f"{slip_mean:+.1f} bps (median {slip_med:+.1f})" if pd.notna(slip_mean) else
             "  Avg slippage vs signal:      n/a")
        emit(f"  Avg alpha capture:           "
             f"{cap_mean*100:+.1f}%" if pd.notna(cap_mean) else
             "  Avg alpha capture:           n/a")
        emit(f"  Commission paid (enriched):  "
             f"${comm_sum:.2f}" if pd.notna(comm_sum) and comm_sum != 0 else
             "  Commission paid (enriched):  not available (IBKR not connected)")

        # ---- Rolling alpha capture window ----
        win = self.cfg['alpha_capture_window']
        roll_cap = buys['alpha_capture_pct'].dropna().tail(win).mean()
        if pd.notna(roll_cap):
            emit(f"  Rolling {win}-trade alpha cap: {roll_cap*100:+.1f}%")

        # ---- Alerts ----
        alerts = []
        recent_fails = recent[~recent['execution_ok']]
        if len(recent_fails):
            alerts.append(f"[ALERT] {len(recent_fails)} execution failure(s) in last 30 trades")
            for _, r in recent_fails.tail(5).iterrows():
                alerts.append(f"        - {r['signal_date'].date()}: signal={r['signal']} "
                              f"action={r['action_taken']} status={r['fill_status']} note={r['note']}")

        if pd.notna(slip_mean) and abs(slip_mean) > self.cfg['slippage_alert_bps']:
            alerts.append(f"[ALERT] Mean slippage {slip_mean:+.1f} bps exceeds "
                          f"{self.cfg['slippage_alert_bps']} bps threshold")

        if pd.notna(roll_cap) and roll_cap < self.cfg['alpha_capture_alert']:
            alerts.append(f"[ALERT] Rolling {win}-trade alpha capture "
                          f"{roll_cap*100:.0f}% below {self.cfg['alpha_capture_alert']*100:.0f}% threshold")

        print("-" * 78)
        if alerts:
            print("ALERTS:")
            for a in alerts:
                print(f"  {a}")
        else:
            print("  All execution health checks passing.")

        return alerts, lines

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------
    def save(self, recon):
        print("\n[6/6] Saving reconciliation log...")
        recon.to_csv(self.cfg['reconciliation_log'], index=False)
        print(f"  [OK] {self.cfg['reconciliation_log']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("STEP 06_04: EXECUTION RECONCILIATION  (model × trader × IBKR)")
    print("=" * 80)

    r = ExecutionReconciler(CONFIG)

    if not r.load_logs():
        print("\n[DONE] Nothing to reconcile yet.")
        return 0

    r.load_prices()
    r.fetch_ibkr_fills()
    recon = r.reconcile()
    alerts, dash_lines = r.dashboard(recon)
    r.save(recon)

    # Email only when there are alerts
    if alerts:
        print("\n[email] Alerts present — attempting to send notification...")
        send_alert_email(alerts, dash_lines)

    print(f"\n[DONE] Elapsed: {time.time()-t0:.1f}s")
    # Always exit 0 — execution alerts are surfaced via stdout + reconciliation CSV.
    # We don't want a slippage alert to fail the entire evening pipeline.
    has_fail = any('EXECUTION_FAIL' in str(n) for n in recon.get('note', pd.Series([])).tolist())
    if has_fail:
        print("[NOTE] One or more EXECUTION_FAIL rows present — review spy_reconciliation_log.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
