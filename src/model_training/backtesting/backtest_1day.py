"""
================================================================================
05_01 BACKTEST POOLED 1-DAY STRATEGY
================================================================================
Purpose:  Simulate realistic P&L for the pooled 1-day cross-sectional model.
          Uses IBKR tiered commission structure and realistic execution.

Strategy: Each day, buy top N stocks at close, sell next day at close.
          With $2K capital + PDT constraints, we run variants:
            A) Top-1 pick (concentrated, no PDT issue with overnight holds)
            B) Top-3 picks (equal weight)
            C) Top-5 picks (equal weight)
          Compare to: SPY buy-and-hold, equal-weight S&P500 daily

IBKR Tiered Commissions (US Equities):
  - $0.0035/share
  - Minimum $0.35 per order
  - Maximum 1.0% of trade value
  - Regulatory: ~$0.0000278/share (FINRA TAF) + ~$0.0000278 * value (SEC)
  - We simplify to: $0.0035/share, min $0.35, max 1% + $0.01 regulatory flat

Execution:
  - Buy at day T close price, sell at day T+1 close price
  - Actual forward 1-day returns from OOS validation data
  - Slippage: 0.02% per side (conservative for liquid S&P500 stocks)

Created: 2026-02-08
================================================================================
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class IBKR_Backtester:
    """Backtest pooled 1-day strategy with IBKR fees."""

    def __init__(self, initial_capital=2000.0):
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.preds_csv = os.path.join(self.base_path, '#_model_training',
                                      'pooled_predictions_validation_1day.csv')
        self.prices_csv = os.path.join(self.base_path, '#_feature_engineering',
                                       'ml_features_master.csv')
        self.output_dir = os.path.join(self.base_path, '#_model_training')
        self.initial_capital = initial_capital

        # IBKR tiered commissions
        self.comm_per_share = 0.0035
        self.comm_min_order = 0.35
        self.comm_max_pct = 0.01  # 1% of trade value
        self.regulatory_flat = 0.01  # simplified regulatory fees per order

        # Execution assumptions
        self.slippage_pct = 0.0002  # 0.02% per side (2 bps)

    def _ibkr_commission(self, shares, price):
        """Calculate IBKR tiered commission for a trade."""
        trade_value = shares * price
        comm = shares * self.comm_per_share
        comm = max(comm, self.comm_min_order)
        comm = min(comm, trade_value * self.comm_max_pct)
        comm += self.regulatory_flat
        return comm

    def _simulate_strategy(self, daily_preds, price_lookup, top_n, label):
        """Simulate a top-N strategy day by day."""
        dates = sorted(daily_preds['date'].unique())
        capital = self.initial_capital
        equity_curve = []
        trade_log = []
        total_commissions = 0
        total_slippage = 0

        for date in dates:
            day_preds = daily_preds[daily_preds['date'] == date].copy()
            if len(day_preds) < top_n:
                equity_curve.append({'date': date, 'equity': capital, 'daily_return': 0})
                continue

            # Pick top N by ensemble probability
            picks = day_preds.nlargest(top_n, 'ensemble_prob')

            # Equal weight allocation
            alloc_per_stock = capital / top_n

            day_return_total = 0
            day_commission_total = 0
            day_slippage_total = 0

            for _, pick in picks.iterrows():
                symbol = pick['symbol']
                fwd_return = pick['fwd_return_1d_raw']  # actual next-day return

                # Get price for position sizing (fast dict lookup)
                price = price_lookup.get((date, symbol))
                if price is None or price <= 0 or np.isnan(price):
                    continue

                # Calculate shares (whole shares only)
                shares = int(alloc_per_stock / price)
                if shares < 1:
                    # Stock too expensive for allocation — skip
                    continue

                actual_trade_value = shares * price

                # Commissions: buy + sell
                buy_comm = self._ibkr_commission(shares, price)
                sell_comm = self._ibkr_commission(shares, price * (1 + fwd_return))
                commission = buy_comm + sell_comm

                # Slippage: buy slightly higher, sell slightly lower
                slippage_cost = actual_trade_value * self.slippage_pct * 2  # both sides

                # P&L
                gross_pnl = actual_trade_value * fwd_return
                net_pnl = gross_pnl - commission - slippage_cost

                day_return_total += net_pnl
                day_commission_total += commission
                day_slippage_total += slippage_cost

                trade_log.append({
                    'date': date,
                    'symbol': symbol,
                    'price': price,
                    'shares': shares,
                    'trade_value': actual_trade_value,
                    'fwd_return': fwd_return,
                    'gross_pnl': gross_pnl,
                    'commission': commission,
                    'slippage': slippage_cost,
                    'net_pnl': net_pnl,
                    'prob': pick['ensemble_prob'],
                })

            capital += day_return_total
            total_commissions += day_commission_total
            total_slippage += day_slippage_total

            daily_return = day_return_total / (capital - day_return_total) if (capital - day_return_total) > 0 else 0

            equity_curve.append({
                'date': date,
                'equity': capital,
                'daily_return': daily_return,
                'commission': day_commission_total,
                'slippage': day_slippage_total,
            })

        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

        return equity_df, trade_df, total_commissions, total_slippage

    def _compute_metrics(self, equity_df, label, total_comm, total_slip):
        """Compute strategy performance metrics."""
        equity_df = equity_df.copy()
        initial = self.initial_capital
        final = equity_df['equity'].iloc[-1]
        total_return = (final / initial - 1) * 100

        n_days = len(equity_df)
        n_years = n_days / 252

        # Annualized return
        if n_years > 0 and final > 0:
            ann_return = ((final / initial) ** (1 / n_years) - 1) * 100
        else:
            ann_return = 0

        # Daily returns
        daily_rets = equity_df['daily_return'].values
        avg_daily = np.mean(daily_rets) * 100
        std_daily = np.std(daily_rets) * 100

        # Sharpe
        sharpe_daily = np.mean(daily_rets) / np.std(daily_rets) if np.std(daily_rets) > 0 else 0
        sharpe_annual = sharpe_daily * np.sqrt(252)

        # Max drawdown
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_dd = drawdown.min() * 100

        # Win rate
        win_days = (daily_rets > 0).sum()
        total_trading_days = (daily_rets != 0).sum()
        win_rate = win_days / total_trading_days * 100 if total_trading_days > 0 else 0

        # Profit factor
        gross_wins = daily_rets[daily_rets > 0].sum()
        gross_losses = abs(daily_rets[daily_rets < 0].sum())
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0

        return {
            'Strategy': label,
            'Initial': f'${initial:,.0f}',
            'Final': f'${final:,.2f}',
            'Total Return': f'{total_return:+.1f}%',
            'Ann. Return': f'{ann_return:+.1f}%',
            'Ann. Sharpe': f'{sharpe_annual:.2f}',
            'Max Drawdown': f'{max_dd:.1f}%',
            'Calmar': f'{calmar:.2f}',
            'Win Rate': f'{win_rate:.1f}%',
            'Profit Factor': f'{profit_factor:.2f}',
            'Avg Daily': f'{avg_daily:+.3f}%',
            'Trading Days': n_days,
            'Total Commissions': f'${total_comm:,.2f}',
            'Total Slippage': f'${total_slip:,.2f}',
            'Comm % of Capital': f'{total_comm/initial*100:.1f}%',
        }

    def _spy_benchmark(self, prices_df, dates):
        """Compute SPY buy-and-hold benchmark."""
        spy = prices_df[prices_df['symbol'] == 'SPY'].copy()
        spy = spy[spy['date'].isin(dates)].sort_values('date')

        if len(spy) < 2:
            return None

        initial_price = spy['Close'].iloc[0]
        shares = int(self.initial_capital / initial_price)
        invested = shares * initial_price
        comm = self._ibkr_commission(shares, initial_price)  # buy once

        equity_curve = []
        for _, row in spy.iterrows():
            equity = shares * row['Close'] - comm  # subtract one-time buy commission
            equity_curve.append({
                'date': row['date'],
                'equity': equity,
            })

        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df['daily_return'] = equity_df['equity'].pct_change().fillna(0)

        final = equity_df['equity'].iloc[-1]
        total_return = (final / self.initial_capital - 1) * 100
        n_years = len(equity_df) / 252
        ann_return = ((final / self.initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_dd = drawdown.min() * 100

        daily_rets = equity_df['daily_return'].values
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0

        return {
            'Strategy': 'SPY Buy & Hold',
            'Initial': f'${self.initial_capital:,.0f}',
            'Final': f'${final:,.2f}',
            'Total Return': f'{total_return:+.1f}%',
            'Ann. Return': f'{ann_return:+.1f}%',
            'Ann. Sharpe': f'{sharpe:.2f}',
            'Max Drawdown': f'{max_dd:.1f}%',
            'Calmar': f'{ann_return/abs(max_dd):.2f}' if abs(max_dd) > 0 else '0',
            'Win Rate': f'{(daily_rets > 0).mean()*100:.1f}%',
            'Profit Factor': f'{daily_rets[daily_rets>0].sum()/abs(daily_rets[daily_rets<0].sum()):.2f}' if daily_rets[daily_rets<0].sum() != 0 else 'inf',
            'Avg Daily': f'{np.mean(daily_rets)*100:+.3f}%',
            'Trading Days': len(equity_df),
            'Total Commissions': f'${comm:,.2f}',
            'Total Slippage': '$0.00',
            'Comm % of Capital': f'{comm/self.initial_capital*100:.1f}%',
        }, equity_df

    def run(self):
        t0 = time.time()
        print("=" * 80)
        print("STEP 05_01: BACKTEST POOLED 1-DAY STRATEGY")
        print("=" * 80)
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Broker: IBKR (tiered commissions)")
        print(f"Commission: ${self.comm_per_share}/share, min ${self.comm_min_order}/order")
        print(f"Slippage: {self.slippage_pct*100:.2f}% per side")
        print(f"Execution: Buy at close day T, sell at close day T+1")
        print("=" * 80)

        # ---- 1. Load predictions ----
        print("\n[1/5] Loading OOS predictions...")
        preds = pd.read_csv(self.preds_csv)
        preds['date'] = pd.to_datetime(preds['date']).dt.strftime('%Y-%m-%d')
        print(f"  OK {len(preds):,} predictions, {preds['date'].nunique()} trading days")

        # ---- 2. Load prices ----
        print("\n[2/5] Loading close prices...")
        prices = pd.read_csv(self.prices_csv, usecols=['date', 'symbol', 'Close'])
        prices['date'] = pd.to_datetime(prices['date']).dt.strftime('%Y-%m-%d')
        print(f"  OK {len(prices):,} price rows")

        # Filter to prediction dates only
        pred_dates = set(preds['date'].unique())
        prices = prices[prices['date'].isin(pred_dates)]
        print(f"  OK Filtered to {len(prices):,} rows for prediction dates")

        # Pre-build fast price lookup dict: (date, symbol) -> Close
        price_lookup = dict(zip(zip(prices['date'], prices['symbol']), prices['Close']))
        print(f"  OK Built price lookup: {len(price_lookup):,} entries")

        # ---- 3. Run strategies ----
        print("\n[3/5] Running strategy simulations...")

        strategies = [
            (1, 'Top-1 Pick'),
            (3, 'Top-3 Picks'),
            (5, 'Top-5 Picks'),
        ]

        all_results = []
        all_equity_curves = {}
        all_trade_logs = {}

        for top_n, label in strategies:
            print(f"\n  --- {label} ---")
            equity_df, trade_df, total_comm, total_slip = self._simulate_strategy(
                preds, price_lookup, top_n, label
            )
            metrics = self._compute_metrics(equity_df, label, total_comm, total_slip)
            all_results.append(metrics)
            all_equity_curves[label] = equity_df
            all_trade_logs[label] = trade_df

            print(f"  Final equity: {metrics['Final']}")
            print(f"  Total return: {metrics['Total Return']}")
            print(f"  Sharpe: {metrics['Ann. Sharpe']}")
            print(f"  Max DD: {metrics['Max Drawdown']}")
            print(f"  Commissions: {metrics['Total Commissions']}")

        # ---- 4. SPY Benchmark ----
        print("\n  --- SPY Buy & Hold ---")
        spy_result, spy_equity = self._spy_benchmark(prices, pred_dates)
        if spy_result:
            all_results.append(spy_result)
            all_equity_curves['SPY Buy & Hold'] = spy_equity
            print(f"  Final equity: {spy_result['Final']}")
            print(f"  Total return: {spy_result['Total Return']}")
            print(f"  Sharpe: {spy_result['Ann. Sharpe']}")

        # ---- 5. Confidence-filtered variant ----
        print("\n  --- Top-1 with Confidence Filter (prob > 0.55) ---")
        high_conf_preds = preds[preds['ensemble_prob'] > 0.55].copy()
        conf_dates = high_conf_preds['date'].nunique()
        print(f"  Dates with qualifying picks: {conf_dates} / {len(pred_dates)} ({conf_dates/len(pred_dates)*100:.0f}%)")

        if conf_dates > 50:
            eq_conf, tr_conf, comm_conf, slip_conf = self._simulate_strategy(
                high_conf_preds, price_lookup, 1, 'Top-1 (conf>0.55)'
            )
            # For days with no qualifying pick, capital sits idle
            # Merge with full date range
            all_dates_df = pd.DataFrame({'date': pd.to_datetime(sorted(pred_dates))})
            eq_conf_full = all_dates_df.merge(eq_conf, on='date', how='left')
            # Forward-fill equity for idle days
            eq_conf_full['equity'] = eq_conf_full['equity'].ffill().fillna(self.initial_capital)
            eq_conf_full['daily_return'] = eq_conf_full['daily_return'].fillna(0)

            metrics_conf = self._compute_metrics(eq_conf_full, 'Top-1 (conf>0.55)', comm_conf, slip_conf)
            all_results.append(metrics_conf)
            all_equity_curves['Top-1 (conf>0.55)'] = eq_conf_full
            print(f"  Final equity: {metrics_conf['Final']}")
            print(f"  Total return: {metrics_conf['Total Return']}")
            print(f"  Sharpe: {metrics_conf['Ann. Sharpe']}")

        # ---- Results ----
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Period: {min(pred_dates)} to {max(pred_dates)} ({len(pred_dates)} trading days, ~{len(pred_dates)/252:.1f} years)")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print()

        results_df = pd.DataFrame(all_results)
        display_cols = ['Strategy', 'Final', 'Total Return', 'Ann. Return',
                       'Ann. Sharpe', 'Max Drawdown', 'Win Rate', 'Profit Factor',
                       'Total Commissions']
        print(results_df[display_cols].to_string(index=False))

        # ---- Trade analysis ----
        print("\n" + "-" * 80)
        print("TRADE ANALYSIS (Top-1 Strategy)")
        print("-" * 80)
        if len(all_trade_logs.get('Top-1 Pick', pd.DataFrame())) > 0:
            tl = all_trade_logs['Top-1 Pick']
            print(f"  Total trades: {len(tl)}")
            print(f"  Win rate: {(tl['net_pnl'] > 0).mean()*100:.1f}%")
            print(f"  Avg winner: ${tl[tl['net_pnl'] > 0]['net_pnl'].mean():.2f}")
            print(f"  Avg loser: ${tl[tl['net_pnl'] < 0]['net_pnl'].mean():.2f}")
            print(f"  Best trade: ${tl['net_pnl'].max():.2f} ({tl.loc[tl['net_pnl'].idxmax(), 'symbol']})")
            print(f"  Worst trade: ${tl['net_pnl'].min():.2f} ({tl.loc[tl['net_pnl'].idxmin(), 'symbol']})")
            print(f"  Avg trade value: ${tl['trade_value'].mean():.0f}")
            print(f"  Avg shares/trade: {tl['shares'].mean():.0f}")
            print(f"  Avg commission/trade: ${tl['commission'].mean():.2f}")

            # Most traded symbols
            symbol_counts = tl['symbol'].value_counts().head(10)
            print(f"\n  Most picked symbols:")
            for sym, cnt in symbol_counts.items():
                sym_pnl = tl[tl['symbol'] == sym]['net_pnl'].sum()
                print(f"    {sym}: {cnt} times, net P&L ${sym_pnl:+.2f}")

            # Monthly breakdown
            tl['month'] = pd.to_datetime(tl['date']).dt.to_period('M')
            monthly = tl.groupby('month')['net_pnl'].sum()
            print(f"\n  Monthly P&L (last 12 months):")
            for month, pnl in monthly.tail(12).items():
                bar = '+' * max(0, int(pnl / 5)) + '-' * max(0, int(-pnl / 5))
                print(f"    {month}: ${pnl:+8.2f} {bar}")

        # ---- Commission impact analysis ----
        print("\n" + "-" * 80)
        print("COMMISSION IMPACT ANALYSIS")
        print("-" * 80)
        for label in ['Top-1 Pick', 'Top-3 Picks', 'Top-5 Picks']:
            if label in all_trade_logs and len(all_trade_logs[label]) > 0:
                tl = all_trade_logs[label]
                gross = tl['gross_pnl'].sum()
                comms = tl['commission'].sum()
                slips = tl['slippage'].sum()
                net = tl['net_pnl'].sum()
                print(f"  {label}:")
                print(f"    Gross P&L:    ${gross:+,.2f}")
                print(f"    Commissions:  ${comms:,.2f} ({comms/abs(gross)*100:.1f}% of gross)" if gross != 0 else f"    Commissions:  ${comms:,.2f}")
                print(f"    Slippage:     ${slips:,.2f}")
                print(f"    Net P&L:      ${net:+,.2f}")
                print()

        # ---- Year-by-year breakdown ----
        print("-" * 80)
        print("YEAR-BY-YEAR RETURNS (Top-1 Pick)")
        print("-" * 80)
        if 'Top-1 Pick' in all_equity_curves:
            eq = all_equity_curves['Top-1 Pick'].copy()
            eq['year'] = eq['date'].dt.year
            for year in sorted(eq['year'].unique()):
                yr_data = eq[eq['year'] == year]
                start_eq = yr_data['equity'].iloc[0]
                end_eq = yr_data['equity'].iloc[-1]
                yr_return = (end_eq / start_eq - 1) * 100
                yr_dd = ((yr_data['equity'] - yr_data['equity'].cummax()) / yr_data['equity'].cummax()).min() * 100
                print(f"  {year}: {yr_return:+.1f}% (drawdown: {yr_dd:.1f}%)")

        # Save
        results_path = os.path.join(self.output_dir, 'backtest_results_1day.csv')
        results_df.to_csv(results_path, index=False)

        # Save equity curves
        for label, eq in all_equity_curves.items():
            safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('>', '')
            eq_path = os.path.join(self.output_dir,
                                   f'backtest_equity_{safe_label}.csv')
            eq.to_csv(eq_path, index=False)

        elapsed = time.time() - t0
        print(f"\n  OK Results saved to {results_path}")
        print(f"\nTIME: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print("=" * 80)


if __name__ == '__main__':
    IBKR_Backtester(initial_capital=2000.0).run()
