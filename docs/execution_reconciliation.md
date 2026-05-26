# Execution Reconciliation (Step 10c)

> "Model accuracy is not realized P&L."
> This is the only place in the pipeline that proves whether the live stack
> is actually capturing the edge the backtest promised.

## What it does

After the SPY timing **tracker** (10a) backfills yesterday's outcome and the
**predictor** (10b) generates today's BUY/CASH signal, the **reconciler** (10c)
joins three independent sources of truth:

| Source                                 | What it tells us              |
| -------------------------------------- | ----------------------------- |
| `spy_timing_prediction_log.csv`        | What the model **said**       |
| `spy_timing_trade_log.csv`             | What the trader **attempted** |
| IBKR `reqExecutions()` (last 7 days)   | What the broker **filled**    |

It then writes one row per (signal, trade) pair to
`spy_reconciliation_log.csv` and prints a dashboard.

## Metrics computed per trade

| Metric                          | Definition                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| `execution_ok`                  | Signal matched action (BUY→Filled, CASH→no-trade/closed)                            |
| `slippage_bps`                  | `(fill_price − signal_close) / signal_close × 10_000` (positive = paid more on BUY) |
| `realized_intraday_pnl`         | `shares × (trade_day_close − fill_price)`                                            |
| `alpha_capture_pct`             | `realized_close_return / theoretical_next_day_return`                                |

When TWS is reachable the fill price is upgraded to the precise VWAP across
partial fills and the commission column is populated from
`CommissionReport`. When TWS is closed the reconciler degrades gracefully and
still produces a useful log from the CSV trade log alone.

## Alert thresholds (configurable in `CONFIG`)

| Trigger                                           | Default    |
| ------------------------------------------------- | ---------- |
| `|mean slippage| > slippage_alert_bps`            | 30 bps     |
| Rolling 20-trade `alpha_capture < threshold`      | 70%        |
| Any execution failure in last 30 trades           | always     |

When any alert fires and `send_email_on_alert` is true (and Gmail credentials
are present in `.env` as `EMAIL_USER` / `EMAIL_PASS` / `EMAIL_TO`), a single
summary email is sent with the alert list and the dashboard.

## Why three-way reconciliation matters

Most retail systems compare the model log to the broker log and call it done.
That misses a real class of bugs:

- **Signal-side bugs**: the predictor wrote `BUY` but the trader read stale
  CSV and did nothing → caught by `execution_ok = False`.
- **Routing bugs**: the order went out but was cancelled (insufficient buying
  power, market closed, etc.) → caught by `EXECUTION_FAIL` note.
- **Cost drift**: real slippage runs hot vs the 10 bps assumed in
  `06_06_walk_forward_spy_timing.py` → caught by the bps alert and forces
  a re-tune of the cost model.
- **Edge decay**: the model is still accurate (high AUC in tracker) but the
  alpha capture is collapsing because fills are getting worse → caught by
  the rolling alpha-capture metric, not by any pure-ML diagnostic.

## Exit policy

The reconciler **always exits 0**, even when alerts fire. Alerts are surfaced
via the dashboard, the CSV log, and (optionally) email. We never want a
slippage warning to fail the entire evening pipeline and prevent the next
day's signal from being generated.
