"""
================================================================================
06_09  PROMOTE REGIME MODEL TO PRODUCTION
================================================================================
Atomic deployment script:
  1. Fit isotonic calibration on the regime model's OOS preds (drop_drifted
     variant) from regime_retrain_oos.csv
  2. Back up the current production model + calibration
  3. Promote spy_timing_model_regime.pkl -> spy_timing_model.pkl
  4. Write spy_timing_calibration.pkl with the new isotonic + threshold
  5. Print before/after summary

After this runs, 06_02 (predict) picks up both files automatically on the
next pipeline run.

Guardrail: aborts if the new eval AUC drops more than `GUARDRAIL_AUC_DROP`
versus the last promoted model. Pass `--force` to override.
================================================================================
"""

import os, sys, json, pickle, shutil, time
from datetime import date
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# Guardrail: abort swap if new eval AUC drops more than this vs last promoted AUC
GUARDRAIL_AUC_DROP = 0.03
FORCE = '--force' in sys.argv

BASE      = CONFIG_BASE_PATH  # Set in config.py
MT_DIR    = os.path.join(BASE, '#_model_training')
MODEL_DIR = os.path.join(MT_DIR, 'trained_models', 'spy_timing')

PROD_PKL   = os.path.join(MODEL_DIR, 'spy_timing_model.pkl')
REGIME_PKL = os.path.join(MODEL_DIR, 'spy_timing_model_regime.pkl')
CAL_PKL    = os.path.join(MODEL_DIR, 'spy_timing_calibration.pkl')

OOS_CSV    = os.path.join(MT_DIR, 'regime_retrain_oos.csv')
LIVE_CSV   = os.path.join(MT_DIR, 'spy_timing_prediction_log.csv')
GUARD_JSON = os.path.join(MODEL_DIR, 'promote_history.json')

# Cost proxy (must match 06_06 walk-forward backtest)
ROUND_TRIP_COST = 2 * (0.0001 + 0.0007)


def expected_pnl(df, threshold):
    """Compute net return + Sharpe at a candidate decision threshold,
    using the same cost model as the walk-forward backtest."""
    p_col = 'prob_cal'
    long_today = (df[p_col].values > threshold).astype(int)
    flips = np.abs(np.diff(np.concatenate([[0], long_today, [0]])))
    n_trips = flips.sum() // 2
    gross = (long_today * df['fwd_ret'].values).sum()
    cost = n_trips * ROUND_TRIP_COST
    net = gross - cost
    daily_strat = long_today * df['fwd_ret'].values
    flip_days = np.where(np.abs(np.diff(np.concatenate([[0], long_today]))) > 0)[0]
    if len(flip_days) > 0:
        daily_strat[flip_days] -= ROUND_TRIP_COST / 2
    sharpe = daily_strat.mean() / daily_strat.std() * np.sqrt(252) if daily_strat.std() > 0 else 0
    return {
        'threshold': threshold, 'invested_pct': long_today.mean()*100,
        'n_trips': int(n_trips), 'net_ret': net*100, 'sharpe': sharpe,
    }


def main():
    t0 = time.time()
    print("=" * 78); print("06_09  PROMOTE REGIME MODEL"); print("=" * 78)

    # --- Preflight ---
    for path, label in [(PROD_PKL,'prod'), (REGIME_PKL,'regime'), (OOS_CSV,'oos csv')]:
        if not os.path.exists(path):
            print(f"  [ABORT] missing {label}: {path}"); return
    with open(REGIME_PKL,'rb') as f: regime = pickle.load(f)
    print(f"  Regime model: variant={regime.get('variant')}  features={len(regime['features'])}  "
          f"start={regime.get('regime_start')}")

    # --- Build calibration data: drop_drifted variant OOS ---
    print("\n[1/5] Assembling calibration dataset...")
    oos = pd.read_csv(OOS_CSV, parse_dates=['date'])
    variant = regime.get('variant', 'drop_drifted')
    oos = oos[oos['variant'] == variant].copy()
    oos = oos.rename(columns={'new_prob': 'prob'})[['date','target','fwd_ret','prob']]
    print(f"  Regime OOS preds ({variant}): {len(oos)}  "
          f"({oos['date'].min().date()} -> {oos['date'].max().date()})")

    # Calibration fit = pre-2026; eval = 2026
    calib = oos[oos['date'] < '2026-01-01'].copy()
    evalw = oos[oos['date'] >= '2026-01-01'].copy()
    print(f"  Calibration fit (pre-2026): {len(calib)}")
    print(f"  Eval (2026 OOS):            {len(evalw)}")
    if len(calib) < 50 or len(evalw) < 20:
        print("  [ABORT] insufficient data"); return

    # --- Fit isotonic ---
    print("\n[2/5] Fitting isotonic regression...")
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(calib['prob'].values, calib['target'].values)
    calib['prob_cal'] = iso.transform(calib['prob'].values)
    evalw['prob_cal'] = iso.transform(evalw['prob'].values)

    def metrics(df, p_col):
        return dict(
            n=len(df),
            auc=roc_auc_score(df['target'], df[p_col]) if df['target'].nunique()>1 else np.nan,
            brier=brier_score_loss(df['target'], df[p_col]),
        )
    print(f"  Calib raw   : {metrics(calib,'prob')}")
    print(f"  Calib cal   : {metrics(calib,'prob_cal')}")
    print(f"  Eval  raw   : {metrics(evalw,'prob')}")
    print(f"  Eval  cal   : {metrics(evalw,'prob_cal')}")

    # Reliability
    evalw['bin'] = pd.cut(evalw['prob_cal'], bins=[0,0.3,0.45,0.55,0.7,1.0], include_lowest=True)
    rel = evalw.groupby('bin', observed=True).agg(n=('target','size'), pred=('prob_cal','mean'),
                                                   actual=('target','mean'))
    print("  Reliability (calibrated, 2026):"); print(rel.to_string())

    # --- Threshold sweep on calib window ---
    print("\n[3/5] Threshold sweep (optimise calib Sharpe)...")
    grid = np.round(np.arange(0.40, 0.71, 0.01), 2)
    sweep = []
    for th in grid:
        mc = expected_pnl(calib, th); me = expected_pnl(evalw, th)
        sweep.append({'threshold': th,
                      'calib_invested': mc['invested_pct'], 'calib_net': mc['net_ret'], 'calib_sharpe': mc['sharpe'],
                      'eval_invested':  me['invested_pct'],  'eval_net':  me['net_ret'],  'eval_sharpe':  me['sharpe']})
    sw = pd.DataFrame(sweep)
    # Pick threshold maximising calib Sharpe with >=25% invested
    cand = sw[sw['calib_invested'] >= 25]
    if cand.empty: cand = sw
    best = cand.loc[cand['calib_sharpe'].idxmax()]
    th_star = float(best['threshold'])
    cur  = sw[np.isclose(sw['threshold'], 0.50)].iloc[0]
    print(sw.iloc[::4].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"  Current 0.50: calib net={cur['calib_net']:+.1f}% Sharpe={cur['calib_sharpe']:.2f} | "
          f"eval net={cur['eval_net']:+.1f}% Sharpe={cur['eval_sharpe']:.2f}")
    print(f"  Recommended {th_star:.2f}: calib net={best['calib_net']:+.1f}% Sharpe={best['calib_sharpe']:.2f} | "
          f"eval net={best['eval_net']:+.1f}% Sharpe={best['eval_sharpe']:.2f}")

    sw.to_csv(os.path.join(MT_DIR, 'regime_threshold_sweep.csv'), index=False)

    # --- Guardrail: compare new eval AUC vs last promoted ---
    new_eval_auc = metrics(evalw, 'prob_cal')['auc']
    history = []
    if os.path.exists(GUARD_JSON):
        try: history = json.load(open(GUARD_JSON))
        except Exception: history = []
    prev_auc = history[-1]['eval_auc'] if history else None
    if prev_auc is not None:
        delta = new_eval_auc - prev_auc
        print(f"\n[Guardrail] prev eval AUC={prev_auc:.4f}  new={new_eval_auc:.4f}  delta={delta:+.4f}")
        if delta < -GUARDRAIL_AUC_DROP and not FORCE:
            print(f"  [ABORT] new AUC dropped > {GUARDRAIL_AUC_DROP} vs last promote. Use --force to override.")
            return
    else:
        print(f"\n[Guardrail] no prior history; new eval AUC={new_eval_auc:.4f} (baseline)")

    # --- Back up production model + existing calibration ---
    print("\n[4/5] Backing up + swapping production model...")
    stamp = date.today().strftime('%Y%m%d')
    legacy_pkl = os.path.join(MODEL_DIR, f'spy_timing_model_legacy_{stamp}.pkl')
    if not os.path.exists(legacy_pkl):
        shutil.copy2(PROD_PKL, legacy_pkl)
        print(f"  Backed up old model -> {legacy_pkl}")
    else:
        print(f"  Backup already exists: {legacy_pkl}")

    if os.path.exists(CAL_PKL):
        legacy_cal = os.path.join(MODEL_DIR, f'spy_timing_calibration_legacy_{stamp}.pkl')
        if not os.path.exists(legacy_cal):
            shutil.copy2(CAL_PKL, legacy_cal)
            print(f"  Backed up old calibration -> {legacy_cal}")

    # Promote regime model
    shutil.copy2(REGIME_PKL, PROD_PKL)
    print(f"  Promoted regime model -> {PROD_PKL}")

    # Save calibration
    cal_obj = {
        'isotonic': iso,
        'threshold': th_star,
        'model_variant': variant,
        'fit_window': f"{calib['date'].min().date()} .. {calib['date'].max().date()}",
        'fit_n': int(len(calib)),
        'eval_n': int(len(evalw)),
        'metrics_eval_raw':  metrics(evalw,'prob'),
        'metrics_eval_cal':  metrics(evalw,'prob_cal'),
        'created_at': pd.Timestamp.now().isoformat(),
    }
    with open(CAL_PKL, 'wb') as f:
        pickle.dump(cal_obj, f)
    print(f"  Calibration saved -> {CAL_PKL}")

    # Append to promote history
    history.append({
        'promoted_at': pd.Timestamp.now().isoformat(),
        'variant': variant,
        'n_features': len(regime['features']),
        'eval_auc': float(new_eval_auc),
        'eval_brier': float(metrics(evalw,'prob_cal')['brier']),
        'threshold': float(th_star),
        'calib_window': cal_obj['fit_window'],
        'forced': bool(FORCE),
    })
    history = history[-12:]  # keep last 12 promotes
    json.dump(history, open(GUARD_JSON,'w'), indent=2)
    print(f"  History updated -> {GUARD_JSON}  (entries={len(history)})")

    # --- Summary ---
    print("\n[5/5] BEFORE/AFTER (2026 live, source: regime_retrain_oos.csv):")
    if os.path.exists(LIVE_CSV):
        live = pd.read_csv(LIVE_CSV)
        live = live[live['actual_next_day_return'].notna()]
        hit = live['correct'].astype(str).str.lower().eq('true').mean()
        print(f"  Production live log so far: n={len(live)}, hit_rate={hit*100:.1f}%")
    print(f"  Regime model (cal, p>{th_star:.2f}) on 2026 OOS: "
          f"net={best['eval_net']:+.1f}%  Sharpe={best['eval_sharpe']:.2f}  "
          f"invested={best['eval_invested']:.0f}%  AUC={metrics(evalw,'prob_cal')['auc']:.3f}")
    print(f"\nDone in {time.time()-t0:.1f}s")
    print("Next: ensure 06_02 (predict) applies the calibration (already wired).")


if __name__ == '__main__':
    main()
