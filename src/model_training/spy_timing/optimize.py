"""
================================================================================
06_04 SPY TIMING — OPTUNA HYPERPARAMETER OPTIMIZATION
================================================================================
Purpose:  Bayesian optimization of LightGBM + XGBoost hyperparameters
          specifically for the SPY next-day timing model.

Approach:
  - Walk-forward evaluation (same 5 folds as 06_01) to prevent lookahead bias
  - Objective: mean OOS AUC across all 5 folds
  - TPE sampler (Optuna default) with aggressive pruning
  - Separate optimization for LGB and XGB, then combined ensemble eval
  - Saves optimized params to JSON for 06_01 to load

Runtime:  ~15-30 min (100 trials x 5 folds x 2 models, but fast on SPY data)

Created:  2026-03-12
================================================================================
"""

import os, sys, warnings, json, time, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SPYHyperparamOptimizer:
    """Optimize SPY timing model hyperparameters with walk-forward CV."""

    def __init__(self, n_trials=100, timeout_per_model=1800):
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.features_csv = os.path.join(self.base_path, '#_feature_engineering', 'ml_features_master.csv')
        self.futures_long_csv = os.path.join(self.base_path, '#_fetch_data', '#_04_fetch_futures', '00_04_futures_long.csv')
        self.model_dir = os.path.join(self.base_path, '#_model_training', 'trained_models', 'spy_timing')
        self.output_dir = os.path.join(self.base_path, '#_model_training')

        self.n_trials = n_trials
        self.timeout = timeout_per_model  # seconds per model

        # Same walk-forward folds as 06_01
        self.folds = [
            {'train_end': '2020-12-31', 'test_year': 2021},
            {'train_end': '2021-12-31', 'test_year': 2022},
            {'train_end': '2022-12-31', 'test_year': 2023},
            {'train_end': '2023-12-31', 'test_year': 2024},
            {'train_end': '2024-12-31', 'test_year': 2025},
        ]

        # Same cross-asset features as 06_01
        self.cross_asset_features = [
            'SPY_Return_1d', 'QQQ_Return_1d',
            'GLD_Return', 'TLT_Return', 'GLD_Return_1d', 'TLT_Return_1d',
            'VXX_Return_1d', 'USO_Return',
            'ES_Return_1d', 'CL_Return_1d', 'NG_Return_1d', 'DX_Return_1d',
            'HG_Return_1d', 'TY_Return_1d', 'FV_Return_1d',
            'EUR_USD_return', 'GBP_USD_return', 'USD_JPY_return',
            'USD_CAD_return', 'AUD_USD_return',
            'Market_Return_Std', 'Stocks_Above_SMA20',
            'Yield_Curve_Momentum',
            'Currency_Market_Momentum_Avg', 'Currency_Market_Momentum_Dispersion',
            'month', 'quarter',
        ]

    def _compute_spy_technicals(self, daily_df):
        """Compute SPY-specific technical indicators — must match 06_01 exactly."""
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

    def prepare_data(self):
        """Load and prepare SPY timing data (same pipeline as 06_01)."""
        print("\n[1/5] Loading data...")
        df = pd.read_csv(self.features_csv)
        df['date'] = pd.to_datetime(df['date'])

        daily = df.groupby('date').first().reset_index()
        daily = daily.sort_values('date').reset_index(drop=True)
        daily = daily[daily['SPY_Close_fut'].notna()].copy()
        print(f"  OK {len(daily)} dates with SPY_Close_fut")

        # Compute technicals
        daily = self._compute_spy_technicals(daily)

        spy_tech_features = [
            'spy_ret_1d', 'spy_ret_5d', 'spy_ret_10d', 'spy_ret_20d', 'spy_ret_60d',
            'spy_vol_5d', 'spy_vol_20d', 'spy_vol_60d', 'spy_vol_ratio',
            'spy_sma5_ratio', 'spy_sma10_ratio', 'spy_sma20_ratio', 'spy_sma50_ratio',
            'spy_rsi_14', 'spy_bb_position', 'spy_gap',
            'spy_consec_up', 'spy_consec_down', 'spy_drawdown',
            'day_of_week',
        ]

        # Auto-discover regime + flow features
        regime_features = sorted([c for c in daily.columns if c.startswith('reg_')])
        flow_features = sorted([c for c in daily.columns if c.startswith('flow_')])
        print(f"  OK Auto-discovered: {len(regime_features)} regime + {len(flow_features)} flow features")

        all_features = self.cross_asset_features + spy_tech_features + regime_features + flow_features
        all_features = list(dict.fromkeys(f for f in all_features if f in daily.columns))

        # Target
        daily['spy_fwd_1d'] = daily['SPY_Close_fut'].shift(-1) / daily['SPY_Close_fut'] - 1
        daily['target'] = (daily['spy_fwd_1d'] > 0).astype(int)
        daily = daily.dropna(subset=['target', 'spy_fwd_1d'])
        daily = daily.iloc[60:].reset_index(drop=True)  # 60-day warmup

        # Drop high-NaN features
        nan_pct = daily[all_features].isna().mean()
        high_nan = nan_pct[nan_pct > 0.5].index.tolist()
        if high_nan:
            all_features = [f for f in all_features if f not in high_nan]
            print(f"  OK Dropped {len(high_nan)} high-NaN features")

        print(f"  OK {len(daily)} samples, {len(all_features)} features")
        print(f"  OK Up: {daily['target'].sum()} ({daily['target'].mean():.1%}), Down: {(~daily['target'].astype(bool)).sum()}")

        self.daily = daily
        self.all_features = all_features
        self.spy_tech_features = spy_tech_features
        self.regime_features = regime_features
        self.flow_features = flow_features

    def _walk_forward_eval(self, model_cls, params, model_type='lgb'):
        """Evaluate params across all walk-forward folds. Returns mean AUC."""
        aucs = []
        for fold in self.folds:
            train_end = pd.Timestamp(fold['train_end'])
            test_start = pd.Timestamp(f"{fold['test_year']}-01-01")
            test_end = pd.Timestamp(f"{fold['test_year']}-12-31")

            train = self.daily[self.daily['date'] <= train_end]
            test = self.daily[(self.daily['date'] >= test_start) & (self.daily['date'] <= test_end)]
            if len(test) < 10:
                continue

            X_tr, y_tr = train[self.all_features], train['target']
            X_te, y_te = test[self.all_features], test['target']

            model = model_cls(**params)
            if model_type == 'lgb':
                model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr, verbose=False)

            probs = model.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(y_te, probs)
            except ValueError:
                auc = 0.5
            aucs.append(auc)

        return np.mean(aucs) if aucs else 0.5

    def optimize_lgb(self):
        """Optimize LightGBM hyperparameters."""
        print("\n[2/5] Optimizing LightGBM...")
        print(f"  Trials: {self.n_trials}, Timeout: {self.timeout}s")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
            }
            return self._walk_forward_eval(lgb.LGBMClassifier, params, 'lgb')

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=20),
            study_name='spy_lgb',
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = study.best_params.copy()
        best.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1})

        print(f"  OK Best CV AUC: {study.best_value:.4f}")
        print(f"  OK Best params: depth={best.get('max_depth')}, leaves={best.get('num_leaves')}, "
              f"lr={best.get('learning_rate', 0):.4f}, n_est={best.get('n_estimators')}")
        print(f"  OK subsample={best.get('subsample', 0):.2f}, colsample={best.get('colsample_bytree', 0):.2f}, "
              f"min_child={best.get('min_child_samples')}")

        self.lgb_best = best
        self.lgb_study = study
        return best

    def optimize_xgb(self):
        """Optimize XGBoost hyperparameters."""
        print("\n[3/5] Optimizing XGBoost...")
        print(f"  Trials: {self.n_trials}, Timeout: {self.timeout}s")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 10, 200),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 5.0),
                'tree_method': 'gpu_hist',
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1,
            }
            return self._walk_forward_eval(xgb.XGBClassifier, params, 'xgb')

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=20),
            study_name='spy_xgb',
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = study.best_params.copy()
        best.update({'tree_method': 'gpu_hist', 'random_state': 42, 'verbosity': 0, 'n_jobs': -1})

        print(f"  OK Best CV AUC: {study.best_value:.4f}")
        print(f"  OK Best params: depth={best.get('max_depth')}, "
              f"lr={best.get('learning_rate', 0):.4f}, n_est={best.get('n_estimators')}")
        print(f"  OK subsample={best.get('subsample', 0):.2f}, colsample={best.get('colsample_bytree', 0):.2f}, "
              f"min_child_w={best.get('min_child_weight')}")

        self.xgb_best = best
        self.xgb_study = study
        return best

    def evaluate_ensemble(self):
        """Evaluate optimized LGB+XGB ensemble vs baseline on walk-forward folds."""
        print("\n[4/5] Evaluating optimized ensemble vs baseline...")

        # Baseline params (same as 06_01 defaults)
        baseline_lgb = {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': 5,
            'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'min_child_samples': 50, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        baseline_xgb = {
            'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03,
            'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 50,
            'reg_alpha': 0.5, 'reg_lambda': 2.0, 'tree_method': 'gpu_hist',
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
        }

        results = []
        for fold in self.folds:
            train_end = pd.Timestamp(fold['train_end'])
            test_start = pd.Timestamp(f"{fold['test_year']}-01-01")
            test_end = pd.Timestamp(f"{fold['test_year']}-12-31")

            train = self.daily[self.daily['date'] <= train_end]
            test = self.daily[(self.daily['date'] >= test_start) & (self.daily['date'] <= test_end)]
            if len(test) < 10:
                continue

            X_tr, y_tr = train[self.all_features], train['target']
            X_te, y_te = test[self.all_features], test['target']

            # Baseline
            bl_lgb = lgb.LGBMClassifier(**baseline_lgb)
            bl_lgb.fit(X_tr, y_tr)
            bl_xgb = xgb.XGBClassifier(**baseline_xgb)
            bl_xgb.fit(X_tr, y_tr, verbose=False)
            bl_probs = (bl_lgb.predict_proba(X_te)[:, 1] + bl_xgb.predict_proba(X_te)[:, 1]) / 2
            bl_auc = roc_auc_score(y_te, bl_probs)
            bl_acc = accuracy_score(y_te, (bl_probs > 0.5).astype(int))

            # Optimized
            opt_lgb = lgb.LGBMClassifier(**self.lgb_best)
            opt_lgb.fit(X_tr, y_tr)
            opt_xgb = xgb.XGBClassifier(**self.xgb_best)
            opt_xgb.fit(X_tr, y_tr, verbose=False)
            opt_probs = (opt_lgb.predict_proba(X_te)[:, 1] + opt_xgb.predict_proba(X_te)[:, 1]) / 2
            opt_auc = roc_auc_score(y_te, opt_probs)
            opt_acc = accuracy_score(y_te, (opt_probs > 0.5).astype(int))

            delta_auc = opt_auc - bl_auc
            delta_acc = opt_acc - bl_acc

            results.append({
                'year': fold['test_year'],
                'baseline_auc': bl_auc, 'optimized_auc': opt_auc, 'delta_auc': delta_auc,
                'baseline_acc': bl_acc, 'optimized_acc': opt_acc, 'delta_acc': delta_acc,
            })

            sign = '+' if delta_auc >= 0 else ''
            print(f"  {fold['test_year']}: Baseline AUC={bl_auc:.4f} -> Optimized={opt_auc:.4f} ({sign}{delta_auc:.4f})  "
                  f"Acc: {bl_acc:.4f} -> {opt_acc:.4f}")

        res_df = pd.DataFrame(results)
        avg_bl_auc = res_df['baseline_auc'].mean()
        avg_opt_auc = res_df['optimized_auc'].mean()
        avg_bl_acc = res_df['baseline_acc'].mean()
        avg_opt_acc = res_df['optimized_acc'].mean()

        print(f"\n  AVG: Baseline AUC={avg_bl_auc:.4f} -> Optimized={avg_opt_auc:.4f} ({avg_opt_auc - avg_bl_auc:+.4f})")
        print(f"  AVG: Baseline Acc={avg_bl_acc:.4f} -> Optimized={avg_opt_acc:.4f} ({avg_opt_acc - avg_bl_acc:+.4f})")

        self.results_df = res_df

    def save_results(self):
        """Save optimized hyperparameters and results."""
        print("\n[5/5] Saving results...")

        # Save hyperparameters
        output = {
            'spy_timing_lgb': self.lgb_best,
            'spy_timing_xgb': self.xgb_best,
            'lgb_cv_auc': float(self.lgb_study.best_value),
            'xgb_cv_auc': float(self.xgb_study.best_value),
            'n_trials': self.n_trials,
            'n_features': len(self.all_features),
            'timestamp': pd.Timestamp.now().isoformat(),
        }

        params_path = os.path.join(self.model_dir, 'spy_optimized_hyperparameters.json')
        with open(params_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  OK Params saved: {params_path}")

        # Save results comparison
        results_path = os.path.join(self.output_dir, 'spy_hyperparam_optimization_results.csv')
        self.results_df.to_csv(results_path, index=False)
        print(f"  OK Results saved: {results_path}")

    def run(self):
        t0 = time.time()
        print("=" * 80)
        print("SPY TIMING MODEL -- OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)

        self.prepare_data()
        self.optimize_lgb()
        self.optimize_xgb()
        self.evaluate_ensemble()
        self.save_results()

        elapsed = time.time() - t0
        print(f"\nTOTAL TIME: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print("=" * 80)
        print("\nNext: Run 06_01 again to retrain with optimized params,")
        print("      or run 06_05_spy_retrain_optimized.py for automated retrain.")
        print("=" * 80)


if __name__ == '__main__':
    # 100 trials per model, 30 min timeout each
    SPYHyperparamOptimizer(n_trials=100, timeout_per_model=1800).run()
