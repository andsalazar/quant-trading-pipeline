#!/usr/bin/env python3
"""
Hyperparameter Optimization for Trading Models
================================================

Uses Optuna for Bayesian optimization of:
1. XGBoost (currently weakest at 49.96%)
2. LightGBM (currently 52.54%)
3. CatBoost (currently 52.85%)

Expected improvements:
- XGBoost: 49.96% → 52-53% (+2-3%)
- Ensemble: 53.21% → 54%+ (+1%)
- High-confidence accuracy: 56.75% → 58%+ (+1-2%)

Runtime: ~2-3 hours for 100 trials per model
"""

import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# GPU detection
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    HAS_GPU = result.returncode == 0
    print(f"✓ GPU detected: {HAS_GPU}")
except:
    HAS_GPU = False
    print("⚠ No GPU detected, using CPU")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_file': CONFIG_BASE_PATH  # Set in config.py,
    'output_dir': CONFIG_BASE_PATH  # Set in config.py,
    'n_trials': 100,  # Number of optimization trials per model
    'cv_splits': 3,   # Time series cross-validation splits
    'test_size': 0.2, # 20% test set
    'random_state': 42,
    'timeout': 7200,  # 2 hours per model
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare training data"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    print(f"Reading: {CONFIG['data_file']}")
    df = pd.read_csv(CONFIG['data_file'])
    print(f"✓ Loaded {len(df):,} rows")
    
    # Prepare features
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)  # Sort by symbol first, then date
    
    # Create target variable (same as training scripts)
    print("\n✓ Creating target variable...")
    df['next_day_return'] = df.groupby('symbol')['Close'].pct_change(1).shift(-1)  # Use pct_change(1)
    df['target'] = (df['next_day_return'] > 0).astype(int)
    
    # Remove rows with missing targets (last day per symbol)
    df = df[df['target'].notna()].copy()
    
    print(f"  Target created: {df['target'].sum():,} Up days, {(~df['target'].astype(bool)).sum():,} Down days")
    print(f"  Class balance: {df['target'].mean():.1%} Up, {(1 - df['target'].mean()):.1%} Down")
    print(f"  Rows after removing last day: {len(df):,}")
    
    # Define feature columns (exclude metadata, target, and OHLCV)
    exclude_cols = ['date', 'symbol', 'next_day_return', 'target',
                    'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Split data
    train_size = int(len(df) * (1 - CONFIG['test_size']))
    
    X_train = df.iloc[:train_size][feature_cols]
    y_train = df.iloc[:train_size]['target']
    
    X_test = df.iloc[train_size:][feature_cols]
    y_test = df.iloc[train_size:]['target']
    
    print(f"\n✓ Train: {len(X_train):,} samples ({y_train.mean():.2%} positive)")
    print(f"✓ Test: {len(X_test):,} samples ({y_test.mean():.2%} positive)")
    
    return X_train, X_test, y_train, y_test, feature_cols

# ============================================================================
# XGBOOST OPTIMIZATION
# ============================================================================

def optimize_xgboost(X_train, y_train, X_test, y_test):
    """Optimize XGBoost hyperparameters"""
    
    print("\n" + "="*80)
    print("OPTIMIZING XGBOOST")
    print("="*80)
    print(f"Current performance: 49.96% accuracy (weakest model)")
    print(f"Target: 52-53% accuracy")
    print(f"Trials: {CONFIG['n_trials']}")
    
    def objective(trial):
        # Hyperparameter search space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'device': 'cuda' if HAS_GPU else 'cpu',
            
            # Optimizable parameters
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=CONFIG['cv_splits'])
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train = X_train.iloc[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            X_cv_val = X_train.iloc[val_idx]
            y_cv_val = y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params, random_state=CONFIG['random_state'])
            model.fit(X_cv_train, y_cv_train, verbose=False)
            
            y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
            auc = roc_auc_score(y_cv_val, y_pred_proba)
            cv_scores.append(auc)
        
        return np.mean(cv_scores)
    
    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=CONFIG['random_state'])
    )
    
    study.optimize(
        objective,
        n_trials=CONFIG['n_trials'],
        timeout=CONFIG['timeout'],
        show_progress_bar=True
    )
    
    # Results
    print(f"\n{'='*80}")
    print("XGBOOST OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"✓ Best CV AUC: {study.best_value:.4f}")
    print(f"✓ Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Test on holdout
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda' if HAS_GPU else 'cpu',
        'random_state': CONFIG['random_state']
    })
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✓ Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    print(f"✓ Test AUC: {test_auc:.4f}")
    
    return study.best_params, test_acc, test_auc

# ============================================================================
# LIGHTGBM OPTIMIZATION
# ============================================================================

def optimize_lightgbm(X_train, y_train, X_test, y_test):
    """Optimize LightGBM hyperparameters"""
    
    print("\n" + "="*80)
    print("OPTIMIZING LIGHTGBM")
    print("="*80)
    print(f"Current performance: 52.54% accuracy")
    print(f"Target: 53-54% accuracy")
    print(f"Trials: {CONFIG['n_trials']}")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'device': 'gpu' if HAS_GPU else 'cpu',
            'verbose': -1,
            
            # Optimizable parameters
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
        }
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=CONFIG['cv_splits'])
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train = X_train.iloc[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            X_cv_val = X_train.iloc[val_idx]
            y_cv_val = y_train.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params, random_state=CONFIG['random_state'])
            model.fit(X_cv_train, y_cv_train)
            
            y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
            auc = roc_auc_score(y_cv_val, y_pred_proba)
            cv_scores.append(auc)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=CONFIG['random_state'])
    )
    
    study.optimize(
        objective,
        n_trials=CONFIG['n_trials'],
        timeout=CONFIG['timeout'],
        show_progress_bar=True
    )
    
    # Results
    print(f"\n{'='*80}")
    print("LIGHTGBM OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"✓ Best CV AUC: {study.best_value:.4f}")
    print(f"✓ Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Test on holdout
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu' if HAS_GPU else 'cpu',
        'verbose': -1,
        'random_state': CONFIG['random_state']
    })
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✓ Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    print(f"✓ Test AUC: {test_auc:.4f}")
    
    return study.best_params, test_acc, test_auc

# ============================================================================
# CATBOOST OPTIMIZATION
# ============================================================================

def optimize_catboost(X_train, y_train, X_test, y_test):
    """Optimize CatBoost hyperparameters"""
    
    print("\n" + "="*80)
    print("OPTIMIZING CATBOOST")
    print("="*80)
    print(f"Current performance: 52.85% accuracy (best base model)")
    print(f"Target: 53-54% accuracy")
    print(f"Trials: {CONFIG['n_trials']}")
    
    def objective(trial):
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'task_type': 'GPU' if HAS_GPU else 'CPU',
            'verbose': False,
            
            # Optimizable parameters
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
        }
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=CONFIG['cv_splits'])
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train = X_train.iloc[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            X_cv_val = X_train.iloc[val_idx]
            y_cv_val = y_train.iloc[val_idx]
            
            model = cb.CatBoostClassifier(**params, random_state=CONFIG['random_state'])
            model.fit(X_cv_train, y_cv_train)
            
            y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
            auc = roc_auc_score(y_cv_val, y_pred_proba)
            cv_scores.append(auc)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=CONFIG['random_state'])
    )
    
    study.optimize(
        objective,
        n_trials=CONFIG['n_trials'],
        timeout=CONFIG['timeout'],
        show_progress_bar=True
    )
    
    # Results
    print(f"\n{'='*80}")
    print("CATBOOST OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"✓ Best CV AUC: {study.best_value:.4f}")
    print(f"✓ Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Test on holdout
    best_params = study.best_params.copy()
    best_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'GPU' if HAS_GPU else 'CPU',
        'verbose': False,
        'random_state': CONFIG['random_state']
    })
    
    model = cb.CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✓ Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    print(f"✓ Test AUC: {test_auc:.4f}")
    
    return study.best_params, test_acc, test_auc

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run hyperparameter optimization"""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {HAS_GPU}")
    print(f"Trials per model: {CONFIG['n_trials']}")
    print(f"Timeout per model: {CONFIG['timeout']/3600:.1f} hours")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    
    # Optimize each model
    results = {}
    
    # XGBoost (weakest - highest priority)
    xgb_params, xgb_acc, xgb_auc = optimize_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = {
        'params': xgb_params,
        'test_accuracy': float(xgb_acc),
        'test_auc': float(xgb_auc),
        'improvement': f"+{(xgb_acc - 0.4996)*100:.2f}%"
    }
    
    # LightGBM
    lgb_params, lgb_acc, lgb_auc = optimize_lightgbm(X_train, y_train, X_test, y_test)
    results['lightgbm'] = {
        'params': lgb_params,
        'test_accuracy': float(lgb_acc),
        'test_auc': float(lgb_auc),
        'improvement': f"+{(lgb_acc - 0.5254)*100:.2f}%"
    }
    
    # CatBoost
    cb_params, cb_acc, cb_auc = optimize_catboost(X_train, y_train, X_test, y_test)
    results['catboost'] = {
        'params': cb_params,
        'test_accuracy': float(cb_acc),
        'test_auc': float(cb_auc),
        'improvement': f"+{(cb_acc - 0.5285)*100:.2f}%"
    }
    
    # Save results
    output_file = f"{CONFIG['output_dir']}/optimized_hyperparameters.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Saved: {output_file}")
    
    print(f"\nSummary:")
    print(f"  XGBoost: {results['xgboost']['test_accuracy']:.2%} ({results['xgboost']['improvement']})")
    print(f"  LightGBM: {results['lightgbm']['test_accuracy']:.2%} ({results['lightgbm']['improvement']})")
    print(f"  CatBoost: {results['catboost']['test_accuracy']:.2%} ({results['catboost']['improvement']})")
    
    avg_improvement = (
        (results['xgboost']['test_accuracy'] - 0.4996) +
        (results['lightgbm']['test_accuracy'] - 0.5254) +
        (results['catboost']['test_accuracy'] - 0.5285)
    ) / 3
    
    print(f"\n✓ Average improvement: +{avg_improvement*100:.2f}%")
    print(f"✓ Expected ensemble boost: +{avg_improvement*100*0.5:.2f}% (conservative)")
    print(f"✓ Predicted high-conf accuracy: {0.5675 + avg_improvement*0.5:.2%}")
    
    print(f"\nNext steps:")
    print(f"1. Review optimized_hyperparameters.json")
    print(f"2. Run: python 00_run_full_training.py (uses optimized params)")
    print(f"3. Compare new vs old performance")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
