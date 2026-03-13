#!/usr/bin/env python3
"""
STEP 30a: TRAIN XGBOOST BASE MODEL (10-DAY HORIZON)
====================================================

Purpose:
- Train XGBoost per-symbol with walk-forward validation for 10-day predictions
- Test if longer horizon provides stronger predictive signal
- Generate out-of-sample predictions for meta-learner

Strategy:
- Walk-forward validation: Train on expanding window, predict next year
- Per-symbol training: Capture stock-specific patterns
- Feature normalization: Cross-sectional + time-series z-scores
- 10-DAY TARGET: Predict next 10 trading days (~2 weeks) direction

Target:
- Binary classification: 10-day return vs SPY (Outperform=1, Underperform=0)
- Expected: 58-60% AUC (hypothesis: longer horizon = stronger trends)

Output:
- trained_models/xgb_10day/*.pkl (per-symbol models)
- xgb_predictions_validation_10day.csv (out-of-sample for meta-learner)
- xgb_training_log_10day.csv (performance metrics)

Created: December 3, 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, log_loss
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class XGBoostTrainer10Day:
    """Train XGBoost models per-symbol with walk-forward validation (10-day horizon)"""
    
    def __init__(self):
        """Initialize paths and parameters"""
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.features_csv = os.path.join(self.base_path, r'#_feature_engineering\ml_features_master.csv')
        self.model_dir = os.path.join(self.base_path, r'#_model_training\trained_models\xgb_10day')
        self.predictions_csv = os.path.join(self.base_path, r'#_model_training\xgb_predictions_validation_10day.csv')
        self.log_csv = os.path.join(self.base_path, r'#_model_training\xgb_training_log_10day.csv')
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        print("=" * 80)
        print("STEP 30a: TRAIN XGBOOST BASE MODEL (10-DAY HORIZON)")
        print("=" * 80)
        print(f"Features: {os.path.basename(self.features_csv)}")
        print(f"Model directory: {self.model_dir}")
        print(f"Target: Market-relative 10-day (Outperform SPY=1, Underperform=0)")
        print("Strategy: Walk-forward validation (2020->2021, 2021->2022, etc.)")
        print("Hypothesis: Longer horizon = stronger trends = better AUC")
        print("Expected: 58-60% AUC (vs 53% for 5-day, 51% for 1-day)")
        print("=" * 80)
        
        # Load optimized hyperparameters if available
        optimized_params_file = os.path.join(self.base_path, r'#_model_training\optimized_hyperparameters.json')
        if os.path.exists(optimized_params_file):
            print("\n[OK] Loading OPTIMIZED hyperparameters from Optuna...")
            with open(optimized_params_file, 'r') as f:
                optimized_params = json.load(f)
                xgb_optimized = optimized_params['xgboost']['params']
            
            # XGBoost optimized hyperparameters
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': int(xgb_optimized['max_depth']),
                'learning_rate': xgb_optimized['learning_rate'],
                'n_estimators': int(xgb_optimized['n_estimators']),
                'min_child_weight': int(xgb_optimized['min_child_weight']),
                'subsample': xgb_optimized['subsample'],
                'colsample_bytree': xgb_optimized['colsample_bytree'],
                'gamma': xgb_optimized['gamma'],
                'reg_alpha': xgb_optimized['reg_alpha'],
                'reg_lambda': xgb_optimized['reg_lambda'],
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'gpu_hist',  # GPU acceleration (RTX 4060)
                'device': 'cuda'  # Use GPU
            }
            print(f"  Improvement: {optimized_params['xgboost']['improvement']}")
            print(f"  Test Accuracy: {optimized_params['xgboost']['test_accuracy']:.4f}")
            print(f"  Test AUC: {optimized_params['xgboost']['test_auc']:.4f}")
        else:
            print("\n[WARNING] Using DEFAULT hyperparameters (optimized file not found)")
            # XGBoost default hyperparameters
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'gpu_hist',  # GPU acceleration (RTX 4060)
                'device': 'cuda'  # Use GPU
            }
        
        print("=" * 80)
        
        # Walk-forward with 10-day embargo gap (target horizon = 10 trading days)
        # Train ends 10+ trading days before test starts to prevent target leakage
        self.validation_splits = [
            {'train_start': '2015-01-01', 'train_end': '2020-12-14', 'test_start': '2021-01-04', 'test_end': '2021-12-31'},
            {'train_start': '2015-01-01', 'train_end': '2021-12-14', 'test_start': '2022-01-04', 'test_end': '2022-12-31'},
            {'train_start': '2015-01-01', 'train_end': '2022-12-14', 'test_start': '2023-01-04', 'test_end': '2023-12-31'},
            {'train_start': '2015-01-01', 'train_end': '2023-12-14', 'test_start': '2024-01-04', 'test_end': '2024-12-31'},
            {'train_start': '2015-01-01', 'train_end': '2024-12-14', 'test_start': '2025-01-06', 'test_end': '2025-12-31'}
        ]
        
    def load_features(self):
        """Load feature dataset"""
        print("\n[1/6] Loading features...")
        
        # Load full dataset
        df = pd.read_csv(self.features_csv, parse_dates=['date'])
        
        print(f"  OK Loaded: {len(df):,} rows x {len(df.columns):,} columns")
        print(f"  OK Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  OK Symbols: {df['symbol'].nunique():,}")
        
        return df
    
    def create_target_variable(self, df):
        """Create market-relative 10-day target (binary: Outperform market=1, Underperform=0)"""
        print("\n[2/6] Creating market-relative 10-day target variable...")
        
        # Calculate 10-day forward return per symbol
        df = df.sort_values(['symbol', 'date'])
        df['next_10day_return'] = df.groupby('symbol')['Close'].pct_change(10).shift(-10)
        
        # Remove rows with no forward return before computing benchmark
        df = df[df['next_10day_return'].notna()].copy()
        
        # Cross-sectional market benchmark: mean 10-day return across all stocks per date
        # This is available for ALL dates (no dependency on SPY data coverage)
        market_benchmark = df.groupby('date')['next_10day_return'].mean().rename('market_10day_return')
        df = df.merge(market_benchmark, on='date', how='left')
        
        # Market-relative target: 1 if stock outperforms the market average
        df['target'] = (df['next_10day_return'] > df['market_10day_return']).astype(int)
        
        # Remove any remaining NaN targets
        df = df[df['target'].notna()].copy()
        
        print(f"  OK Benchmark: cross-sectional mean 10-day return per date")
        print(f"  OK Outperforms market: {df['target'].sum():,}, Underperforms: {(~df['target'].astype(bool)).sum():,}")
        print(f"  OK Class balance: {df['target'].mean():.1%} Outperform, {(1 - df['target'].mean()):.1%} Underperform")
        print(f"  OK Rows after cleaning: {len(df):,}")
        
        return df
    
    def select_features(self, df):
        """Select feature columns with basic filtering"""
        print("\n[3/6] Selecting features...")
        
        # Exclude non-feature columns (including forward-looking benchmark)
        exclude_cols = ['date', 'symbol', 'next_10day_return', 'market_10day_return', 'target', 
                        'Open', 'High', 'Low', 'Close', 'Volume']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        initial_count = len(feature_cols)
        
        # Remove near-constant features (variance < 0.001)
        variances = df[feature_cols].fillna(0).var()
        low_var = variances[variances < 0.001].index.tolist()
        if low_var:
            feature_cols = [c for c in feature_cols if c not in low_var]
            print(f"  OK Removed {len(low_var)} near-constant features")
        
        print(f"  OK Selected {len(feature_cols):,} feature columns (from {initial_count})")
        
        return feature_cols
    
    def normalize_features(self, X_train, X_test):
        """Normalize features using z-score standardization"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def train_per_symbol(self, df, feature_cols):
        """Train XGBoost per symbol with walk-forward validation"""
        print("\n[4/6] Training per-symbol models (walk-forward validation)...")
        
        all_predictions = []
        training_log = []
        
        symbols = sorted(df['symbol'].unique())
        
        print(f"  Training {len(symbols):,} symbols across {len(self.validation_splits)} folds...")
        print(f"  (This may take 30-60 minutes)")
        
        for fold_idx, split in enumerate(self.validation_splits, 1):
            print(f"\n  Fold {fold_idx}/{len(self.validation_splits)}: Train {split['train_start'][:4]}->{split['train_end'][:4]}, Test {split['test_start'][:4]}")
            
            fold_predictions = []
            fold_metrics = []
            
            for sym_idx, symbol in enumerate(symbols, 1):
                try:
                    # Filter data for this symbol
                    symbol_data = df[df['symbol'] == symbol].copy()
                    
                    # Split train/test by date
                    train_data = symbol_data[
                        (symbol_data['date'] >= split['train_start']) & 
                        (symbol_data['date'] <= split['train_end'])
                    ]
                    test_data = symbol_data[
                        (symbol_data['date'] >= split['test_start']) & 
                        (symbol_data['date'] <= split['test_end'])
                    ]
                    
                    # Skip if insufficient data
                    if len(train_data) < 100 or len(test_data) < 10:
                        continue
                    
                    # Subsample training: keep every 5th row to reduce 10-day target overlap
                    # (consecutive 10-day targets share 9/10 days → ~90% correlated)
                    train_data_sub = train_data.iloc[::5]
                    if len(train_data_sub) < 50:
                        continue
                    
                    # Prepare features and target (tree models handle NaN natively)
                    X_train = train_data_sub[feature_cols]
                    y_train = train_data_sub['target']
                    X_test = test_data[feature_cols]
                    y_test = test_data['target']
                    
                    # Train XGBoost (no scaler — trees are scale-invariant)
                    model = xgb.XGBClassifier(**self.xgb_params)
                    model.fit(X_train, y_train, verbose=False)
                    
                    # Predict probabilities
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Calculate metrics
                    auc = roc_auc_score(y_test, y_pred_proba)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    logloss = log_loss(y_test, y_pred_proba)
                    
                    # Save predictions
                    fold_predictions.append(pd.DataFrame({
                        'date': test_data['date'],
                        'symbol': symbol,
                        'actual': y_test,
                        'xgb_pred_proba': y_pred_proba,
                        'xgb_pred': y_pred,
                        'fold': fold_idx
                    }))
                    
                    # Log metrics
                    fold_metrics.append({
                        'fold': fold_idx,
                        'symbol': symbol,
                        'train_size': len(train_data),
                        'test_size': len(test_data),
                        'auc': auc,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'logloss': logloss
                    })
                    
                    # Save model (only for final fold - used for production)
                    if fold_idx == len(self.validation_splits):
                        model_path = os.path.join(self.model_dir, f'{symbol}.pkl')
                        with open(model_path, 'wb') as f:
                            pickle.dump({'model': model, 'feature_cols': feature_cols}, f)
                    
                    # Progress update every 50 symbols
                    if sym_idx % 50 == 0:
                        avg_auc = np.mean([m['auc'] for m in fold_metrics])
                        print(f"    Processed {sym_idx}/{len(symbols)} symbols | Avg AUC: {avg_auc:.4f}")
                
                except Exception as e:
                    print(f"    WARNING: Error training {symbol}: {str(e)[:50]}")
                    continue
            
            # Combine fold predictions
            if fold_predictions:
                all_predictions.append(pd.concat(fold_predictions, ignore_index=True))
                training_log.extend(fold_metrics)
                
                # Fold summary
                fold_avg_auc = np.mean([m['auc'] for m in fold_metrics])
                fold_avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
                print(f"  OK Fold {fold_idx} complete: {len(fold_metrics)} symbols | Avg AUC: {fold_avg_auc:.4f} | Avg Acc: {fold_avg_acc:.4f}")
        
        # Combine all predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        training_log_df = pd.DataFrame(training_log)
        
        return predictions_df, training_log_df
    
    def save_outputs(self, predictions_df, training_log_df):
        """Save predictions and training log"""
        print("\n[5/6] Saving outputs...")
        
        # Save predictions (for meta-learner)
        predictions_df.to_csv(self.predictions_csv, index=False)
        print(f"  OK Predictions saved: {os.path.basename(self.predictions_csv)} ({len(predictions_df):,} rows)")
        
        # Save training log
        training_log_df.to_csv(self.log_csv, index=False)
        print(f"  OK Training log saved: {os.path.basename(self.log_csv)} ({len(training_log_df):,} rows)")
        
        # Count saved models
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        print(f"  OK Models saved: {len(model_files):,} symbols in {os.path.basename(self.model_dir)}/")
    
    def print_summary(self, training_log_df):
        """Print performance summary"""
        print("\n[6/6] Performance Summary")
        print("=" * 80)
        
        # Overall metrics
        overall_auc = training_log_df['auc'].mean()
        overall_acc = training_log_df['accuracy'].mean()
        
        print(f"  Overall AUC: {overall_auc:.4f} (±{training_log_df['auc'].std():.4f})")
        print(f"  Overall Accuracy: {overall_acc:.4f} (±{training_log_df['accuracy'].std():.4f})")
        print(f"  Overall Precision: {training_log_df['precision'].mean():.4f} (±{training_log_df['precision'].std():.4f})")
        print(f"  Overall Recall: {training_log_df['recall'].mean():.4f} (±{training_log_df['recall'].std():.4f})")
        
        # Multi-horizon comparison
        print(f"\n  Multi-Horizon Performance Comparison:")
        print(f"    1-Day XGBoost:  51% AUC (baseline)")
        print(f"    5-Day XGBoost:  53% AUC (+2% improvement)")
        print(f"    10-Day XGBoost: {overall_auc:.1%} AUC ({(overall_auc - 0.51) / 0.51 * 100:+.1f}% vs 1-day)")
        
        if overall_auc >= 0.60:
            print(f"    [EXCELLENT] EXCELLENT SIGNAL - 10-day predictions are STRONG!")
            print(f"    -> Recommendation: Build full 10-day ensemble immediately")
        elif overall_auc >= 0.58:
            print(f"    [STRONG] STRONG SIGNAL - 10-day predictions show clear improvement")
            print(f"    -> Recommendation: Build full 10-day ensemble")
        elif overall_auc >= 0.55:
            print(f"    [GOOD] GOOD SIGNAL - 10-day predictions worth pursuing")
            print(f"    -> Recommendation: Test LightGBM/CatBoost on 10-day")
        elif overall_auc >= 0.53:
            print(f"    [MARGINAL] MARGINAL - 10-day only slightly better than 5-day")
            print(f"    -> Recommendation: Focus on feature engineering instead")
        else:
            print(f"    [WEAK] NO IMPROVEMENT - Longer horizon doesn't help")
            print(f"    -> Recommendation: Problem is features, not time horizon")
        
        # Per-fold metrics
        print("\n  Per-Fold Performance:")
        for fold in sorted(training_log_df['fold'].unique()):
            fold_data = training_log_df[training_log_df['fold'] == fold]
            print(f"    Fold {fold}: AUC={fold_data['auc'].mean():.4f}, Acc={fold_data['accuracy'].mean():.4f}, N={len(fold_data)}")
        
        # Top/bottom performers
        print("\n  Top 10 Performers (by AUC):")
        top10 = training_log_df.groupby('symbol')['auc'].mean().sort_values(ascending=False).head(10)
        for symbol, auc in top10.items():
            print(f"    {symbol}: {auc:.4f}")
        
        print("\n  Bottom 10 Performers (by AUC):")
        bottom10 = training_log_df.groupby('symbol')['auc'].mean().sort_values().head(10)
        for symbol, auc in bottom10.items():
            print(f"    {symbol}: {auc:.4f}")
        
        print("\n" + "=" * 80)
        print("OK XGBoost 10-day training complete!")
        print("=" * 80)
    
    def run(self):
        """Execute full training pipeline"""
        start_time = datetime.now()
        
        # Load and prepare data
        df = self.load_features()
        df = self.create_target_variable(df)
        feature_cols = self.select_features(df)
        
        # Train models
        predictions_df, training_log_df = self.train_per_symbol(df, feature_cols)
        
        # Save outputs
        self.save_outputs(predictions_df, training_log_df)
        
        # Print summary
        self.print_summary(training_log_df)
        
        # Execution time
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nTIME: Total execution time: {duration:.1f} seconds ({duration/60:.1f} minutes)")


if __name__ == '__main__':
    trainer = XGBoostTrainer10Day()
    trainer.run()
