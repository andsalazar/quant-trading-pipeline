#!/usr/bin/env python3
"""
STEP 30d: TRAIN META-LEARNER ENSEMBLE (10-DAY HORIZON)
========================================================

Purpose:
- Blend XGBoost, LightGBM, and CatBoost 10-day predictions
- Train on out-of-sample validation predictions (no look-ahead bias)
- Learn optimal weighting for 10-day forecasts

Strategy:
- Load all 3 base model 10-day out-of-sample predictions
- Evaluate on holdout fold 5, then train production model on all folds
- Uses only 3 base model probabilities (no leaky adaptive features)
- Meta-learner: shallow LightGBM for nonlinear blending

Target:
- Binary classification: Outperform (1) vs Underperform (0)
- Expected Honest OOS AUC: ~0.57 (vs ~0.55-0.56 best base model)

Input:
- xgb_predictions_validation_10day.csv
- lightgbm_predictions_validation_10day.csv
- catboost_predictions_validation_10day.csv

Output:
- meta_learner_10day.pkl (trained meta-model + feature_cols)
- ensemble_predictions_validation_10day.csv (final blended predictions)
- ensemble_training_log_10day.csv (performance comparison)

Created: December 4, 2025
Updated: Fixed data leakage — removed adaptive features that used actuals as inputs
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, log_loss
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MetaLearnerTrainer10Day:
    """Train meta-learner to ensemble 10-day base model predictions"""
    
    def __init__(self):
        """Initialize paths and parameters"""
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        
        # Input: Base model 10-day predictions
        self.xgb_predictions_csv = os.path.join(self.base_path, 'xgb_predictions_validation_10day.csv')
        self.lightgbm_predictions_csv = os.path.join(self.base_path, 'lightgbm_predictions_validation_10day.csv')
        self.catboost_predictions_csv = os.path.join(self.base_path, 'catboost_predictions_validation_10day.csv')
        
        # Output: Meta-learner and ensemble predictions
        self.meta_learner_path = os.path.join(self.base_path, 'meta_learner_10day.pkl')
        self.ensemble_predictions_csv = os.path.join(self.base_path, 'ensemble_predictions_validation_10day.csv')
        self.ensemble_log_csv = os.path.join(self.base_path, 'ensemble_training_log_10day.csv')
        
        print("=" * 80)
        print("STEP 30d: TRAIN META-LEARNER ENSEMBLE (10-DAY HORIZON)")
        print("=" * 80)
        print(f"Meta-Learner: LightGBM Gradient Boosted Ensemble")
        print(f"Features: 3 base model probabilities (clean, no target leakage)")
        print(f"Evaluation: Holdout fold 5, then retrain on all folds for production")
        print(f"Output: {os.path.basename(self.ensemble_predictions_csv)}")
        print("=" * 80)
        
        # Meta-learner parameters (shallow LightGBM for nonlinear blending)
        self.meta_params = {
            'objective': 'binary',
            'max_depth': 2,
            'n_estimators': 50,
            'learning_rate': 0.05,
            'num_leaves': 4,
            'min_child_samples': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
    
    def load_base_predictions(self):
        """Load all base model predictions"""
        print("\n[1/5] Loading base model predictions...")
        
        # Load predictions
        xgb_pred = pd.read_csv(self.xgb_predictions_csv, parse_dates=['date'])
        lightgbm_pred = pd.read_csv(self.lightgbm_predictions_csv, parse_dates=['date'])
        catboost_pred = pd.read_csv(self.catboost_predictions_csv, parse_dates=['date'])
        
        print(f"  OK XGBoost: {len(xgb_pred):,} predictions")
        print(f"  OK LightGBM: {len(lightgbm_pred):,} predictions")
        print(f"  OK CatBoost: {len(catboost_pred):,} predictions")
        
        return xgb_pred, lightgbm_pred, catboost_pred
    
    def merge_predictions(self, xgb_pred, lightgbm_pred, catboost_pred):
        """Merge all predictions on date + symbol"""
        print("\n[2/5] Merging predictions...")
        
        # Merge on date + symbol
        merged = xgb_pred[['date', 'symbol', 'actual', 'xgb_pred_proba', 'fold']].merge(
            lightgbm_pred[['date', 'symbol', 'lightgbm_pred_proba']],
            on=['date', 'symbol'],
            how='inner'
        ).merge(
            catboost_pred[['date', 'symbol', 'catboost_pred_proba']],
            on=['date', 'symbol'],
            how='inner'
        )
        
        print(f"  OK Merged: {len(merged):,} predictions (inner join on date + symbol)")
        print(f"  OK Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
        print(f"  OK Symbols: {merged['symbol'].nunique():,}")
        print(f"  OK Folds: {sorted(merged['fold'].unique())}")
        
        return merged
    
    def add_uncertainty_features(self, merged_df):
        """Add uncertainty features derived purely from model predictions (no actuals used).
        
        These are kept for reporting/confidence scoring but NOT used as model inputs
        since they add zero predictive value beyond the 3 raw probabilities.
        """
        print("\n[3/6] Computing uncertainty metrics (for reporting only)...")
        
        predictions = merged_df[['xgb_pred_proba', 'lightgbm_pred_proba', 'catboost_pred_proba']].values
        
        # Prediction variance (model disagreement)
        merged_df['prediction_variance'] = np.var(predictions, axis=1)
        
        # Model disagreement percentage
        max_proba = np.max(predictions, axis=1)
        min_proba = np.min(predictions, axis=1)
        mean_proba = np.mean(predictions, axis=1)
        merged_df['model_disagreement_pct'] = (max_proba - min_proba) / (mean_proba + 1e-8)
        
        # Prediction entropy
        eps = 1e-8
        merged_df['prediction_entropy'] = -mean_proba * np.log(mean_proba + eps) - (1 - mean_proba) * np.log(1 - mean_proba + eps)
        
        # Max-min spread
        merged_df['max_min_spread'] = max_proba - min_proba
        
        print(f"  OK Added 4 uncertainty metrics (not used as model inputs)")
        
        return merged_df
    
    def train_meta_learner(self, merged_df):
        """Train meta-learner with proper holdout evaluation.
        
        Strategy:
        1. Evaluate on holdout (train folds 1-4, test fold 5) to get honest metrics
        2. Train production model on ALL folds for deployment
        
        Features: Only the 3 base model probabilities (no leaky adaptive features)
        """
        print("\n[4/6] Training meta-learner...")
        
        # Clean features only — no actuals used as inputs
        feature_cols = [
            'xgb_pred_proba', 'lightgbm_pred_proba', 'catboost_pred_proba'
        ]
        
        # ---- Step 1: Holdout evaluation (honest metrics) ----
        print("\n  --- Holdout Evaluation (train folds 1-4, test fold 5) ---")
        train_df = merged_df[merged_df['fold'] <= 4]
        test_df = merged_df[merged_df['fold'] == 5]
        
        eval_model = lgb.LGBMClassifier(**self.meta_params)
        eval_model.fit(train_df[feature_cols], train_df['actual'])
        
        test_proba = eval_model.predict_proba(test_df[feature_cols])[:, 1]
        holdout_auc = roc_auc_score(test_df['actual'], test_proba)
        holdout_acc = accuracy_score(test_df['actual'], (test_proba > 0.5).astype(int))
        
        print(f"  OK Holdout AUC: {holdout_auc:.4f}")
        print(f"  OK Holdout Acc: {holdout_acc:.4f}")
        print(f"  OK Train size: {len(train_df):,}, Test size: {len(test_df):,}")
        
        # Per-fold holdout
        print("\n  --- Per-Fold Forward Validation ---")
        for test_fold in [2, 3, 4, 5]:
            tr = merged_df[merged_df['fold'] < test_fold]
            te = merged_df[merged_df['fold'] == test_fold]
            m = lgb.LGBMClassifier(**self.meta_params)
            m.fit(tr[feature_cols], tr['actual'])
            p = m.predict_proba(te[feature_cols])[:, 1]
            a = roc_auc_score(te['actual'], p)
            ac = accuracy_score(te['actual'], (p > 0.5).astype(int))
            print(f"    Fold {test_fold} (train 1-{test_fold-1}): AUC={a:.4f}, Acc={ac:.4f}, N={len(te):,}")
        
        # ---- Step 2: Train production model on ALL folds ----
        print("\n  --- Training Production Model (all folds) ---")
        X = merged_df[feature_cols]
        y = merged_df['actual']
        
        meta_model = lgb.LGBMClassifier(**self.meta_params)
        meta_model.fit(X, y)
        
        # Get feature importances
        importances = meta_model.feature_importances_
        print(f"  OK Production model trained on {len(merged_df):,} predictions")
        print(f"  OK Features: {feature_cols}")
        print(f"  OK Feature importances:")
        for feat, imp in zip(feature_cols, importances):
            print(f"      {feat}: {imp:.0f}")
        
        # Predict ensemble probabilities (for CSV output — these are in-sample for historical data)
        ensemble_proba = meta_model.predict_proba(X)[:, 1]
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        merged_df['ensemble_pred_proba'] = ensemble_proba
        merged_df['ensemble_pred'] = ensemble_pred
        
        # Confidence score (distance from 0.5)
        merged_df['confidence'] = np.abs(ensemble_proba - 0.5) * 2
        
        # Store feature_cols and holdout metrics in the model dict for downstream use
        self._meta_model = meta_model
        self._feature_cols = feature_cols
        self._holdout_auc = holdout_auc
        self._holdout_acc = holdout_acc
        
        return merged_df
    
    def evaluate_performance(self, merged_df):
        """Evaluate performance: base models vs ensemble (in-sample for comparison)"""
        print("\n[5/6] Performance comparison (in-sample for base model comparison)...")
        
        y_true = merged_df['actual']
        
        results = []
        
        for model_name, pred_col in [
            ('XGBoost', 'xgb_pred_proba'),
            ('LightGBM', 'lightgbm_pred_proba'),
            ('CatBoost', 'catboost_pred_proba'),
            ('Ensemble', 'ensemble_pred_proba')
        ]:
            y_pred_proba = merged_df[pred_col]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc = roc_auc_score(y_true, y_pred_proba)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            logloss = log_loss(y_true, y_pred_proba)
            
            results.append({
                'model': model_name,
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'logloss': logloss
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n  In-Sample Performance (all folds):")
        print(results_df.to_string(index=False))
        
        # Holdout metrics (the honest numbers)
        print(f"\n  *** HONEST OUT-OF-SAMPLE (Holdout Fold 5) ***")
        print(f"  *** Ensemble AUC: {self._holdout_auc:.4f} ***")
        print(f"  *** Ensemble Acc: {self._holdout_acc:.4f} ***")
        
        # Per-fold base model performance
        print("\n  Base Model Performance (Fold 5 only):")
        f5 = merged_df[merged_df['fold'] == 5]
        for name, col in [('XGBoost', 'xgb_pred_proba'), ('LightGBM', 'lightgbm_pred_proba'), ('CatBoost', 'catboost_pred_proba')]:
            auc = roc_auc_score(f5['actual'], f5[col])
            acc = accuracy_score(f5['actual'], (f5[col] > 0.5).astype(int))
            print(f"    {name}: AUC={auc:.4f}, Acc={acc:.4f}")
        
        # Simple average baseline
        avg_proba = f5[['xgb_pred_proba', 'lightgbm_pred_proba', 'catboost_pred_proba']].mean(axis=1)
        avg_auc = roc_auc_score(f5['actual'], avg_proba)
        avg_acc = accuracy_score(f5['actual'], (avg_proba > 0.5).astype(int))
        print(f"    Simple Avg: AUC={avg_auc:.4f}, Acc={avg_acc:.4f}")
        
        # High-confidence performance
        print("\n  High-Confidence Performance (top 20%):")
        high_conf_threshold = merged_df['confidence'].quantile(0.8)
        high_conf = merged_df[merged_df['confidence'] >= high_conf_threshold]
        high_conf_auc = roc_auc_score(high_conf['actual'], high_conf['ensemble_pred_proba'])
        high_conf_acc = accuracy_score(high_conf['actual'], high_conf['ensemble_pred'])
        print(f"    Confidence threshold: {high_conf_threshold:.3f}")
        print(f"    AUC: {high_conf_auc:.4f}")
        print(f"    Accuracy: {high_conf_acc:.4f}")
        print(f"    Predictions: {len(high_conf):,} ({len(high_conf)/len(merged_df):.1%} of total)")
        
        return results_df
    
    def save_outputs(self, merged_df, results_df):
        """Save meta-learner (with feature_cols), predictions, and log"""
        print("\n[6/6] Saving outputs...")
        
        # Save meta-learner with feature_cols for downstream use
        meta_dict = {
            'model': self._meta_model,
            'feature_cols': self._feature_cols,
            'holdout_auc': self._holdout_auc,
            'holdout_acc': self._holdout_acc,
            'trained_date': datetime.now().isoformat()
        }
        with open(self.meta_learner_path, 'wb') as f:
            pickle.dump(meta_dict, f)
        print(f"  OK Meta-learner saved: {os.path.basename(self.meta_learner_path)}")
        print(f"     Features: {self._feature_cols}")
        print(f"     Holdout AUC: {self._holdout_auc:.4f}")
        
        # Save ensemble predictions
        merged_df.to_csv(self.ensemble_predictions_csv, index=False)
        print(f"  OK Predictions saved: {os.path.basename(self.ensemble_predictions_csv)} ({len(merged_df):,} rows)")
        
        # Save performance log
        results_df.to_csv(self.ensemble_log_csv, index=False)
        print(f"  OK Performance log saved: {os.path.basename(self.ensemble_log_csv)}")
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("OK Meta-learner training complete!")
        print("=" * 80)
        print(f"\nHonest Out-of-Sample AUC: {self._holdout_auc:.4f}")
        print(f"Honest Out-of-Sample Acc: {self._holdout_acc:.4f}")
        print("\nFeatures Used (clean — no target leakage):")
        for f in self._feature_cols:
            print(f"  - {f}")
        print("\nNext Steps:")
        print("  1. Run 03_05_generate_predictions_10day.py for production predictions")
        print("  2. Backtest strategy using ensemble predictions")
        print("=" * 80)
    
    def run(self):
        """Execute full meta-learning pipeline"""
        start_time = datetime.now()
        
        # Load base predictions
        xgb_pred, lightgbm_pred, catboost_pred = self.load_base_predictions()
        
        # Merge predictions
        merged_df = self.merge_predictions(xgb_pred, lightgbm_pred, catboost_pred)
        
        # Add uncertainty features (for reporting only, not model inputs)
        merged_df = self.add_uncertainty_features(merged_df)
        
        # Train meta-learner with proper holdout evaluation
        merged_df = self.train_meta_learner(merged_df)
        
        # Evaluate performance
        results_df = self.evaluate_performance(merged_df)
        
        # Save outputs
        self.save_outputs(merged_df, results_df)
        
        # Print summary
        self.print_summary()
        
        # Execution time
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nTIME: Total execution time: {duration:.1f} seconds")


if __name__ == '__main__':
    trainer = MetaLearnerTrainer10Day()
    trainer.run()
