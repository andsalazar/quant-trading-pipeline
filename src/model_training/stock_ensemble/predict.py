#!/usr/bin/env python3
"""
STEP 30e: GENERATE 10-DAY PREDICTIONS (PRODUCTION)
====================================================

Purpose:
- Load trained 10-day models (XGBoost, LightGBM, CatBoost, Meta-Learner)
- Generate predictions for next 10 trading days (~2 weeks)
- Production-ready for daily workflow

Strategy:
- Load latest ml_features_master.csv
- Load trained 10-day models from disk
- Predict per-symbol using all 3 base models
- Blend predictions using 10-day meta-learner
- Output predictions with confidence scores

Target:
- Binary classification: 10-day return vs market average (Outperform=1, Underperform=0)
- Output: Probability, prediction, confidence score
- Prediction horizon: 10 trading days ahead (~2 weeks)

Input:
- ml_features_master.csv (latest features)
- trained_models/xgb_10day/*.pkl
- trained_models/lightgbm_10day/*.pkl
- trained_models/catboost_10day/*.pkl
- meta_learner_10day.pkl

Output:
- predictions_latest_10day.csv (date, symbol, ensemble_pred_proba, confidence)

Created: December 4, 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProductionPredictor10Day:
    """Generate 10-day predictions using trained models"""
    
    def __init__(self, prediction_date=None):
        """Initialize paths and prediction date"""
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.features_csv = os.path.join(self.base_path, r'#_feature_engineering\ml_features_master.csv')
        self.model_dir = os.path.join(self.base_path, r'#_model_training\trained_models')
        self.meta_learner_path = os.path.join(self.base_path, r'#_model_training\meta_learner_10day.pkl')
        self.ensemble_log_path = os.path.join(self.base_path, r'#_model_training\ensemble_training_log_10day.csv')
        self.output_csv = os.path.join(self.base_path, r'#_model_training\predictions_latest_10day.csv')
        self.adaptive_features_path = os.path.join(self.base_path, r'#_model_training\adaptive_features_10day.json')
        
        # Prediction date (10 trading days ahead, ~2 weeks)
        if prediction_date is None:
            # Approximate: 10 trading days = 14 calendar days
            self.prediction_date = datetime.now().date() + timedelta(days=14)
        else:
            self.prediction_date = pd.to_datetime(prediction_date).date()
        
        print("=" * 80)
        print("STEP 30e: GENERATE 10-DAY PREDICTIONS (PRODUCTION)")
        print("=" * 80)
        print(f"Features: {os.path.basename(self.features_csv)}")
        print(f"Models: {os.path.basename(self.model_dir)}/*_10day/")
        print(f"Predicting for: ~{self.prediction_date} (10 trading days ahead)")
        print(f"Output: {os.path.basename(self.output_csv)}")
        print("=" * 80)
    
    def load_features(self):
        """Load latest features (most recent date per symbol)"""
        print("\n[1/5] Loading features...")
        
        # Load full dataset
        df = pd.read_csv(self.features_csv, parse_dates=['date'])
        
        # Get most recent date per symbol
        df = df.sort_values(['symbol', 'date'])
        latest_df = df.groupby('symbol').tail(1).reset_index(drop=True)
        
        print(f"  OK Loaded: {len(df):,} total rows")
        print(f"  OK Latest features: {len(latest_df):,} symbols")
        print(f"  OK Most recent date: {latest_df['date'].max().date()}")
        print(f"  OK Features: {len(latest_df.columns):,} columns")
        
        return latest_df
    
    def load_system_metrics(self):
        """Load system-wide performance metrics (AUC, accuracy)"""
        try:
            ensemble_log = pd.read_csv(self.ensemble_log_path)
            ensemble_row = ensemble_log[ensemble_log['model'] == 'Ensemble'].iloc[0]
            return {
                'system_auc': ensemble_row['auc'],
                'system_accuracy': ensemble_row['accuracy'],
                'system_precision': ensemble_row['precision'],
                'system_recall': ensemble_row['recall']
            }
        except Exception as e:
            print(f"  ⚠ Could not load system metrics: {str(e)}")
            return {
                'system_auc': 0.50,
                'system_accuracy': 0.50,
                'system_precision': 0.50,
                'system_recall': 0.50
            }
    
    def select_features(self, df):
        """Select feature columns (exclude metadata)"""
        print("\n[2/5] Selecting features...")
        
        # Exclude non-feature columns
        exclude_cols = ['date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"  OK Selected {len(feature_cols):,} feature columns")
        
        return feature_cols
    
    def load_adaptive_features(self):
        """Load adaptive features from calculated file (or use defaults)"""
        if os.path.exists(self.adaptive_features_path):
            with open(self.adaptive_features_path, 'r') as f:
                features = json.load(f)
            print(f"  OK Adaptive features loaded from: {os.path.basename(self.adaptive_features_path)}")
            print(f"     Calculated: {features.get('calculation_date', 'unknown')} {features.get('calculation_time', '')}")
            print(f"     Based on {features.get('n_predictions_used', 0)} recent predictions")
            return features
        else:
            print(f"  ⚠ Adaptive features file not found: {os.path.basename(self.adaptive_features_path)}")
            print(f"     Using default values (run 00_10_calculate_adaptive_features.py first)")
            return {
                'xgb_recent_accuracy': 0.50,
                'lightgbm_recent_accuracy': 0.53,
                'catboost_recent_accuracy': 0.53,
                'xgb_error_lag1': 0.50,
                'lightgbm_error_lag1': 0.47,
                'catboost_error_lag1': 0.47
            }
    
    def predict_base_models(self, df, feature_cols):
        """Generate predictions from all 3 base models"""
        print("\n[3/5] Generating base model predictions...")
        
        predictions = []
        symbols = sorted(df['symbol'].unique())
        
        print(f"  Predicting for {len(symbols):,} symbols...")
        
        for sym_idx, symbol in enumerate(symbols, 1):
            try:
                # Get symbol data
                symbol_data = df[df['symbol'] == symbol].copy()
                
                if len(symbol_data) == 0:
                    print(f"    WARNING: No data for {symbol}")
                    continue
                
                if len(symbol_data) > 1:
                    print(f"    WARNING: Multiple rows for {symbol}: {len(symbol_data)} rows")
                    symbol_data = symbol_data.iloc[[-1]]  # Take last row
                
                # Load 10-day models for this symbol
                xgb_path = os.path.join(self.model_dir, 'xgb_10day', f'{symbol}.pkl')
                lightgbm_path = os.path.join(self.model_dir, 'lightgbm_10day', f'{symbol}.pkl')
                catboost_path = os.path.join(self.model_dir, 'catboost_10day', f'{symbol}.pkl')
                
                # Skip if any model is missing
                if not all([os.path.exists(p) for p in [xgb_path, lightgbm_path, catboost_path]]):
                    continue
                
                # Load models
                with open(xgb_path, 'rb') as f:
                    xgb_dict = pickle.load(f)
                with open(lightgbm_path, 'rb') as f:
                    lightgbm_dict = pickle.load(f)
                with open(catboost_path, 'rb') as f:
                    catboost_dict = pickle.load(f)
                
                # Use model's stored feature columns (matches training variance filter)
                xgb_fcols = xgb_dict.get('feature_cols', feature_cols)
                lgb_fcols = lightgbm_dict.get('feature_cols', feature_cols)
                cb_fcols = catboost_dict.get('feature_cols', feature_cols)
                
                # Prepare features per model (tree models handle NaN natively)
                X_xgb = symbol_data[xgb_fcols]
                X_lgb = symbol_data[lgb_fcols]
                X_cb = symbol_data[cb_fcols]
                
                # Predict probabilities (no scaler — trees are scale-invariant)
                xgb_proba = xgb_dict['model'].predict_proba(X_xgb)[0, 1]
                lightgbm_proba = lightgbm_dict['model'].predict_proba(X_lgb)[0, 1]
                catboost_proba = catboost_dict['model'].predict_proba(X_cb)[0, 1]
                
                # Store predictions
                predictions.append({
                    'symbol': symbol,
                    'date': symbol_data['date'].iloc[0],
                    'xgb_pred_proba': xgb_proba,
                    'lightgbm_pred_proba': lightgbm_proba,
                    'catboost_pred_proba': catboost_proba
                })
                
                # Progress update every 100 symbols
                if sym_idx % 100 == 0:
                    print(f"    Processed {sym_idx}/{len(symbols)} symbols...")
            
            except Exception as e:
                import traceback
                print(f"    WARNING: Error predicting {symbol}:")
                print(f"      {type(e).__name__}: {str(e)[:500]}")
                traceback.print_exc()  # Uncomment for full traceback
                continue
        
        predictions_df = pd.DataFrame(predictions)
        print(f"  OK Generated predictions for {len(predictions_df):,} symbols")
        
        return predictions_df
    
    def blend_predictions(self, predictions_df):
        """Blend base model predictions using meta-learner"""
        print("\n[4/5] Blending predictions with meta-learner...")
        
        # Load meta-learner (now saved as dict with model + feature_cols)
        with open(self.meta_learner_path, 'rb') as f:
            meta_dict = pickle.load(f)
        
        # Handle both old format (raw model) and new format (dict)
        if isinstance(meta_dict, dict):
            meta_model = meta_dict['model']
            feature_cols = meta_dict['feature_cols']
            print(f"  OK Loaded meta-learner (holdout AUC: {meta_dict.get('holdout_auc', 'N/A')})")
        else:
            meta_model = meta_dict
            feature_cols = ['xgb_pred_proba', 'lightgbm_pred_proba', 'catboost_pred_proba']
            print(f"  OK Loaded meta-learner (legacy format)")
        
        print(f"  OK Features: {feature_cols}")
        
        # Prepare features for meta-learner
        X_meta = predictions_df[feature_cols]
        
        # Predict ensemble probabilities
        ensemble_proba = meta_model.predict_proba(X_meta)[:, 1]
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Calculate confidence (distance from 0.5, scaled to [0, 1])
        confidence = np.abs(ensemble_proba - 0.5) * 2
        
        # Add to dataframe
        predictions_df['ensemble_pred_proba'] = ensemble_proba
        predictions_df['ensemble_pred'] = ensemble_pred
        predictions_df['confidence'] = confidence
        
        # Add prediction labels
        predictions_df['prediction'] = predictions_df['ensemble_pred'].map({1: 'Outperform', 0: 'Underperform'})
        
        print(f"  OK Ensemble predictions generated")
        print(f"  OK Up predictions: {(predictions_df['ensemble_pred'] == 1).sum():,} (Outperform market)")
        print(f"  OK Down predictions: {(predictions_df['ensemble_pred'] == 0).sum():,} (Underperform market)")
        print(f"  OK Average confidence: {predictions_df['confidence'].mean():.3f}")
        
        return predictions_df
    
    def save_predictions(self, predictions_df):
        """Save predictions to CSV"""
        print("\n[5/5] Saving predictions...")
        
        # Add prediction date
        predictions_df['prediction_date'] = self.prediction_date
        
        # Reorder columns
        output_cols = [
            'prediction_date', 'symbol', 'date',
            'ensemble_pred_proba', 'ensemble_pred', 'prediction', 'confidence',
            'xgb_pred_proba', 'lightgbm_pred_proba', 'catboost_pred_proba'
        ]
        predictions_df = predictions_df[output_cols]
        
        # Sort by confidence (highest first)
        predictions_df = predictions_df.sort_values('confidence', ascending=False)
        
        # Save to CSV
        predictions_df.to_csv(self.output_csv, index=False)
        print(f"  OK Predictions saved: {os.path.basename(self.output_csv)}")
        print(f"  OK Total predictions: {len(predictions_df):,}")
        
        # Show top 10 high-confidence predictions
        print("\n  Top 10 High-Confidence Predictions:")
        top10 = predictions_df.head(10)
        for _, row in top10.iterrows():
            print(f"    {row['symbol']}: {row['prediction']} (prob={row['ensemble_pred_proba']:.3f}, conf={row['confidence']:.3f})")
        
        return predictions_df
    
    def print_summary(self, predictions_df, system_metrics):
        """Print summary statistics"""
        print("\n" + "=" * 80)
        print("OK Prediction generation complete!")
        print("=" * 80)
        
        # System-wide performance (from training)
        print("\n  Ensemble Performance (Training):")
        print(f"    AUC: {system_metrics['system_auc']:.1%}")
        print(f"    Accuracy: {system_metrics['system_accuracy']:.1%}")
        print(f"    Precision: {system_metrics['system_precision']:.1%}")
        print(f"    Recall: {system_metrics['system_recall']:.1%}")
        
        # Confidence distribution
        print("\n  Confidence Distribution:")
        print(f"    High (>0.6): {(predictions_df['confidence'] > 0.6).sum():,} ({(predictions_df['confidence'] > 0.6).mean():.1%})")
        print(f"    Medium (0.3-0.6): {((predictions_df['confidence'] >= 0.3) & (predictions_df['confidence'] <= 0.6)).sum():,}")
        print(f"    Low (<0.3): {(predictions_df['confidence'] < 0.3).sum():,} ({(predictions_df['confidence'] < 0.3).mean():.1%})")
        
        # Up/Down split
        print("\n  Prediction Split:")
        print(f"    Outperform market: {(predictions_df['ensemble_pred'] == 1).sum():,} ({(predictions_df['ensemble_pred'] == 1).mean():.1%})")
        print(f"    Underperform market: {(predictions_df['ensemble_pred'] == 0).sum():,} ({(predictions_df['ensemble_pred'] == 0).mean():.1%})")
        
        print("\n" + "=" * 80)
        print(f"Predictions saved to: {self.output_csv}")
        print("=" * 80)
    
    def run(self):
        """Execute full prediction pipeline"""
        start_time = datetime.now()
        
        # Load system metrics
        system_metrics = self.load_system_metrics()
        
        # Load features
        df = self.load_features()
        feature_cols = self.select_features(df)
        
        # Generate base model predictions
        predictions_df = self.predict_base_models(df, feature_cols)
        
        # Blend predictions
        predictions_df = self.blend_predictions(predictions_df)
        
        # Save predictions
        predictions_df = self.save_predictions(predictions_df)
        
        # Print summary
        self.print_summary(predictions_df, system_metrics)
        
        # Execution time
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nTIME: Total execution time: {duration:.1f} seconds")


if __name__ == '__main__':
    # Usage: python 01_05_generate_predictions.py
    # Optional: Specify prediction date
    # predictor = ProductionPredictor(prediction_date='2025-10-17')
    
    predictor = ProductionPredictor10Day()
    predictor.run()
