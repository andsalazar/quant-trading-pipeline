#!/usr/bin/env python3
"""
BACKTESTING SCRIPT: Evaluate Ensemble Trading Strategy
========================================================

Purpose:
- Backtest high-confidence trading strategy on historical validation data
- Test multiple confidence thresholds (0.05, 0.10, 0.15, 0.20)
- Calculate performance metrics: accuracy, Sharpe ratio, returns, win rate
- Compare ensemble vs base models vs buy-and-hold

Input:
- ensemble_predictions_validation.csv (591,903 out-of-sample predictions 2021-2025)

Output:
- Backtest results by confidence threshold
- Performance comparison across models
- Trading statistics and risk metrics

Created: October 17, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """Backtest trading strategies on historical predictions"""
    
    def __init__(self):
        """Initialize backtest engine"""
        self.predictions_file = r'ensemble_predictions_validation.csv'
        self.results = {}
        
    def load_data(self):
        """Load historical validation predictions"""
        print("[1/6] Loading historical predictions...")
        df = pd.read_csv(self.predictions_file)
        print(f"  OK Loaded {len(df):,} predictions")
        print(f"  OK Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  OK Symbols: {df['symbol'].nunique()}")
        print(f"  OK Folds: {df['fold'].nunique()}")
        return df
    
    def calculate_returns(self, df):
        """
        Calculate hypothetical returns based on predictions
        
        Strategy:
        - If predict UP (ensemble_pred=1) and actual=1: +1 return
        - If predict DOWN (ensemble_pred=0) and actual=0: +1 return  
        - Otherwise: -1 return
        """
        print("\n[2/6] Calculating returns...")
        
        # Binary returns (+1 if correct, -1 if wrong)
        df['return'] = np.where(df['ensemble_pred'] == df['actual'], 1, -1)
        
        # Percent correct
        df['correct'] = (df['ensemble_pred'] == df['actual']).astype(int)
        
        print(f"  OK Overall accuracy: {df['correct'].mean():.2%}")
        print(f"  OK Total predictions: {len(df):,}")
        return df
    
    def backtest_threshold(self, df, threshold, name):
        """Backtest a specific confidence threshold"""
        
        # Filter by confidence
        filtered = df[df['confidence'] >= threshold].copy()
        
        if len(filtered) == 0:
            return None
        
        # Calculate metrics
        accuracy = filtered['correct'].mean()
        total_trades = len(filtered)
        total_return = filtered['return'].sum()
        avg_return = filtered['return'].mean()
        
        # Win/Loss breakdown
        wins = (filtered['return'] > 0).sum()
        losses = (filtered['return'] < 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (annualized, assuming 252 trading days)
        daily_returns = filtered.groupby('date')['return'].mean()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Max drawdown
        cumulative = filtered.groupby('date')['return'].sum().cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Per-fold performance
        fold_performance = []
        for fold in sorted(filtered['fold'].unique()):
            fold_df = filtered[filtered['fold'] == fold]
            fold_acc = fold_df['correct'].mean()
            fold_performance.append({
                'fold': fold,
                'accuracy': fold_acc,
                'trades': len(fold_df)
            })
        
        results = {
            'name': name,
            'threshold': threshold,
            'accuracy': accuracy,
            'total_trades': total_trades,
            'trades_pct': total_trades / len(df),
            'total_return': total_return,
            'avg_return': avg_return,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'fold_performance': fold_performance,
            'avg_confidence': filtered['confidence'].mean()
        }
        
        return results
    
    def run_backtest(self, df):
        """Run backtest across multiple confidence thresholds"""
        print("\n[3/6] Running backtest across confidence thresholds...")
        
        thresholds = [
            (0.00, "All Predictions"),
            (0.05, "Moderate Confidence"),
            (0.09, "High Confidence (Top 20%)"),
            (0.10, "Very High Confidence"),
            (0.15, "Ultra High Confidence"),
            (0.20, "Maximum Confidence")
        ]
        
        results = []
        for threshold, name in thresholds:
            print(f"\n  Testing: {name} (threshold >= {threshold:.2f})")
            result = self.backtest_threshold(df, threshold, name)
            if result:
                results.append(result)
                print(f"    -> Accuracy: {result['accuracy']:.2%}, Trades: {result['total_trades']:,}, Sharpe: {result['sharpe_ratio']:.2f}")
        
        return results
    
    def compare_models(self, df):
        """Compare ensemble vs individual base models"""
        print("\n[4/6] Comparing ensemble vs base models...")
        
        models = []
        
        # XGBoost (convert probability to prediction)
        xgb_pred = (df['xgb_pred_proba'] > 0.5).astype(int)
        xgb_correct = (xgb_pred == df['actual']).mean()
        models.append({'model': 'XGBoost', 'accuracy': xgb_correct})
        print(f"  XGBoost:  {xgb_correct:.2%}")
        
        # LightGBM
        lgbm_pred = (df['lightgbm_pred_proba'] > 0.5).astype(int)
        lgbm_correct = (lgbm_pred == df['actual']).mean()
        models.append({'model': 'LightGBM', 'accuracy': lgbm_correct})
        print(f"  LightGBM: {lgbm_correct:.2%}")
        
        # CatBoost
        catb_pred = (df['catboost_pred_proba'] > 0.5).astype(int)
        catb_correct = (catb_pred == df['actual']).mean()
        models.append({'model': 'CatBoost', 'accuracy': catb_correct})
        print(f"  CatBoost: {catb_correct:.2%}")
        
        # Ensemble
        ensemble_correct = df['correct'].mean()
        models.append({'model': 'Ensemble', 'accuracy': ensemble_correct})
        print(f"  Ensemble: {ensemble_correct:.2%}")
        
        return pd.DataFrame(models)
    
    def analyze_by_fold(self, df):
        """Analyze performance by validation fold (year)"""
        print("\n[5/6] Analyzing performance by fold/year...")
        
        fold_results = []
        for fold in sorted(df['fold'].unique()):
            fold_df = df[df['fold'] == fold]
            
            # Determine year (fold 1 = 2021, fold 2 = 2022, etc.)
            year = 2020 + fold
            
            accuracy = fold_df['correct'].mean()
            trades = len(fold_df)
            avg_conf = fold_df['confidence'].mean()
            
            # High confidence subset (top 20%)
            high_conf_threshold = fold_df['confidence'].quantile(0.8)
            high_conf_df = fold_df[fold_df['confidence'] >= high_conf_threshold]
            high_conf_acc = high_conf_df['correct'].mean() if len(high_conf_df) > 0 else 0
            
            fold_results.append({
                'fold': fold,
                'year': year,
                'accuracy': accuracy,
                'high_conf_accuracy': high_conf_acc,
                'trades': trades,
                'avg_confidence': avg_conf
            })
            
            print(f"  Fold {fold} ({year}): Acc {accuracy:.2%}, High-Conf {high_conf_acc:.2%}, Trades {trades:,}")
        
        return pd.DataFrame(fold_results)
    
    def print_summary(self, backtest_results, model_comparison, fold_analysis):
        """Print comprehensive backtest summary"""
        print("\n" + "="*80)
        print("=== BACKTEST SUMMARY ===")
        print("="*80)
        
        # Confidence threshold comparison
        print("\n[A] CONFIDENCE THRESHOLD COMPARISON")
        print("-" * 80)
        print(f"{'Strategy':<30} {'Threshold':<10} {'Accuracy':<10} {'Trades':<10} {'Sharpe':<10} {'Win Rate':<10}")
        print("-" * 80)
        
        for result in backtest_results:
            print(f"{result['name']:<30} {result['threshold']:<10.2f} "
                  f"{result['accuracy']:<10.2%} {result['total_trades']:<10,} "
                  f"{result['sharpe_ratio']:<10.2f} {result['win_rate']:<10.2%}")
        
        # Find optimal threshold
        best_result = max(backtest_results, key=lambda x: x['accuracy'])
        print(f"\nOptimal Strategy: {best_result['name']} (Accuracy: {best_result['accuracy']:.2%})")
        
        # Model comparison
        print("\n[B] MODEL COMPARISON")
        print("-" * 80)
        print(model_comparison.to_string(index=False))
        
        # Fold/Year analysis
        print("\n[C] PERFORMANCE BY YEAR")
        print("-" * 80)
        print(fold_analysis.to_string(index=False))
        
        # Key insights
        print("\n[D] KEY INSIGHTS")
        print("-" * 80)
        
        all_preds = backtest_results[0]  # All predictions baseline
        best_high_conf = backtest_results[2] if len(backtest_results) > 2 else None  # High confidence
        
        print(f"1. Baseline (All Predictions):")
        print(f"   - Accuracy: {all_preds['accuracy']:.2%}")
        print(f"   - Total Trades: {all_preds['total_trades']:,}")
        print(f"   - Sharpe Ratio: {all_preds['sharpe_ratio']:.2f}")
        
        if best_high_conf:
            print(f"\n2. High-Confidence Strategy (Top 20%):")
            print(f"   - Accuracy: {best_high_conf['accuracy']:.2%} (+{(best_high_conf['accuracy'] - all_preds['accuracy']):.2%})")
            print(f"   - Total Trades: {best_high_conf['total_trades']:,} ({best_high_conf['trades_pct']:.1%} of all)")
            print(f"   - Sharpe Ratio: {best_high_conf['sharpe_ratio']:.2f}")
            print(f"   - Win Rate: {best_high_conf['win_rate']:.2%}")
        
        print(f"\n3. Ensemble Advantage:")
        ensemble_acc = model_comparison[model_comparison['model'] == 'Ensemble']['accuracy'].values[0]
        best_base = model_comparison[model_comparison['model'] != 'Ensemble']['accuracy'].max()
        print(f"   - Ensemble: {ensemble_acc:.2%}")
        print(f"   - Best Base Model: {best_base:.2%}")
        print(f"   - Improvement: +{(ensemble_acc - best_base):.2%}")
        
        # Consistency
        fold_std = fold_analysis['accuracy'].std()
        print(f"\n4. Consistency Across Years:")
        print(f"   - Accuracy Std Dev: {fold_std:.2%}")
        print(f"   - Best Year: {fold_analysis.loc[fold_analysis['accuracy'].idxmax(), 'year']:.0f} "
              f"({fold_analysis['accuracy'].max():.2%})")
        print(f"   - Worst Year: {fold_analysis.loc[fold_analysis['accuracy'].idxmin(), 'year']:.0f} "
              f"({fold_analysis['accuracy'].min():.2%})")
    
    def save_results(self, backtest_results):
        """Save backtest results to CSV"""
        print("\n[6/6] Saving results...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(backtest_results)
        results_df = results_df.drop('fold_performance', axis=1)  # Remove nested data
        
        # Save
        output_file = 'backtest_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"  OK Saved to: {output_file}")
        
        return output_file
    
    def run(self):
        """Main execution"""
        print("\n" + "="*80)
        print("ENSEMBLE TRADING STRATEGY BACKTEST")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        df = self.load_data()
        
        # Calculate returns
        df = self.calculate_returns(df)
        
        # Run backtest
        backtest_results = self.run_backtest(df)
        
        # Compare models
        model_comparison = self.compare_models(df)
        
        # Analyze by fold
        fold_analysis = self.analyze_by_fold(df)
        
        # Print summary
        self.print_summary(backtest_results, model_comparison, fold_analysis)
        
        # Save results
        self.save_results(backtest_results)
        
        print("\n" + "="*80)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        return backtest_results, model_comparison, fold_analysis

if __name__ == "__main__":
    engine = BacktestEngine()
    backtest_results, model_comparison, fold_analysis = engine.run()
