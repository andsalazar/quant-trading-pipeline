#!/usr/bin/env python3
"""
STEP 30: MASTER MODEL TRAINING SCRIPT (10-DAY HORIZON)
=======================================================

Purpose:
- Run all 10-day model training scripts in sequence
- Train XGBoost, LightGBM, CatBoost, and Meta-Learner for 10-day predictions
- Generate production predictions for ~2 weeks ahead

Execution Order:
1. 03_01_train_xgb_10day.py (~30-60 min)
2. 03_02_train_lightgbm_10day.py (~20-40 min)
3. 03_03_train_catboost_10day.py (~25-50 min)
4. 03_04_train_meta_learner_10day.py (~1 min)
5. 03_05_generate_predictions_10day.py (~1 min)

Total Time: ~1.5-2.5 hours

Created: January 11, 2026
"""

import subprocess
import os
import sys
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and capture output"""
    print("\n" + "=" * 80)
    print(f"RUNNING: {script_name}")
    print(f"Description: {description}")
    print("=" * 80)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    start_time = datetime.now()
    
    try:
        # Run script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n✅ {script_name} completed successfully ({duration:.1f}s)")
            return True
        else:
            print(f"\n❌ {script_name} failed with return code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {str(e)}")
        return False

def main():
    """Run full 10-day training pipeline"""
    print("=" * 80)
    print("STEP 30: MASTER MODEL TRAINING PIPELINE (10-DAY HORIZON)")
    print("=" * 80)
    print("Strategy: XGBoost + LightGBM + CatBoost → Meta-Learner → Predictions")
    print("Horizon: 10 trading days (~2 weeks ahead)")
    print("Estimated time: 1.5-2.5 hours")
    print("=" * 80)
    
    overall_start = datetime.now()
    
    # Script sequence
    scripts = [
        ('03_01_train_xgb_10day.py', 'Train XGBoost per-symbol (10-day walk-forward)'),
        ('03_02_train_lightgbm_10day.py', 'Train LightGBM per-symbol (10-day walk-forward)'),
        ('03_03_train_catboost_10day.py', 'Train CatBoost per-symbol (10-day walk-forward)'),
        ('03_04_train_meta_learner_10day.py', 'Train meta-learner ensemble (blend predictions)'),
        ('03_05_generate_predictions_10day.py', 'Generate production predictions (10 days ahead)')
    ]
    
    # Run each script
    results = []
    for script_name, description in scripts:
        success = run_script(script_name, description)
        results.append((script_name, success))
        
        if not success:
            print("\n" + "=" * 80)
            print(f"❌ PIPELINE FAILED at {script_name}")
            print("=" * 80)
            print("\nFix the issue and re-run this script to continue.")
            return 1
    
    # Summary
    overall_duration = (datetime.now() - overall_start).total_seconds()
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    
    print("\nResults:")
    for script_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {script_name}")
    
    print(f"\nTotal Duration: {overall_duration:.1f} seconds ({overall_duration/60:.1f} minutes)")
    
    # Check if all succeeded
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✅ ALL TRAINING STEPS COMPLETED SUCCESSFULLY!")
        print("\nNext Steps:")
        print("  1. Review predictions_latest_10day.csv")
        print("  2. Run 00_10_calculate_adaptive_features.py to update adaptive weights")
        print("  3. Resume daily workflow with MASTER_DAILY_WORKFLOW_10DAY.bat")
        return 0
    else:
        print("\n❌ SOME TRAINING STEPS FAILED - Check logs above")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
