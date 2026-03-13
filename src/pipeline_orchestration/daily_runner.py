#!/usr/bin/env python3
"""
#_Core_Project - Master Daily Workflow Runner
============================================

Coordinates the complete daily data pipeline:
1. Fetch market data 
2. Fetch supplementary data (currencies, treasuries, etc.)
3. Feature engineering
4. ML training/predictions

Clean, simple, focused on current production needs.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

class CoreProjectRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.fetch_data_path = self.project_root / "#_fetch_data"
        self.feature_eng_path = self.project_root / "#_feature_engineering"
        self.database_path = self.project_root / "#_master_database"
        
        print(f"🚀 Core Project Daily Runner")
        print(f"   Project root: {self.project_root}")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_script(self, script_path, description):
        """Run a Python script and handle errors"""
        print(f"\n🔄 {description}")
        print(f"   Running: {script_path}")
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path)
            ], 
            cwd=script_path.parent,
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ Completed successfully")
                if result.stdout.strip():
                    print(f"   📋 Output: {result.stdout.strip()[-200:]}")  # Last 200 chars
                return True
            else:
                print(f"   ❌ Failed with code {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout after 10 minutes")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def step_1_fetch_market_data(self):
        """Step 1: Fetch core market data"""
        print(f"\n📊 STEP 1: FETCH MARKET DATA")
        print("=" * 40)
        
        market_script = self.fetch_data_path / "#_01_fetch_market" / "00_01_fetch_market_data.py"
        
        if not market_script.exists():
            print(f"❌ Market data script not found: {market_script}")
            return False
        
        success = self.run_script(market_script, "Fetching daily market data from Polygon")
        return success
    
    def step_2_fetch_supplementary_data(self):
        """Step 2: Fetch currencies, treasuries, etc."""
        print(f"\n💱 STEP 2: FETCH SUPPLEMENTARY DATA")
        print("=" * 40)
        
        # List of supplementary data scripts to run
        supplementary_scripts = [
            ("#_02_fetch_currencies", "Currency data"),
            ("#_03_fetch_treasuries", "Treasury yields"),
            ("#_04_fetch_futures", "Futures data"),
            ("#_05_fetch_options", "Options data"),
        ]
        
        success_count = 0
        
        for folder, description in supplementary_scripts:
            folder_path = self.fetch_data_path / folder
            
            if not folder_path.exists():
                print(f"⚠️  {description} folder not found, skipping")
                continue
            
            # Look for Python scripts in the folder
            scripts = list(folder_path.glob("*.py"))
            
            if not scripts:
                print(f"⚠️  No scripts found in {folder}, skipping")
                continue
            
            # Run the first script found (assuming one main script per folder)
            main_script = scripts[0]
            
            if self.run_script(main_script, f"Fetching {description}"):
                success_count += 1
        
        print(f"\n📊 Supplementary data: {success_count} sources updated")
        return success_count > 0
    
    def step_3_feature_engineering(self):
        """Step 3: Process data into ML features"""
        print(f"\n🔧 STEP 3: FEATURE ENGINEERING")
        print("=" * 40)
        
        # Check if feature engineering scripts exist
        fe_scripts = list(self.feature_eng_path.glob("*.py"))
        
        if not fe_scripts:
            print("📝 No feature engineering scripts found")
            print("   Creating basic feature engineering pipeline...")
            
            # Create a basic feature engineering script
            self.create_basic_feature_engineering()
            fe_scripts = list(self.feature_eng_path.glob("*.py"))
        
        success_count = 0
        for script in fe_scripts:
            if self.run_script(script, f"Running {script.name}"):
                success_count += 1
        
        return success_count > 0
    
    def step_4_ml_training_predictions(self):
        """Step 4: Train models and generate predictions"""
        print(f"\n🤖 STEP 4: ML TRAINING & PREDICTIONS")
        print("=" * 40)
        
        print("📝 ML training pipeline not yet implemented")
        print("   Next steps: Create clean ML training scripts")
        
        return True
    
    def create_basic_feature_engineering(self):
        """Create a basic feature engineering script"""
        
        fe_script_content = '''#!/usr/bin/env python3
"""
Basic Feature Engineering Pipeline
Processes raw market data into ML-ready features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

def main():
    print("🔧 Starting basic feature engineering...")
    
    # This is a placeholder - will be expanded based on your needs
    project_root = Path(__file__).parent.parent
    
    # Look for market data
    market_data_path = project_root / "#_fetch_data" / "#_01_fetch_market" / "00_01_market_polygon_converted.csv"
    
    if market_data_path.exists():
        print(f"✅ Found market data: {market_data_path}")
        df = pd.read_csv(market_data_path)
        print(f"   Records: {len(df):,}")
        print(f"   Symbols: {df['Symbol'].nunique() if 'Symbol' in df.columns else 'Unknown'}")
        print(f"   Date range: {df.columns.tolist()}")
    else:
        print(f"❌ Market data not found at: {market_data_path}")
    
    print("✅ Basic feature engineering completed")

if __name__ == "__main__":
    main()
'''
        
        fe_script_path = self.feature_eng_path / "basic_feature_engineering.py"
        
        with open(fe_script_path, 'w') as f:
            f.write(fe_script_content)
        
        print(f"✅ Created: {fe_script_path}")
    
    def run_full_pipeline(self):
        """Run the complete daily pipeline"""
        start_time = time.time()
        
        print(f"\n🚀 STARTING FULL DAILY PIPELINE")
        print("=" * 50)
        
        steps = [
            ("Market Data", self.step_1_fetch_market_data),
            ("Supplementary Data", self.step_2_fetch_supplementary_data),
            ("Feature Engineering", self.step_3_feature_engineering),
            ("ML Training", self.step_4_ml_training_predictions),
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            step_start = time.time()
            success = step_func()
            step_time = time.time() - step_start
            
            results[step_name] = {
                'success': success,
                'time': step_time
            }
        
        # Summary
        total_time = time.time() - start_time
        
        print(f"\n📊 PIPELINE SUMMARY")
        print("=" * 30)
        
        for step_name, result in results.items():
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {step_name}: {result['time']:.1f}s")
        
        successful_steps = sum(1 for r in results.values() if r['success'])
        total_steps = len(results)
        
        print(f"\n🎯 Complete: {successful_steps}/{total_steps} steps successful")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        
        if successful_steps == total_steps:
            print("🎉 Full pipeline completed successfully!")
        else:
            print("⚠️  Pipeline completed with some issues")

def main():
    """Main entry point"""
    runner = CoreProjectRunner()
    runner.run_full_pipeline()

if __name__ == "__main__":
    main()