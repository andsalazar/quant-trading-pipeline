"""
Master Daily Update Script
===========================

Purpose:
- Run complete daily update pipeline (Steps 0-8b)
- Fetch latest data from all sources
- Update database with new records
- Generate update summary report

Usage:
- Run every trading day (automated via Task Scheduler)
- Executes in sequence: data fetchers → database update
- Total duration: ~15-30 seconds

Created: October 16, 2025
"""

import subprocess
import os
import sys
from datetime import datetime
import time

# Force UTF-8 output for Task Scheduler (cp1252 can't handle Unicode symbols)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = CONFIG_BASE_PATH  # Set in config.py
PYTHON_EXE = r'<project_root>nda\envs\xgb_gpu_env\python.exe'

print("=" * 80)
print("MASTER DAILY UPDATE - Complete Pipeline")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Track execution
start_time = datetime.now()
results = []

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def run_script(step_name, script_path, description):
    """Run a Python script and track results"""
    print(f"\n[{step_name}] {description}")
    print(f"  Script: {os.path.basename(script_path)}")
    
    step_start = datetime.now()
    
    try:
        # Set UTF-8 encoding environment variable for subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [PYTHON_EXE, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of crashing
            timeout=7200,  # 2 hour timeout (for sentiment analyzer)
            env=env
        )
        
        elapsed = (datetime.now() - step_start).total_seconds()
        
        if result.returncode == 0:
            print(f"  ✓ SUCCESS ({elapsed:.1f}s)")
            results.append({
                'step': step_name,
                'status': 'SUCCESS',
                'elapsed': elapsed,
                'script': os.path.basename(script_path)
            })
            return True
        else:
            print(f"  ✗ FAILED ({elapsed:.1f}s)")
            # Show last 500 chars of error for better debugging
            error_msg = result.stderr[-500:] if result.stderr else "No error message"
            print(f"  Error: {error_msg}")
            results.append({
                'step': step_name,
                'status': 'FAILED',
                'elapsed': elapsed,
                'script': os.path.basename(script_path),
                'error': error_msg
            })
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  \u2717 TIMEOUT (exceeded 2 hours)")
        results.append({
            'step': step_name,
            'status': 'TIMEOUT',
            'elapsed': 7200,
            'script': os.path.basename(script_path)
        })
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        results.append({
            'step': step_name,
            'status': 'ERROR',
            'elapsed': 0,
            'script': os.path.basename(script_path),
            'error': str(e)
        })
        return False

# ============================================================================
# STEP 0: TICKER UNIVERSE (Optional - usually static)
# ============================================================================

# Skip ticker universe update for daily operations (rarely changes)
print("\n[STEP 0] Ticker Universe")
print("  → Skipped (no daily updates needed)")

# ============================================================================
# STEP 1: MARKET DATA
# ============================================================================

# Step 1a: Fetch market data
market_fetcher = os.path.join(BASE_PATH, r'#_fetch_data\#_01_fetch_market\00_01_fetch_market_data.py')
if os.path.exists(market_fetcher):
    run_script('STEP 1a', market_fetcher, 'Fetch latest market data (OHLCV)')
else:
    print(f"\n[STEP 1a] Market Data Fetcher")
    print(f"  ⚠ Script not found: {market_fetcher}")
    results.append({'step': 'STEP 1a', 'status': 'NOT_FOUND', 'elapsed': 0})

# Step 1b: Enhance market data
market_enhancer = os.path.join(BASE_PATH, r'#_fetch_data\#_01_fetch_market\00_01_market_simple_enhancer.py')
if os.path.exists(market_enhancer):
    run_script('STEP 1b', market_enhancer, 'Market enhancements (technical indicators + features)')
else:
    print(f"\n[STEP 1b] Market Enhancer")
    print(f"  ⚠ Script not found: {market_enhancer}")
    results.append({'step': 'STEP 1b', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 2: CURRENCY DATA
# ============================================================================

# Step 2a: Fetch currency data
currency_fetcher = os.path.join(BASE_PATH, r'#_fetch_data\#_02_fetch_currencies\currency_fetcher_clean.py')
if os.path.exists(currency_fetcher):
    run_script('STEP 2a', currency_fetcher, 'Fetch latest currency data (FX pairs)')
else:
    print(f"\n[STEP 2a] Currency Fetcher")
    print(f"  ⚠ Script not found: {currency_fetcher}")
    results.append({'step': 'STEP 2a', 'status': 'NOT_FOUND', 'elapsed': 0})

# Step 2b: Enhance currency data
currency_enhancer = os.path.join(BASE_PATH, r'#_fetch_data\#_02_fetch_currencies\00_02_currency_daily_enhancer.py')
if os.path.exists(currency_enhancer):
    run_script('STEP 2b', currency_enhancer, 'Currency enhancements (technical indicators + features)')
else:
    print(f"\n[STEP 2b] Currency Enhancer")
    print(f"  ⚠ Script not found: {currency_enhancer}")
    results.append({'step': 'STEP 2b', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 3: TREASURY DATA
# ============================================================================

treasury_fetcher = os.path.join(BASE_PATH, r'#_fetch_data\#_03_fetch_treasuries\00_03_enhanced_treasuryyields_fetcher.py')
if os.path.exists(treasury_fetcher):
    run_script('STEP 3', treasury_fetcher, 'Fetch latest treasury yields')
else:
    print(f"\n[STEP 3] Treasury Data")
    print(f"  ⚠ Script not found: {treasury_fetcher}")
    results.append({'step': 'STEP 3', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 4: FUTURES DATA
# ============================================================================

futures_updater = os.path.join(BASE_PATH, r'#_fetch_data\#_04_fetch_futures\00_04_futures_daily_updater.py')
if os.path.exists(futures_updater):
    run_script('STEP 4', futures_updater, 'Fetch latest futures data (ES, TY, CL, etc.)')
else:
    print(f"\n[STEP 4] Futures Data")
    print(f"  ⚠ Script not found: {futures_updater}")
    results.append({'step': 'STEP 4', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 5: OPTIONS DATA
# ============================================================================

options_updater = os.path.join(BASE_PATH, r'#_fetch_data\#_05_fetch_options\00_05_options_daily_updater.py')
if os.path.exists(options_updater):
    run_script('STEP 5', options_updater, 'Fetch latest options data (P/C ratios, IV)')
else:
    print(f"\n[STEP 5] Options Data")
    print(f"  ⚠ Script not found: {options_updater}")
    results.append({'step': 'STEP 5', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 6: MACRO EVENTS
# ============================================================================

# Step 6a: Fetch macro events from BLS.gov + FOMC calendar
events_fetcher = os.path.join(BASE_PATH, r'#_fetch_data\#_06_fetch_events\00_06_fetch_macro_events.py')
if os.path.exists(events_fetcher):
    run_script('STEP 6a', events_fetcher, 'Fetch macro events from BLS.gov + FOMC calendar')
else:
    print(f"\n[STEP 6a] Events Fetcher")
    print(f"  ⚠ Script not found: {events_fetcher}")
    results.append({'step': 'STEP 6a', 'status': 'NOT_FOUND', 'elapsed': 0})

# Step 6b: Process events into ML features
events_updater = os.path.join(BASE_PATH, r'#_fetch_data\#_06_fetch_events\00_06_events_daily_updater.py')
if os.path.exists(events_updater):
    run_script('STEP 6b', events_updater, 'Process macro events features')
else:
    print(f"\n[STEP 6b] Events Processor")
    print(f"  ⚠ Script not found: {events_updater}")
    results.append({'step': 'STEP 6b', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 7: NEWS & SENTIMENT
# ============================================================================

# Step 7a: Fetch headlines
headlines_fetcher = os.path.join(BASE_PATH, r'#_fetch_data\#_07_news_sentiment\00_07_headlines_fetcher.py')
if os.path.exists(headlines_fetcher):
    run_script('STEP 7a', headlines_fetcher, 'Fetch latest news headlines')
else:
    print(f"\n[STEP 7a] Headlines")
    print(f"  ⚠ Script not found: {headlines_fetcher}")
    results.append({'step': 'STEP 7a', 'status': 'NOT_FOUND', 'elapsed': 0})

# Step 7b: Analyze sentiment
sentiment_updater = os.path.join(BASE_PATH, r'#_fetch_data\#_07_news_sentiment\00_07_sentiment_daily_updater.py')
if os.path.exists(sentiment_updater):
    run_script('STEP 7b', sentiment_updater, 'Analyze sentiment (OpenAI GPT)')
else:
    print(f"\n[STEP 7b] Sentiment Analysis")
    print(f"  ⚠ Script not found: {sentiment_updater}")
    results.append({'step': 'STEP 7b', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 8b: DATABASE UPDATE (Incremental)
# ============================================================================

db_updater = os.path.join(BASE_PATH, r'#_master_database\00_08b_update_database_daily.py')
if os.path.exists(db_updater):
    run_script('STEP 8b', db_updater, 'Update database (incremental append)')
else:
    print(f"\n[STEP 8b] Database Update")
    print(f"  ⚠ Script not found: {db_updater}")
    results.append({'step': 'STEP 8b', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 9: FEATURE ENGINEERING
# ============================================================================

feature_eng = os.path.join(BASE_PATH, r'#_feature_engineering\00_09_feature_engineering.py')
if os.path.exists(feature_eng):
    run_script('STEP 9', feature_eng, 'Feature engineering (ml_features_master.csv)')
else:
    print(f"\n[STEP 9] Feature Engineering")
    print(f"  ⚠ Script not found: {feature_eng}")
    results.append({'step': 'STEP 9', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# STEP 10: SPY TIMING SIGNAL
# ============================================================================

# Step 10a: Backfill actuals from yesterday's prediction & compute accuracy
spy_tracker = os.path.join(BASE_PATH, r'#_model_training\06_03_track_spy_timing.py')
if os.path.exists(spy_tracker):
    run_script('STEP 10a', spy_tracker, 'SPY timing — backfill actuals & accuracy dashboard')
else:
    print(f"\n[STEP 10a] SPY Timing Tracker")
    print(f"  ⚠ Script not found: {spy_tracker}")
    results.append({'step': 'STEP 10a', 'status': 'NOT_FOUND', 'elapsed': 0})

# Step 10b: Generate today's BUY/CASH signal
spy_predictor = os.path.join(BASE_PATH, r'#_model_training\06_02_predict_spy_timing.py')
if os.path.exists(spy_predictor):
    run_script('STEP 10b', spy_predictor, 'SPY timing — generate BUY/CASH signal')
else:
    print(f"\n[STEP 10b] SPY Timing Predictor")
    print(f"  ⚠ Script not found: {spy_predictor}")
    results.append({'step': 'STEP 10b', 'status': 'NOT_FOUND', 'elapsed': 0})

# ============================================================================
# SUMMARY
# ============================================================================

total_elapsed = (datetime.now() - start_time).total_seconds()

print("\n" + "=" * 80)
print("DAILY UPDATE COMPLETE")
print("=" * 80)

# Count successes/failures
success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
failed_count = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])
not_found_count = sum(1 for r in results if r['status'] == 'NOT_FOUND')

print(f"\nExecution Summary:")
print(f"  Total Steps: {len(results)}")
print(f"  ✓ Success: {success_count}")
print(f"  ✗ Failed: {failed_count}")
print(f"  ⚠ Not Found: {not_found_count}")
print(f"  Total Duration: {total_elapsed:.1f} seconds")

print(f"\nDetailed Results:")
for r in results:
    status_icon = {
        'SUCCESS': '✓',
        'FAILED': '✗',
        'ERROR': '✗',
        'TIMEOUT': '✗',
        'NOT_FOUND': '⚠'
    }.get(r['status'], '?')
    
    print(f"  [{status_icon}] {r['step']}: {r['status']} ({r.get('elapsed', 0):.1f}s)")
    if 'error' in r:
        print(f"      Error: {r['error'][:100]}")

print("\n" + "=" * 80)

if failed_count > 0:
    print("[WARNING] Some steps failed - check logs above")
    print("Recommendation: Review failed scripts and re-run manually")
elif not_found_count > 0:
    print("[INFO] Some scripts not found - check file paths")
    print("Recommendation: Verify all updater scripts exist")
else:
    print("[SUCCESS] All steps completed successfully")
    print("Next: Proceed to Step 9 (Feature Engineering)")

print("=" * 80)

# Save log
import pandas as pd
df_log = pd.DataFrame(results)
df_log['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
log_path = os.path.join(BASE_PATH, r'#_master_database\DAILY_PIPELINE_LOG.csv')

if os.path.exists(log_path):
    df_existing = pd.read_csv(log_path)
    df_log = pd.concat([df_existing, df_log], ignore_index=True)

df_log.to_csv(log_path, index=False)
print(f"\nLog saved: {log_path}")
