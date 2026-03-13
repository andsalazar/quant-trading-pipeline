"""
Step 8b: Incremental Database Update (Daily Operations)
========================================================

Purpose:
- Append new data to existing database (NO full rebuild)
- Only process records newer than last database date
- Fast execution (<1 second for typical daily update)
- Safe to run multiple times (PRIMARY KEY prevents duplicates)

Usage:
- Run AFTER daily data fetchers (Steps 0-7)
- Automated daily execution via scheduler
- Initial setup: Use 00_08_build_normalized_database.py first

Strategy:
- Query MAX(date) from each table
- Filter CSV files to only new records
- Use if_exists='append' (not 'replace')
- Log updates for tracking

Created: October 16, 2025
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (hardcoded)
BASE_PATH = CONFIG_BASE_PATH  # Set in config.py
DB_PATH = os.path.join(BASE_PATH, r'#_master_database\quant_trading_v2.db')
DATA_PATH = os.path.join(BASE_PATH, r'#_fetch_data')

# Data source files (same as Step 8)
DATA_SOURCES = {
    'ticker_universe': os.path.join(DATA_PATH, r'#_00_ticker_universe\00_01_ticker_universe.csv'),
    'market_data': os.path.join(DATA_PATH, r'#_01_fetch_market\00_01_market_enhanced_long.csv'),
    'market_features': os.path.join(DATA_PATH, r'#_01_fetch_market\00_01_market_features.csv'),
    'currency_data': os.path.join(DATA_PATH, r'#_02_fetch_currencies\00_02_currency_enhanced.csv'),
    'treasury_data': os.path.join(DATA_PATH, r'#_03_fetch_treasuries\00_03_treasuryyields_fred.csv'),
    'futures_features': os.path.join(DATA_PATH, r'#_04_fetch_futures\00_04_futures_market_features.csv'),
    'options_features': os.path.join(DATA_PATH, r'#_05_fetch_options\00_05_options_market_features.csv'),
    'events_data': os.path.join(DATA_PATH, r'#_06_fetch_events\00_06_events_wide_optimized.csv'),
    'sentiment_data': os.path.join(DATA_PATH, r'#_07_news_sentiment\00_07_sentiment_long.csv'),
    'sentiment_features': os.path.join(DATA_PATH, r'#_07_news_sentiment\00_07_sentiment_wide_optimized.csv'),
}

print("=" * 80)
print("STEP 8b: INCREMENTAL DATABASE UPDATE (Daily)")
print("=" * 80)
print(f"Database: {DB_PATH}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# VALIDATE DATABASE EXISTS
# ============================================================================

if not os.path.exists(DB_PATH):
    print("\n[ERROR] Database not found!")
    print(f"  Expected: {DB_PATH}")
    print("\n  Action Required:")
    print("  1. Run 00_08_build_normalized_database.py first (initial setup)")
    print("  2. Then use this script for daily updates")
    print("=" * 80)
    exit(1)

# Connect to existing database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_last_date(table_name, date_column='date'):
    """Get the most recent date in a table"""
    try:
        cursor.execute(f"SELECT MAX({date_column}) FROM {table_name}")
        result = cursor.fetchone()[0]
        return result if result else '1900-01-01'  # Default to very old date if empty
    except Exception as e:
        print(f"  ⚠ Warning: Could not get last date from {table_name}: {e}")
        return '1900-01-01'

def update_table_incremental(table_name, csv_path, date_column='Date', symbol_column=None):
    """
    Append only new records to table
    
    Args:
        table_name: SQL table name
        csv_path: Path to CSV file
        date_column: Name of date column in CSV (will be renamed to 'date')
        symbol_column: Name of symbol column in CSV (for ticker-level tables)
    
    Returns:
        Number of new records added
    """
    if not os.path.exists(csv_path):
        print(f"  ⚠ File not found: {csv_path}")
        return 0
    
    # Get last date in database
    last_db_date = get_last_date(table_name)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    if date_column in df.columns and date_column != 'date':
        df.rename(columns={date_column: 'date'}, inplace=True)
    if symbol_column and symbol_column in df.columns and symbol_column != 'symbol':
        df.rename(columns={symbol_column: 'symbol'}, inplace=True)
    
    # Convert date to string format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Filter to only new records (date > last_db_date)
    df_new = df[df['date'] > last_db_date].copy()
    
    if len(df_new) == 0:
        return 0
    
    # Append to database
    try:
        df_new.to_sql(table_name, conn, if_exists='append', index=False)
        return len(df_new)
    except Exception as e:
        print(f"  ⚠ Error appending to {table_name}: {e}")
        return 0

def update_ticker_universe():
    """Update ticker_universe (handles adds/changes, not deletes)"""
    if not os.path.exists(DATA_SOURCES['ticker_universe']):
        return 0
    
    # Read CSV
    df_csv = pd.read_csv(DATA_SOURCES['ticker_universe'])
    cols_to_keep = ['symbol', 'company_name', 'sector', 'is_active', 'priority']
    df_csv = df_csv[cols_to_keep].copy()
    df_csv.columns = ['symbol', 'name', 'sector', 'active', 'industry']
    df_csv['market_cap'] = None
    df_csv = df_csv[['symbol', 'name', 'sector', 'industry', 'market_cap', 'active']]
    
    # Get existing symbols from database
    cursor.execute("SELECT symbol FROM ticker_universe")
    existing_symbols = set([row[0] for row in cursor.fetchall()])
    
    # Find new symbols
    df_new = df_csv[~df_csv['symbol'].isin(existing_symbols)]
    
    if len(df_new) > 0:
        df_new.to_sql('ticker_universe', conn, if_exists='append', index=False)
        return len(df_new)
    
    return 0

# ============================================================================
# UPDATE TABLES
# ============================================================================

print("\n[1/2] Checking for new data...")

update_log = []
start_time = datetime.now()

# 1. Update ticker_universe (new symbols only)
new_symbols = update_ticker_universe()
if new_symbols > 0:
    print(f"  ✓ ticker_universe: Added {new_symbols} new symbols")
    update_log.append({'table': 'ticker_universe', 'new_records': new_symbols})
else:
    print(f"  → ticker_universe: No new symbols")

# 2. Update market_data (LARGE TABLE)
print("  → Checking market_data (may take 5-10 seconds)...")
new_market = update_table_incremental('market_data', DATA_SOURCES['market_data'], 
                                       date_column='Date', symbol_column='Symbol')
if new_market > 0:
    print(f"  ✓ market_data: Added {new_market:,} new records")
    update_log.append({'table': 'market_data', 'new_records': new_market})
else:
    print(f"  → market_data: No new data")

# 3. Update market_features
new_market_feat = update_table_incremental('market_features', DATA_SOURCES['market_features'])
if new_market_feat > 0:
    print(f"  ✓ market_features: Added {new_market_feat} new records")
    update_log.append({'table': 'market_features', 'new_records': new_market_feat})
else:
    print(f"  → market_features: No new data")

# 4. Update currency_data
new_currency = update_table_incremental('currency_data', DATA_SOURCES['currency_data'])
if new_currency > 0:
    print(f"  ✓ currency_data: Added {new_currency} new records")
    update_log.append({'table': 'currency_data', 'new_records': new_currency})
else:
    print(f"  → currency_data: No new data")

# 5. Update treasury_data
new_treasury = update_table_incremental('treasury_data', DATA_SOURCES['treasury_data'])
if new_treasury > 0:
    print(f"  ✓ treasury_data: Added {new_treasury} new records")
    update_log.append({'table': 'treasury_data', 'new_records': new_treasury})
else:
    print(f"  → treasury_data: No new data")

# 6. Update futures_features
new_futures = update_table_incremental('futures_features', DATA_SOURCES['futures_features'])
if new_futures > 0:
    print(f"  ✓ futures_features: Added {new_futures} new records")
    update_log.append({'table': 'futures_features', 'new_records': new_futures})
else:
    print(f"  → futures_features: No new data")

# 7. Update options_features
new_options = update_table_incremental('options_features', DATA_SOURCES['options_features'])
if new_options > 0:
    print(f"  ✓ options_features: Added {new_options} new records")
    update_log.append({'table': 'options_features', 'new_records': new_options})
else:
    print(f"  → options_features: No new data")

# 8. Update events_data
new_events = update_table_incremental('events_data', DATA_SOURCES['events_data'])
if new_events > 0:
    print(f"  ✓ events_data: Added {new_events} new records")
    update_log.append({'table': 'events_data', 'new_records': new_events})
else:
    print(f"  → events_data: No new data")

# 9. Update sentiment_data (CSV has 'ticker' column matching DB schema - do NOT rename)
new_sentiment = update_table_incremental('sentiment_data', DATA_SOURCES['sentiment_data'], 
                                          date_column='Date')
if new_sentiment > 0:
    print(f"  ✓ sentiment_data: Added {new_sentiment:,} new records")
    update_log.append({'table': 'sentiment_data', 'new_records': new_sentiment})
else:
    print(f"  → sentiment_data: No new data")

# 10. Update sentiment_features
new_sentiment_feat = update_table_incremental('sentiment_features', DATA_SOURCES['sentiment_features'])
if new_sentiment_feat > 0:
    print(f"  ✓ sentiment_features: Added {new_sentiment_feat} new records")
    update_log.append({'table': 'sentiment_features', 'new_records': new_sentiment_feat})
else:
    print(f"  → sentiment_features: No new data")

conn.commit()

# ============================================================================
# VALIDATION & LOGGING
# ============================================================================

print("\n[2/2] Validating updates...")

# Check current database state
validation_results = []
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]

for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    row_count = cursor.fetchone()[0]
    
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'date' in columns:
        cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
        date_min, date_max = cursor.fetchone()
        cursor.execute(f"SELECT COUNT(DISTINCT date) FROM {table}")
        unique_dates = cursor.fetchone()[0]
        validation_results.append({
            'table': table,
            'total_records': row_count,
            'unique_dates': unique_dates,
            'date_range': f"{date_min} to {date_max}"
        })
    else:
        validation_results.append({
            'table': table,
            'total_records': row_count,
            'unique_dates': None,
            'date_range': None
        })

# Get database size
db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

# Save update log
if len(update_log) > 0:
    df_update_log = pd.DataFrame(update_log)
    df_update_log['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = os.path.join(BASE_PATH, r'#_master_database\DATABASE_UPDATE_LOG.csv')
    
    # Append to existing log (or create new)
    if os.path.exists(log_path):
        df_existing = pd.read_csv(log_path)
        df_update_log = pd.concat([df_existing, df_update_log], ignore_index=True)
    
    df_update_log.to_csv(log_path, index=False)
    print(f"  ✓ Saved update log: {log_path}")

# Save validation snapshot
df_validation = pd.DataFrame(validation_results)
validation_path = os.path.join(BASE_PATH, r'#_master_database\DATABASE_VALIDATION.csv')
df_validation.to_csv(validation_path, index=False)
print(f"  ✓ Updated validation log: {validation_path}")

# ============================================================================
# SUMMARY
# ============================================================================

elapsed = (datetime.now() - start_time).total_seconds()
total_new_records = sum([log['new_records'] for log in update_log])

print("\n" + "=" * 80)
print("INCREMENTAL UPDATE COMPLETE")
print("=" * 80)
print(f"Database: {DB_PATH}")
print(f"Size: {db_size_mb:.2f} MB")
print(f"Total New Records: {total_new_records:,}")
print(f"Tables Updated: {len(update_log)}")
print(f"Execution Time: {elapsed:.1f} seconds")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

if total_new_records > 0:
    print("\n[SUCCESS] Database updated with new data")
    print("\nUpdated Tables:")
    for log in update_log:
        print(f"  - {log['table']}: +{log['new_records']:,} records")
    print("\nRECOMMENDATION:")
    print("  - Proceed to Step 9: Update ML features for new dates")
    print("  - Run: python 00_09_update_features_daily.py")
else:
    print("\n[INFO] No new data found")
    print("  - All tables are up to date")
    print("  - Check if data fetchers (Steps 0-7) ran successfully")

print("=" * 80)

conn.close()
