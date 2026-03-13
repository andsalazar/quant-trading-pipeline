"""
Step 8: Build Normalized Database (quant_trading_v2.db)
========================================================

Purpose:
- Create fresh SQLite database with normalized schema
- Load all 7 data sources into separate tables
- Preserve full historical data (market: 2015-2025, treasury: 2010-2025)
- Enable LEFT JOIN strategy for feature engineering

Strategy:
- Market data provides "spine" (2,618 trading days, 552 symbols)
- Other datasets join where dates overlap
- Missing values = NULL (natural SQL handling)
- Normalized storage in Step 8 → Denormalized ML features in Step 9

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

# Data source files (12 files optimized for database)
DATA_SOURCES = {
    # Core reference data
    'ticker_universe': os.path.join(DATA_PATH, r'#_00_ticker_universe\00_01_ticker_universe.csv'),
    
    # Market data (2 files: ticker-level + market aggregates)
    'market_data': os.path.join(DATA_PATH, r'#_01_fetch_market\00_01_market_enhanced_long.csv'),
    'market_features': os.path.join(DATA_PATH, r'#_01_fetch_market\00_01_market_features.csv'),
    
    # Macro data (market-level)
    'currency_data': os.path.join(DATA_PATH, r'#_02_fetch_currencies\00_02_currency_enhanced.csv'),
    'treasury_data': os.path.join(DATA_PATH, r'#_03_fetch_treasuries\00_03_treasuryyields_fred.csv'),
    
    # Futures/Options (market features, not ticker-level)
    'futures_features': os.path.join(DATA_PATH, r'#_04_fetch_futures\00_04_futures_market_features.csv'),
    'options_features': os.path.join(DATA_PATH, r'#_05_fetch_options\00_05_options_market_features.csv'),
    
    # Events (wide format with flags)
    'events_data': os.path.join(DATA_PATH, r'#_06_fetch_events\00_06_events_wide_optimized.csv'),
    
    # Sentiment (2 files: ticker-level + market features)
    'sentiment_data': os.path.join(DATA_PATH, r'#_07_news_sentiment\00_07_sentiment_long.csv'),
    'sentiment_features': os.path.join(DATA_PATH, r'#_07_news_sentiment\00_07_sentiment_wide_optimized.csv'),
}

print("=" * 80)
print("STEP 8: BUILD NORMALIZED DATABASE")
print("=" * 80)
print(f"Database: {DB_PATH}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# STEP 1: CREATE DATABASE & SCHEMA
# ============================================================================

print("\n[1/4] Creating database schema...")

# Remove old database if exists
if os.path.exists(DB_PATH):
    try:
        os.remove(DB_PATH)
        print(f"  ✓ Removed old database")
    except PermissionError:
        print(f"  ⚠ Warning: Cannot remove existing database (file in use)")
        print(f"  → Will overwrite tables instead")

# Create connection
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Table 1: ticker_universe
cursor.execute("""
CREATE TABLE ticker_universe (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    active INTEGER DEFAULT 1
)
""")
print("  ✓ Created table: ticker_universe")

# Table 2: market_data (TICKER-LEVEL, DAILY) - dynamic schema
cursor.execute("""
CREATE TABLE market_data (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    PRIMARY KEY (date, symbol)
)
""")
print("  ✓ Created table: market_data (schema TBD)")

# Table 3: currency_data (MARKET-LEVEL, DAILY) - dynamic schema
cursor.execute("""
CREATE TABLE currency_data (
    date TEXT PRIMARY KEY
)
""")
print("  ✓ Created table: currency_data (schema TBD)")

# Table 4: treasury_data (MARKET-LEVEL, DAILY) - dynamic schema
cursor.execute("""
CREATE TABLE treasury_data (
    date TEXT PRIMARY KEY
)
""")
print("  ✓ Created table: treasury_data (schema TBD)")

# Table 5: market_features (MARKET-LEVEL, DAILY AGGREGATES)
cursor.execute("""
CREATE TABLE market_features (
    date TEXT PRIMARY KEY
)
""")
print("  ✓ Created table: market_features (schema TBD)")

# Table 6: futures_features (MARKET-LEVEL, DAILY)
cursor.execute("""
CREATE TABLE futures_features (
    date TEXT PRIMARY KEY
)
""")
print("  ✓ Created table: futures_features (schema TBD)")

# Table 7: options_features (MARKET-LEVEL, DAILY)
cursor.execute("""
CREATE TABLE options_features (
    date TEXT PRIMARY KEY
)
""")
print("  ✓ Created table: options_features (schema TBD)")

# Table 8: events_data (MARKET-LEVEL, DAILY FLAGS)
cursor.execute("""
CREATE TABLE events_data (
    date TEXT PRIMARY KEY
)
""")
print("  ✓ Created table: events_data (schema TBD)")

# Table 9: sentiment_data (TICKER-LEVEL, DAILY)
cursor.execute("""
CREATE TABLE sentiment_data (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    sentiment_mean REAL,
    sentiment_std REAL,
    sentiment_count INTEGER,
    sentiment_min REAL,
    sentiment_max REAL,
    news_volume INTEGER,
    sentiment_ma_3d REAL,
    sentiment_ma_7d REAL,
    sentiment_ma_30d REAL,
    sentiment_momentum_3d REAL,
    sentiment_momentum_7d REAL,
    sentiment_volatility_7d REAL,
    extreme_positive INTEGER,
    extreme_negative INTEGER,
    PRIMARY KEY (date, ticker)
)
""")
cursor.execute("CREATE INDEX idx_sentiment_date ON sentiment_data(date)")
cursor.execute("CREATE INDEX idx_sentiment_ticker ON sentiment_data(ticker)")
print("  ✓ Created table: sentiment_data (with indexes)")

# Table 10: sentiment_features (MARKET-LEVEL, DAILY)
cursor.execute("""
CREATE TABLE sentiment_features (
    date TEXT PRIMARY KEY
)
""")
print("  ✓ Created table: sentiment_features (schema TBD)")

conn.commit()
print(f"\n  ✓ Created 10 tables with indexes")

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

print("\n[2/4] Loading data into tables...")

load_log = []

# Helper function to load CSV → SQL
def load_table(table_name, csv_path, date_col='Date', print_progress=False):
    """Load CSV into SQL table with progress tracking"""
    if not os.path.exists(csv_path):
        print(f"  ⚠ File not found: {csv_path}")
        return
    
    start_time = datetime.now()
    df = pd.read_csv(csv_path)
    
    # Standardize date column name
    if date_col in df.columns and date_col != 'date':
        df.rename(columns={date_col: 'date'}, inplace=True)
    
    # Convert date to string format (YYYY-MM-DD)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Load to SQL
    df.to_sql(table_name, conn, if_exists='append', index=False)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"  ✓ Loaded {table_name}: {len(df):,} records in {elapsed:.1f}s")
    
    # Log details
    load_log.append({
        'table': table_name,
        'records': len(df),
        'elapsed_seconds': elapsed,
        'source_file': os.path.basename(csv_path)
    })

# Load ticker_universe
if os.path.exists(DATA_SOURCES['ticker_universe']):
    df_tickers = pd.read_csv(DATA_SOURCES['ticker_universe'])
    # Keep existing columns (already in correct format)
    cols_to_keep = ['symbol', 'company_name', 'sector', 'is_active', 'priority']
    df_tickers = df_tickers[cols_to_keep].copy()
    df_tickers.columns = ['symbol', 'name', 'sector', 'active', 'industry']
    df_tickers['market_cap'] = None
    # Reorder to match schema
    df_tickers = df_tickers[['symbol', 'name', 'sector', 'industry', 'market_cap', 'active']]
    df_tickers.to_sql('ticker_universe', conn, if_exists='append', index=False)
    print(f"  ✓ Loaded ticker_universe: {len(df_tickers):,} symbols")
    load_log.append({
        'table': 'ticker_universe',
        'records': len(df_tickers),
        'elapsed_seconds': 0,
        'source_file': '00_01_ticker_universe.csv'
    })

# Load market_data (LARGE TABLE - takes time, dynamic schema)
print("\n  Loading market_data (1.25M records, may take 30-60 seconds)...")
if os.path.exists(DATA_SOURCES['market_data']):
    start_time = datetime.now()
    df_market = pd.read_csv(DATA_SOURCES['market_data'])
    if 'Date' in df_market.columns:
        df_market.rename(columns={'Date': 'date', 'Symbol': 'symbol'}, inplace=True)
    df_market['date'] = pd.to_datetime(df_market['date']).dt.strftime('%Y-%m-%d')
    
    # Drop old table and recreate with full schema
    cursor.execute("DROP TABLE market_data")
    df_market.to_sql('market_data', conn, if_exists='replace', index=False)
    cursor.execute("CREATE INDEX idx_market_date ON market_data(date)")
    cursor.execute("CREATE INDEX idx_market_symbol ON market_data(symbol)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  ✓ Loaded market_data: {len(df_market):,} records in {elapsed:.1f}s (dynamic schema with indexes)")
    load_log.append({
        'table': 'market_data',
        'records': len(df_market),
        'elapsed_seconds': elapsed,
        'source_file': '00_01_market_enhanced_long.csv'
    })

# Load market_features (dynamic schema)
if os.path.exists(DATA_SOURCES['market_features']):
    df_market_feat = pd.read_csv(DATA_SOURCES['market_features'])
    if 'Date' in df_market_feat.columns:
        df_market_feat.rename(columns={'Date': 'date'}, inplace=True)
    df_market_feat['date'] = pd.to_datetime(df_market_feat['date']).dt.strftime('%Y-%m-%d')
    cursor.execute("DROP TABLE market_features")
    df_market_feat.to_sql('market_features', conn, if_exists='replace', index=False)
    print(f"  ✓ Loaded market_features: {len(df_market_feat):,} records (34 features)")
    load_log.append({
        'table': 'market_features',
        'records': len(df_market_feat),
        'elapsed_seconds': 0,
        'source_file': '00_01_market_features.csv'
    })

# Load currency_data (dynamic schema)
if os.path.exists(DATA_SOURCES['currency_data']):
    df_currency = pd.read_csv(DATA_SOURCES['currency_data'])
    if 'Date' in df_currency.columns:
        df_currency.rename(columns={'Date': 'date'}, inplace=True)
    df_currency['date'] = pd.to_datetime(df_currency['date']).dt.strftime('%Y-%m-%d')
    cursor.execute("DROP TABLE currency_data")
    df_currency.to_sql('currency_data', conn, if_exists='replace', index=False)
    print(f"  ✓ Loaded currency_data: {len(df_currency):,} records (dynamic schema)")
    load_log.append({
        'table': 'currency_data',
        'records': len(df_currency),
        'elapsed_seconds': 0,
        'source_file': '00_02_currency_enhanced.csv'
    })

# Load treasury_data (dynamic schema)
if os.path.exists(DATA_SOURCES['treasury_data']):
    df_treasury = pd.read_csv(DATA_SOURCES['treasury_data'])
    if 'Date' in df_treasury.columns:
        df_treasury.rename(columns={'Date': 'date'}, inplace=True)
    df_treasury['date'] = pd.to_datetime(df_treasury['date']).dt.strftime('%Y-%m-%d')
    cursor.execute("DROP TABLE treasury_data")
    df_treasury.to_sql('treasury_data', conn, if_exists='replace', index=False)
    print(f"  ✓ Loaded treasury_data: {len(df_treasury):,} records (dynamic schema)")
    load_log.append({
        'table': 'treasury_data',
        'records': len(df_treasury),
        'elapsed_seconds': 0,
        'source_file': '00_03_treasuryyields_fred.csv'
    })

# Load futures_features (dynamic schema)
if os.path.exists(DATA_SOURCES['futures_features']):
    df_futures = pd.read_csv(DATA_SOURCES['futures_features'])
    if 'Date' in df_futures.columns:
        df_futures.rename(columns={'Date': 'date'}, inplace=True)
    df_futures['date'] = pd.to_datetime(df_futures['date']).dt.strftime('%Y-%m-%d')
    cursor.execute("DROP TABLE futures_features")
    df_futures.to_sql('futures_features', conn, if_exists='replace', index=False)
    print(f"  ✓ Loaded futures_features: {len(df_futures):,} records (35 features)")
    load_log.append({
        'table': 'futures_features',
        'records': len(df_futures),
        'elapsed_seconds': 0,
        'source_file': '00_04_futures_market_features.csv'
    })

# Load options_features (dynamic schema)
if os.path.exists(DATA_SOURCES['options_features']):
    df_options = pd.read_csv(DATA_SOURCES['options_features'])
    if 'Date' in df_options.columns:
        df_options.rename(columns={'Date': 'date'}, inplace=True)
    df_options['date'] = pd.to_datetime(df_options['date']).dt.strftime('%Y-%m-%d')
    cursor.execute("DROP TABLE options_features")
    df_options.to_sql('options_features', conn, if_exists='replace', index=False)
    print(f"  ✓ Loaded options_features: {len(df_options):,} records (30 features)")
    load_log.append({
        'table': 'options_features',
        'records': len(df_options),
        'elapsed_seconds': 0,
        'source_file': '00_05_options_market_features.csv'
    })

# Load events_data (dynamic schema)
if os.path.exists(DATA_SOURCES['events_data']):
    df_events = pd.read_csv(DATA_SOURCES['events_data'])
    if 'Date' in df_events.columns:
        df_events.rename(columns={'Date': 'date'}, inplace=True)
    df_events['date'] = pd.to_datetime(df_events['date']).dt.strftime('%Y-%m-%d')
    cursor.execute("DROP TABLE events_data")
    df_events.to_sql('events_data', conn, if_exists='replace', index=False)
    print(f"  ✓ Loaded events_data: {len(df_events):,} records (21 features)")
    load_log.append({
        'table': 'events_data',
        'records': len(df_events),
        'elapsed_seconds': 0,
        'source_file': '00_06_events_wide_optimized.csv'
    })

# Load sentiment_data
load_table('sentiment_data', DATA_SOURCES['sentiment_data'])

# Load sentiment_features (dynamic schema)
if os.path.exists(DATA_SOURCES['sentiment_features']):
    df_sentiment_feat = pd.read_csv(DATA_SOURCES['sentiment_features'])
    if 'Date' in df_sentiment_feat.columns:
        df_sentiment_feat.rename(columns={'Date': 'date'}, inplace=True)
    df_sentiment_feat['date'] = pd.to_datetime(df_sentiment_feat['date']).dt.strftime('%Y-%m-%d')
    cursor.execute("DROP TABLE sentiment_features")
    df_sentiment_feat.to_sql('sentiment_features', conn, if_exists='replace', index=False)
    print(f"  ✓ Loaded sentiment_features: {len(df_sentiment_feat):,} records (37 features)")
    load_log.append({
        'table': 'sentiment_features',
        'records': len(df_sentiment_feat),
        'elapsed_seconds': 0,
        'source_file': '00_07_sentiment_wide_optimized.csv'
    })

conn.commit()

# ============================================================================
# STEP 3: VALIDATE DATA
# ============================================================================

print("\n[3/4] Validating database integrity...")

validation_results = []

# Get list of all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]

for table in tables:
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    row_count = cursor.fetchone()[0]
    
    # Get date range if 'date' column exists
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'date' in columns:
        cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
        date_min, date_max = cursor.fetchone()
        cursor.execute(f"SELECT COUNT(DISTINCT date) FROM {table}")
        unique_dates = cursor.fetchone()[0]
        print(f"  ✓ {table}: {row_count:,} records, {unique_dates:,} dates ({date_min} to {date_max})")
        validation_results.append({
            'table': table,
            'records': row_count,
            'unique_dates': unique_dates,
            'date_range': f"{date_min} to {date_max}"
        })
    else:
        print(f"  ✓ {table}: {row_count:,} records")
        validation_results.append({
            'table': table,
            'records': row_count,
            'unique_dates': None,
            'date_range': None
        })

# Check database file size
db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
print(f"\n  ✓ Database size: {db_size_mb:.2f} MB")

# ============================================================================
# STEP 4: SAVE LOAD LOG
# ============================================================================

print("\n[4/4] Saving load log...")

# Create load log DataFrame
df_load_log = pd.DataFrame(load_log)
log_path = os.path.join(BASE_PATH, r'#_master_database\DATABASE_LOAD_LOG.csv')
df_load_log.to_csv(log_path, index=False)
print(f"  ✓ Saved load log: {log_path}")

# Create validation log
df_validation = pd.DataFrame(validation_results)
validation_path = os.path.join(BASE_PATH, r'#_master_database\DATABASE_VALIDATION.csv')
df_validation.to_csv(validation_path, index=False)
print(f"  ✓ Saved validation log: {validation_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DATABASE BUILD COMPLETE")
print("=" * 80)
print(f"Database: {DB_PATH}")
print(f"Size: {db_size_mb:.2f} MB")
print(f"Tables: {len(tables)}")
print(f"Total Records: {sum([r['records'] for r in validation_results]):,}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nRECOMMENDATION:")
print("  - Review DATABASE_VALIDATION.csv for data quality checks")
print("  - Proceed to Step 9: Feature Engineering (denormalized ML features)")
print("  - Strategy: LEFT JOIN on market_data spine (2,618 trading days)")
print("=" * 80)

conn.close()
