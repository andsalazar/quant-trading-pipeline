#!/usr/bin/env python3
"""
Daily Market Data Updater with Dynamic Ticker Management
Replaces hardcoded ticker lists with dynamic universe management.

INPUT/OUTPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_01_fetch_market\\00_01_market_polygon_converted.csv

Reads existing data from this file, fetches new data from Polygon API, and saves back to same file.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_dynamic_tickers():
    """Load active tickers from dynamic ticker universe CSV"""
    # Get the script directory and navigate to ticker universe location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    ticker_file = os.path.join(project_root, "#_Core_Project", "#_fetch_data", "#_00_ticker_universe", "00_01_ticker_universe.csv")
    
    if not os.path.exists(ticker_file):
        print(f"❌ Ticker universe file not found: {ticker_file}")
        print("   Expected location: #_Core_Project/#_fetch_data/#_00_ticker_universe/00_01_ticker_universe.csv")
        return {}
    
    df = pd.read_csv(ticker_file)
    active_tickers = df[df['is_active'] == True]
    
    # Create symbol mapping (symbol as both key and value for compatibility)
    symbols = {}
    for _, row in active_tickers.iterrows():
        symbol = row['symbol']
        # Use descriptive name as key where possible
        if symbol == 'SPY':
            symbols['SP500'] = symbol
        elif symbol == 'QQQ':
            symbols['NASDAQ'] = symbol
        elif symbol == 'IWM':
            symbols['IWM'] = symbol
        elif symbol == 'VTI':
            symbols['VTI'] = symbol
        elif symbol == 'VIXY':
            symbols['VIX_ETF'] = symbol
        elif symbol == 'GLD':
            symbols['Gold_ETF'] = symbol
        elif symbol == 'USO':
            symbols['Oil_ETF'] = symbol
        else:
            symbols[symbol] = symbol
    
    print(f"📊 Loaded {len(symbols)} active tickers from dynamic universe")
    return symbols

def main():
    """Main market data update process"""
    
    # API Setup
    API_KEY = os.getenv("POLYGON_API_KEY")
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable is not set")
    client = RESTClient(API_KEY)
    
    # Hardcoded input/output file path
    data_file = CONFIG_BASE_PATH  # Set in config.py
    
    # Load existing data
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, parse_dates=["Date"])
        latest_date = df["Date"].max().normalize().date()
        print(f"📅 Latest data date: {latest_date}")
        print(f"📂 Input file: {data_file}")
    else:
        print(f"📝 No existing data file found, will create {data_file}")
        print(f"📂 Output file: {data_file}")
        df = pd.DataFrame()
        latest_date = None
    
    # Define timezone and date range
    ny_tz = pytz.timezone('America/New_York')
    today_ny = datetime.now(ny_tz).date()
    
    # Check if update is needed
    if latest_date and latest_date >= today_ny:
        print(f"✅ Data is already updated to today's date ({today_ny}). No update needed.")
        return
    
    # Set date range with overlap to catch missing data
    if latest_date:
        from_date = latest_date - timedelta(days=10)
    else:
        from_date = today_ny - timedelta(days=365)  # Get 1 year of data for new installs
    to_date = today_ny
    
    print(f"🔄 Fetching data from Polygon from {from_date} to {to_date}")
    
    # Load dynamic tickers
    symbols = load_dynamic_tickers()
    if not symbols:
        print("❌ No tickers loaded, exiting...")
        return
    
    # Fetch data for all symbols
    all_rows = []
    total_symbols = len(symbols)
    
    for idx, (name, ticker) in enumerate(symbols.items(), 1):
        print(f"📈 Fetching: {ticker} ({name}) [{idx}/{total_symbols}]")
        
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=from_date,
                to=to_date,
                limit=50000
            )
            
            for agg in aggs:
                # Store in long format directly
                row = {
                    "Date": pd.to_datetime(agg.timestamp, unit='ms').date(),
                    "Symbol": ticker,
                    "Open": agg.open,
                    "High": agg.high,
                    "Low": agg.low,
                    "Close": agg.close,
                    "Volume": agg.volume,
                    "VWAP": getattr(agg, 'vwap', None),
                    "Transactions": getattr(agg, 'transactions', None)
                }
                all_rows.append(row)
                
        except Exception as e:
            print(f"⚠️  WARNING: Polygon request failed for {ticker}: {e}")
            continue
        
        # Rate limiting
        time.sleep(0.1)
    
    if not all_rows:
        print("❌ No data fetched, exiting...")
        return
    
    # Process new data
    print("🔄 Processing fetched data...")
    new_df = pd.DataFrame(all_rows)
    new_df["Date"] = pd.to_datetime(new_df["Date"])
    
    if latest_date:
        # Filter to only new data
        new_df = new_df[pd.to_datetime(new_df["Date"]).dt.date > latest_date]
        
        if new_df.empty:
            print("✅ No new data to add")
            return
        
        # Load existing long format data
        existing_df = pd.read_csv(data_file)
        existing_df["Date"] = pd.to_datetime(existing_df["Date"])
        
        # Merge with existing data
        print(f"🔄 Merging {len(new_df)} new records with existing data...")
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Handle overlapping data by taking the most recent values
        merged = merged.sort_values(["Date", "Symbol"])
        merged = merged.groupby(["Date", "Symbol"]).last().reset_index()
    else:
        merged = new_df
    
    # Save updated data in long format (hardcoded path)
    output_file = CONFIG_BASE_PATH  # Set in config.py
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    merged.to_csv(output_file, index=False)
    
    print(f"✅ Market data update complete!")
    print(f"   📊 Total records: {len(merged):,}")
    print(f"   📅 Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
    print(f"   🎯 Symbols covered: {merged['Symbol'].nunique()}")
    print(f"   💾 Output file: {output_file}")

if __name__ == "__main__":
    main()