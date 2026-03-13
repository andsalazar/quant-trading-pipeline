#!/usr/bin/env python3
"""
Clean Currency Data Fetcher - Fixed Column Structure
Creates consistent column naming and data structure

OUTPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_02_fetch_currencies\\00_02_currency_polygon_clean.csv
"""

from polygon import RESTClient
import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

load_dotenv()

class CleanCurrencyFetcher:
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable is not set")
        
        self.client = RESTClient(self.api_key)
        
        # Hardcoded output file path
        self.csv_file = CONFIG_BASE_PATH  # Set in config.py
        
        # Define consistent currency pairs with clean names
        self.currency_pairs = {
            'EUR_USD': 'C:EURUSD',
            'GBP_USD': 'C:GBPUSD', 
            'USD_JPY': 'C:USDJPY',
            'USD_CHF': 'C:USDCHF',
            'USD_CAD': 'C:USDCAD',
            'AUD_USD': 'C:AUDUSD',
            'NZD_USD': 'C:NZDUSD',
            'EUR_GBP': 'C:EURGBP',
            'EUR_JPY': 'C:EURJPY',
            'GBP_JPY': 'C:GBPJPY'
        }
        
        print(f"🚀 Clean Currency Data Fetcher")
        print(f"   Pairs to fetch: {len(self.currency_pairs)}")
        print(f"   💾 Output file: {self.csv_file}")
    
    def load_existing_data(self):
        """Load existing data if available"""
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file, index_col=0, parse_dates=True)
            latest_date = df.index.max().date()
            print(f"📊 Existing data found: {len(df)} records")
            print(f"   Latest date: {latest_date}")
            return df, latest_date
        else:
            print("📝 No existing data found, starting fresh")
            return pd.DataFrame(), None
    
    def fetch_currency_data(self, pair_symbol, from_date, to_date):
        """Fetch data for a single currency pair"""
        try:
            aggs = self.client.get_aggs(
                ticker=pair_symbol,
                multiplier=1,
                timespan="day", 
                from_=from_date,
                to=to_date,
                limit=50000
            )
            
            data = []
            for agg in aggs:
                data.append({
                    'date': pd.Timestamp(agg.timestamp, unit='ms').date(),
                    'close': agg.close
                })
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df['close']
            else:
                return pd.Series(dtype=float)
                
        except Exception as e:
            print(f"   ❌ Error fetching {pair_symbol}: {e}")
            return pd.Series(dtype=float)
    
    def run_update(self, force_full_history=False):
        """Run the currency data update"""
        
        # Load existing data
        existing_df, latest_date = self.load_existing_data()
        
        # Determine date range 
        ny_tz = pytz.timezone('America/New_York')
        today = datetime.now(ny_tz).date()
        
        if force_full_history:
            # Full historical backfill to match market data (2015-05-18)
            from_date = datetime(2015, 5, 18).date()
            print(f"🔄 FULL HISTORICAL BACKFILL MODE")
        elif latest_date:
            from_date = latest_date - timedelta(days=5)  # 5 day overlap for safety
        else:
            from_date = today - timedelta(days=365)  # 1 year for new setup
        
        to_date = today
        
        print(f"🔄 Fetching data from {from_date} to {to_date}")
        
        # Fetch new data
        new_data = {}
        for pair_name, pair_symbol in self.currency_pairs.items():
            print(f"   📈 Fetching {pair_name} ({pair_symbol})")
            
            series = self.fetch_currency_data(pair_symbol, from_date, to_date)
            
            if not series.empty:
                new_data[pair_name] = series
                print(f"      ✅ Got {len(series)} records")
            else:
                print(f"      ⚠️  No data retrieved")
        
        if not new_data:
            print("❌ No new data retrieved")
            return
        
        # Combine new data
        new_df = pd.DataFrame(new_data)
        
        # Merge with existing data
        if not existing_df.empty and not force_full_history:
            # Only add truly new dates (convert latest_date to datetime for comparison)
            latest_datetime = pd.Timestamp(latest_date)
            new_dates = new_df.index[new_df.index > latest_datetime]
            if len(new_dates) > 0:
                new_df = new_df.loc[new_dates]
                combined_df = pd.concat([existing_df, new_df])
            else:
                print("✅ No new data to add")
                return
        else:
            # Full replacement for historical backfill
            combined_df = new_df
        
        # Sort and remove duplicates
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # Save to CSV
        combined_df.to_csv(self.csv_file)
        
        print(f"\n✅ Currency update complete!")
        print(f"   📊 Total records: {len(combined_df):,}")
        print(f"   📅 Date range: {combined_df.index.min().date()} to {combined_df.index.max().date()}")
        print(f"   💱 Currency pairs: {len(combined_df.columns)}")
        print(f"   💾 Saved to: {self.csv_file}")
        
        # Show sample of latest data
        print(f"\n📈 Latest data sample:")
        print(combined_df.tail(3).round(4))

def main():
    """
    Main execution function
    
    Usage:
        Daily updates (incremental):
            python currency_fetcher_clean.py
        
        Full historical backfill (one-time):
            python currency_fetcher_clean.py --full-history
    
    Daily mode:
        - Fetches only new data since last update (with 5-day overlap)
        - Appends to existing CSV file
        - Fast and efficient for daily operations
    
    Full history mode:
        - Fetches all data from 2015-05-18 to today
        - Replaces entire CSV file
        - Use only for initial setup or complete rebuild
    """
    fetcher = CleanCurrencyFetcher()
    
    # Check if we need full historical backfill
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--full-history':
        print("=" * 80)
        print("FULL HISTORICAL BACKFILL MODE")
        print("Fetching currency data from 2015-05-18 to match market data coverage")
        print("This will REPLACE all existing data")
        print("=" * 80)
        fetcher.run_update(force_full_history=True)
    else:
        print("=" * 80)
        print("INCREMENTAL UPDATE MODE (Daily Operations)")
        print("Fetching only new data since last update")
        print("=" * 80)
        fetcher.run_update(force_full_history=False)

if __name__ == "__main__":
    main()