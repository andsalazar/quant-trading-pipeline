#!/usr/bin/env python3
"""
INTEGRATED FUTURES DAILY UPDATER
Single script for efficient daily futures data updates

OUTPUT FILES (hardcoded):
1. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_04_fetch_futures\\00_04_futures_long.csv
   - Enhanced long format (detailed analysis)
2. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_04_fetch_futures\\00_04_futures_market_features.csv
   - Market features (database merge)
3. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_04_fetch_futures\\00_04_futures_wide_optimized.csv
   - Smart wide format (ML ready)

Daily Update Workflow:
1. Single Polygon API fetch (rate limit friendly)
2. Calculate all features once
3. Generate all three output formats
4. Ready for SQLite database integration

Usage:
- Daily: python 00_04_futures_daily_updater.py
- Integration: Use market_features.csv for database merge
"""

import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import time
import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class IntegratedFuturesUpdater:
    """Single script for all futures data needs - optimized for daily updates"""
    
    def __init__(self):
        """Initialize with all output paths (hardcoded)"""
        # Hardcoded output file paths
        base_path = CONFIG_BASE_PATH  # Set in config.py
        
        # Primary outputs
        self.long_csv = os.path.join(base_path, "00_04_futures_long.csv")
        self.market_features_csv = os.path.join(base_path, "00_04_futures_market_features.csv")
        self.wide_optimized_csv = os.path.join(base_path, "00_04_futures_wide_optimized.csv")
        
        # Backup
        self.backup_path = os.path.join(base_path, f"00_04_futures_backup_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # API setup
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("❌ POLYGON_API_KEY not found in environment variables")
        self.client = RESTClient(self.api_key)
        
        # Verified working symbols from Polygon
        self.symbols_config = {
            # Core futures (actual contracts)
            "ES": {"name": "E-mini S&P 500", "category": "Equity", "type": "futures", "priority": "critical"},
            "TY": {"name": "10-Year Treasury", "category": "Rates", "type": "futures", "priority": "critical"}, 
            "FV": {"name": "5-Year Treasury", "category": "Rates", "type": "futures", "priority": "high"},
            "CL": {"name": "Crude Oil WTI", "category": "Energy", "type": "futures", "priority": "critical"},
            "NG": {"name": "Natural Gas", "category": "Energy", "type": "futures", "priority": "medium"},
            "DX": {"name": "Dollar Index", "category": "Currency", "type": "futures", "priority": "high"},
            "SI": {"name": "Silver", "category": "Metals", "type": "futures", "priority": "low"},
            "HG": {"name": "Copper", "category": "Metals", "type": "futures", "priority": "medium"},
            "ZS": {"name": "Soybeans", "category": "Agricultural", "type": "futures", "priority": "low"},
            
            # ETF proxies (for missing contracts)
            "SPY": {"name": "S&P 500 ETF", "category": "Equity", "type": "etf", "priority": "high"},
            "QQQ": {"name": "NASDAQ ETF", "category": "Equity", "type": "etf", "priority": "high"},
            "TLT": {"name": "Long Treasury ETF", "category": "Rates", "type": "etf", "priority": "medium"},
            "GLD": {"name": "Gold ETF", "category": "Metals", "type": "etf", "priority": "high"},
            "VXX": {"name": "VIX ETF", "category": "Volatility", "type": "etf", "priority": "high"}
        }
        
        # Load existing data for incremental updates
        self.existing_data = self.load_existing_data()
        
        print("🚀 Integrated Futures Daily Updater")
        print(f"📊 Symbols: {len(self.symbols_config)} ({sum(1 for s in self.symbols_config.values() if s['type']=='futures')} futures + {sum(1 for s in self.symbols_config.values() if s['type']=='etf')} ETFs)")
        print("💡 Single fetch → All outputs (Long, Market Features, Wide Optimized)")
        print(f"💾 Output 1: {os.path.basename(self.long_csv)}")
        print(f"💾 Output 2: {os.path.basename(self.market_features_csv)}")
        print(f"💾 Output 3: {os.path.basename(self.wide_optimized_csv)}")
    
    def load_existing_data(self):
        """Load existing long format data for incremental updates"""
        if os.path.exists(self.long_csv):
            print(f"📂 Loading existing data for incremental update...")
            df = pd.read_csv(self.long_csv, parse_dates=['Date'])
            print(f"   ✅ Found {len(df):,} existing records")
            if not df.empty:
                latest_date = df['Date'].max()
                print(f"   📅 Latest date: {latest_date.strftime('%Y-%m-%d')}")
                
                # Create backup
                df.to_csv(self.backup_path, index=False)
                print(f"   💾 Backup created")
            return df
        else:
            print(f"📂 No existing data - starting fresh")
            return pd.DataFrame()
    
    def get_update_date_range(self, force_full_history=False):
        """Determine what dates we need to fetch"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        if force_full_history:
            # Full historical backfill to match market data (2015-05-18)
            start_date = '2015-05-18'
            print(f"📅 FULL HISTORICAL BACKFILL: {start_date} to {end_date}")
            print(f"   ⚠️  This will replace all existing data")
            return start_date, end_date
        
        if not self.existing_data.empty:
            last_date = self.existing_data['Date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if start_date > end_date:
                print(f"📅 Data is current (latest: {last_date.strftime('%Y-%m-%d')})")
                return None, None
            else:
                print(f"📅 Incremental update: {start_date} to {end_date}")
        else:
            # Initial load: get 2+ years for solid backtesting
            start_date = (datetime.now() - timedelta(days=760)).strftime('%Y-%m-%d')
            print(f"📅 Initial historical load: {start_date} to {end_date}")
        
        return start_date, end_date
    
    def fetch_all_futures_data(self, start_date, end_date):
        """Single efficient fetch for all symbols"""
        print(f"🔄 Fetching futures data (single API run)...")
        print(f"   📊 Period: {start_date} to {end_date}")
        print(f"   🎯 Symbols: {len(self.symbols_config)}")
        
        all_data = []
        successful = 0
        failed = 0
        
        for symbol, config in self.symbols_config.items():
            try:
                icon = "🎯" if config['type'] == 'futures' else "📊"
                print(f"   {icon} {symbol} ({config['name'][:25]})...", end=" ")
                
                # Fetch from Polygon
                aggs = self.client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=start_date,
                    to=end_date,
                    limit=50000
                )
                
                # Process aggregates
                records = []
                for agg in aggs:
                    record = {
                        'Date': pd.to_datetime(agg.timestamp, unit='ms'),  # Keep as datetime for consistency
                        'Symbol': symbol,
                        'Name': config['name'],
                        'Category': config['category'],
                        'Type': config['type'],
                        'Priority': config['priority'],
                        'Open': agg.open,
                        'High': agg.high,
                        'Low': agg.low,
                        'Close': agg.close,
                        'Volume': agg.volume,
                        'VWAP': getattr(agg, 'vwap', None),
                        'Transactions': getattr(agg, 'transactions', None)
                    }
                    records.append(record)
                
                all_data.extend(records)
                print(f"✅ {len(records)} bars")
                successful += 1
                
            except Exception as e:
                print(f"❌ {str(e)[:30]}...")
                failed += 1
            
            # Rate limiting
            time.sleep(0.12)
        
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"\n✅ Single fetch complete: {len(df):,} records from {successful} symbols")
            return df
        else:
            print("❌ No data fetched")
            return pd.DataFrame()
    
    def calculate_comprehensive_features(self, df):
        """Calculate all technical features in one pass"""
        print("🔧 Calculating comprehensive features...")
        
        feature_df = df.copy()
        feature_df = feature_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Technical features by symbol
        for symbol in feature_df['Symbol'].unique():
            mask = feature_df['Symbol'] == symbol
            symbol_data = feature_df[mask].copy()
            
            # Price features
            symbol_data['Return_1d'] = symbol_data['Close'].pct_change()
            symbol_data['Return_5d'] = symbol_data['Close'].pct_change(5)
            symbol_data['Return_20d'] = symbol_data['Close'].pct_change(20)
            
            # Volatility
            symbol_data['Volatility_5d'] = symbol_data['Return_1d'].rolling(5).std()
            symbol_data['Volatility_20d'] = symbol_data['Return_1d'].rolling(20).std()
            
            # Moving averages
            symbol_data['MA_5'] = symbol_data['Close'].rolling(5).mean()
            symbol_data['MA_20'] = symbol_data['Close'].rolling(20).mean()
            symbol_data['MA_50'] = symbol_data['Close'].rolling(50).mean()
            
            # Momentum indicators
            symbol_data['Price_MA_Ratio'] = symbol_data['Close'] / symbol_data['MA_20']
            symbol_data['MA_Slope_5d'] = (symbol_data['MA_20'] - symbol_data['MA_20'].shift(5)) / symbol_data['MA_20'].shift(5)
            
            # Volume indicators (where available)
            if symbol_data['Volume'].notna().any():
                symbol_data['Volume_MA_20'] = symbol_data['Volume'].rolling(20).mean()
                symbol_data['Volume_Ratio'] = symbol_data['Volume'] / symbol_data['Volume_MA_20']
            
            # Update main dataframe
            feature_df.loc[mask, symbol_data.columns] = symbol_data
        
        print(f"   ✅ Technical features calculated for all symbols")
        return feature_df
    
    def create_market_features_table(self, long_df):
        """Create database-ready market features table"""
        print("🏗️ Creating market features table (database ready)...")
        
        daily_features = []
        
        for date in sorted(long_df['Date'].unique()):
            date_data = long_df[long_df['Date'] == date]
            features = {'Date': date}
            
            # Core equity signals
            for symbol in ['ES', 'SPY', 'QQQ']:
                data = date_data[date_data['Symbol'] == symbol]
                if not data.empty:
                    row = data.iloc[0]
                    features[f'{symbol}_Close'] = row['Close']
                    features[f'{symbol}_Return_1d'] = row.get('Return_1d')
                    features[f'{symbol}_Volatility_5d'] = row.get('Volatility_5d')
            
            # Interest rate signals
            for symbol in ['TY', 'FV', 'TLT']:
                data = date_data[date_data['Symbol'] == symbol]
                if not data.empty:
                    row = data.iloc[0]
                    features[f'{symbol}_Close'] = row['Close']
                    features[f'{symbol}_Return_1d'] = row.get('Return_1d')
            
            # Yield curve spread
            if 'TY_Close' in features and 'FV_Close' in features:
                features['Yield_Curve_Spread'] = features['TY_Close'] - features['FV_Close']
            
            # Energy & commodity signals
            for symbol in ['CL', 'NG', 'GLD', 'DX', 'VXX', 'HG']:
                data = date_data[date_data['Symbol'] == symbol]
                if not data.empty:
                    row = data.iloc[0]
                    features[f'{symbol}_Close'] = row['Close']
                    features[f'{symbol}_Return_1d'] = row.get('Return_1d')
            
            daily_features.append(features)
        
        market_df = pd.DataFrame(daily_features)
        
        # Cross-asset composite features
        if len(market_df) > 20:  # Need history for rolling calculations
            # Risk regime detection
            if 'VXX_Close' in market_df.columns:
                market_df['VXX_MA_20'] = market_df['VXX_Close'].rolling(20).mean()
                market_df['Risk_Off_Binary'] = (market_df['VXX_Close'] > market_df['VXX_MA_20']).astype(int)
            
            # Yield curve momentum
            if 'Yield_Curve_Spread' in market_df.columns:
                market_df['Yield_Curve_Momentum'] = market_df['Yield_Curve_Spread'].diff()
                market_df['Yield_Curve_Steepening'] = (market_df['Yield_Curve_Momentum'] > 0).astype(int)
            
            # Cross-asset momentum breadth
            return_cols = [col for col in market_df.columns if '_Return_1d' in col]
            if len(return_cols) >= 3:
                positive_momentum = market_df[return_cols].apply(lambda x: (x > 0).sum(), axis=1)
                market_df['Momentum_Breadth'] = positive_momentum / len(return_cols)
                market_df['Broad_Risk_On'] = (market_df['Momentum_Breadth'] > 0.6).astype(int)
        
        print(f"   ✅ Market features: {len(market_df)} dates × {len(market_df.columns)} features")
        return market_df
    
    def create_wide_optimized_format(self, combined_long_df):
        """Create ML-optimized wide format from complete historical data"""
        print("📊 Creating wide optimized format (ML ready)...")
        
        # Use the COMPLETE long format data to create wide format
        # This ensures we preserve ALL historical data, not just recent updates
        print(f"   📊 Processing {len(combined_long_df):,} long format records...")
        
        daily_wide_features = []
        all_dates = sorted(combined_long_df['Date'].unique())
        
        print(f"   📅 Converting {len(all_dates):,} dates to wide format...")
        
        for i, date in enumerate(all_dates):
            if i % 500 == 0 and i > 0:
                print(f"      Progress: {i:,}/{len(all_dates):,} dates")
            
            date_data = combined_long_df[combined_long_df['Date'] == date]
            
            # Initialize daily record
            daily_record = {'Date': date}
            
            # EQUITY CLUSTER - Market directional signals
            for symbol in ['ES', 'SPY', 'QQQ']:
                symbol_data = date_data[date_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    row = symbol_data.iloc[0]
                    daily_record[f'{symbol}_Close'] = row['Close']
                    daily_record[f'{symbol}_Return_1d'] = row.get('Return_1d')
                    daily_record[f'{symbol}_Volatility_5d'] = row.get('Volatility_5d')
            
            # RATES CLUSTER - Yield curve and duration signals  
            for symbol in ['TY', 'FV', 'TLT']:
                symbol_data = date_data[date_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    row = symbol_data.iloc[0]
                    daily_record[f'{symbol}_Close'] = row['Close']
                    daily_record[f'{symbol}_Return_1d'] = row.get('Return_1d')
            
            # Calculate yield curve spread
            if 'TY_Close' in daily_record and 'FV_Close' in daily_record:
                ty_close = daily_record['TY_Close']
                fv_close = daily_record['FV_Close']
                if pd.notna(ty_close) and pd.notna(fv_close):
                    daily_record['Yield_Curve_Spread'] = ty_close - fv_close
            
            # ENERGY & COMMODITIES - Inflation and commodity signals
            for symbol in ['CL', 'NG', 'GLD', 'HG']:
                symbol_data = date_data[date_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    row = symbol_data.iloc[0]
                    daily_record[f'{symbol}_Close'] = row['Close']
                    daily_record[f'{symbol}_Return_1d'] = row.get('Return_1d')
            
            # CURRENCY & VOLATILITY - Dollar and risk indicators
            for symbol in ['DX', 'VXX']:
                symbol_data = date_data[date_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    row = symbol_data.iloc[0]
                    daily_record[f'{symbol}_Close'] = row['Close']
                    daily_record[f'{symbol}_Return_1d'] = row.get('Return_1d')
            
            # CROSS-ASSET RELATIONSHIPS
            # Calculate cross-asset ratios
            if 'ES_Close' in daily_record and 'TY_Close' in daily_record:
                es_close = daily_record['ES_Close']
                ty_close = daily_record['TY_Close']
                if pd.notna(es_close) and pd.notna(ty_close) and ty_close != 0:
                    daily_record['ES_TY_Ratio'] = es_close / ty_close
            
            if 'CL_Close' in daily_record and 'DX_Close' in daily_record:
                cl_close = daily_record['CL_Close']
                dx_close = daily_record['DX_Close']
                if pd.notna(cl_close) and pd.notna(dx_close) and dx_close != 0:
                    daily_record['CL_DX_Ratio'] = cl_close / dx_close
            
            daily_wide_features.append(daily_record)
        
        wide_df = pd.DataFrame(daily_wide_features)
        wide_df = wide_df.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✅ Wide format: {len(wide_df):,} dates × {len(wide_df.columns)} features")
        print(f"   📅 Date range: {wide_df['Date'].min()} to {wide_df['Date'].max()}")
        return wide_df
    
    def save_all_outputs(self, long_df, market_df, wide_df):
        """Save all three output formats"""
        print("💾 Saving all output formats...")
        
        # long_df is already the complete combined dataset
        combined_long = long_df.copy()
        combined_long['Date'] = pd.to_datetime(combined_long['Date'])
        
        # Sort all dataframes
        combined_long = combined_long.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        market_df = market_df.sort_values('Date').reset_index(drop=True)
        wide_df = wide_df.sort_values('Date').reset_index(drop=True)
        
        # Save all formats
        combined_long.to_csv(self.long_csv, index=False)
        market_df.to_csv(self.market_features_csv, index=False)
        wide_df.to_csv(self.wide_optimized_csv, index=False)
        
        print(f"   ✅ Long format: {len(combined_long):,} records → {os.path.basename(self.long_csv)}")
        print(f"   ✅ Market features: {len(market_df):,} dates → {os.path.basename(self.market_features_csv)}")
        print(f"   ✅ Wide optimized: {len(wide_df):,} dates → {os.path.basename(self.wide_optimized_csv)}")
        
        return {
            'long': len(combined_long),
            'market': len(market_df), 
            'wide': len(wide_df),
            'date_range': f"{combined_long['Date'].min()} to {combined_long['Date'].max()}"
        }
    
    def run_daily_update(self, force_full_history=False):
        """Main daily update workflow - single command"""
        print("🚀 FUTURES DAILY UPDATE - Integrated Workflow")
        print("=" * 60)
        
        try:
            # Check if update needed
            start_date, end_date = self.get_update_date_range(force_full_history)
            if start_date is None:
                print("✅ No update needed - data is current")
                return
            
            # Single API fetch for all symbols
            raw_data = self.fetch_all_futures_data(start_date, end_date)
            if raw_data.empty:
                print("📊 No new data to process")
                return
            
            # BUG FIX: Combine raw data with existing BEFORE calculating features
            # pct_change() needs the previous day's close which isn't available
            # in a small batch of new data — causing NaN for Return_1d, etc.
            if not self.existing_data.empty:
                raw_data['Date'] = pd.to_datetime(raw_data['Date'])
                self.existing_data['Date'] = pd.to_datetime(self.existing_data['Date'])
                
                # Keep raw OHLCV columns from new data, merge with existing
                combined_raw = pd.concat([self.existing_data, raw_data], ignore_index=True)
                combined_raw = combined_raw.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
                combined_raw = combined_raw.sort_values(['Symbol', 'Date']).reset_index(drop=True)
            else:
                combined_raw = raw_data
                combined_raw['Date'] = pd.to_datetime(combined_raw['Date'])
                combined_raw = combined_raw.sort_values(['Symbol', 'Date']).reset_index(drop=True)
            
            # Calculate features on COMPLETE dataset so pct_change sees prior days
            combined_long = self.calculate_comprehensive_features(combined_raw)
            
            print(f"   📊 Complete dataset: {len(combined_long):,} records across {len(combined_long['Date'].unique())} dates")
            
            # Generate all output formats using COMPLETE historical data
            market_features = self.create_market_features_table(combined_long)
            wide_optimized = self.create_wide_optimized_format(combined_long)
            
            # Save everything (pass combined_long as long data)
            summary = self.save_all_outputs(combined_long, market_features, wide_optimized)
            
            print(f"\n🎯 DAILY UPDATE COMPLETE!")
            print(f"   📈 Enhanced long format: {summary['long']:,} records")
            print(f"   🗄️ Market features (DB ready): {summary['market']:,} dates")
            print(f"   🤖 Wide optimized (ML ready): {summary['wide']:,} dates")
            print(f"   📅 Date range: {summary['date_range']}")
            print(f"\n✅ Ready for SQLite database integration!")
            
        except Exception as e:
            print(f"❌ Error in daily update: {str(e)}")
            raise

def main():
    """
    Daily futures update - single command
    
    Usage:
        Daily updates (incremental):
            python 00_04_futures_daily_updater.py
        
        Full historical backfill (one-time):
            python 00_04_futures_daily_updater.py --full-history
    
    Daily mode:
        - Fetches only new data since last update
        - Appends to existing CSV files
        - Fast and efficient for daily operations
    
    Full history mode:
        - Fetches all data from 2015-05-18 to today
        - Replaces entire CSV files
        - Use only for initial setup or complete rebuild
    """
    import sys
    
    print("🎯 INTEGRATED FUTURES DAILY UPDATER")
    print("   Single script → All outputs (Long, Market Features, Wide)")
    print("   Optimized for daily workflow and database integration")
    
    updater = IntegratedFuturesUpdater()
    
    # Check for full history flag
    if len(sys.argv) > 1 and sys.argv[1] == '--full-history':
        print("=" * 80)
        print("FULL HISTORICAL BACKFILL MODE")
        print("Fetching futures data from 2015-05-18 to match market data coverage")
        print("This will REPLACE all existing data")
        print("=" * 80)
        print()
        updater.run_daily_update(force_full_history=True)
    else:
        print("=" * 80)
        print("INCREMENTAL UPDATE MODE (Daily Operations)")
        print("Fetching only new data since last update")
        print("=" * 80)
        print()
        updater.run_daily_update(force_full_history=False)

if __name__ == "__main__":
    main()