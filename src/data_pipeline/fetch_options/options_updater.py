#!/usr/bin/env python3
"""
INTEGRATED OPTIONS DAILY UPDATER
Clean CSV-based approach following futures pattern

OUTPUT FILES (hardcoded):
1. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_05_fetch_options\\00_05_options_long.csv
   - Enhanced long format (detailed analysis)
2. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_05_fetch_options\\00_05_options_market_features.csv
   - Market features (database merge)
3. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_05_fetch_options\\00_05_options_wide_optimized.csv
   - Smart wide format (ML ready)

Strategy:
- Focus on liquid symbols for reliable options data
- At-the-money options for current sentiment
- Multiple expirations for term structure
- Put/Call ratios and volume for market sentiment
- Macro symbols (GLD, USO) for regime detection

Daily Update Workflow:
1. Single Polygon API approach (rate limit friendly)
2. Smart strike selection around current prices
3. Generate comprehensive options features
4. Output all formats for database integration
"""

import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import time
import os
import warnings
from dotenv import load_dotenv
import requests
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class IntegratedOptionsUpdater:
    """Clean options data fetcher following futures pattern"""
    
    def __init__(self):
        """Initialize with comprehensive options strategy (hardcoded paths)"""
        # Hardcoded output file paths
        base_path = CONFIG_BASE_PATH  # Set in config.py
        
        # Output files (parallel to futures structure)
        self.long_csv = os.path.join(base_path, "00_05_options_long.csv")
        self.market_features_csv = os.path.join(base_path, "00_05_options_market_features.csv")
        self.wide_optimized_csv = os.path.join(base_path, "00_05_options_wide_optimized.csv")
        self.backup_path = os.path.join(base_path, f"00_05_options_backup_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # API setup
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("❌ POLYGON_API_KEY not found in environment variables")
        self.client = RESTClient(self.api_key)
        
        # Liquid symbols for reliable options data
        self.symbols_config = {
            # High-volume ETFs (most liquid options)
            "SPY": {"name": "SPDR S&P 500", "category": "Equity ETF", "priority": "critical", "liquid": True},
            "QQQ": {"name": "Invesco NASDAQ", "category": "Equity ETF", "priority": "critical", "liquid": True},
            "IWM": {"name": "iShares Russell 2000", "category": "Equity ETF", "priority": "high", "liquid": True},
            
            # Mega-cap stocks (high options volume)
            "AAPL": {"name": "Apple Inc", "category": "Mega Cap", "priority": "critical", "liquid": True},
            "MSFT": {"name": "Microsoft Corp", "category": "Mega Cap", "priority": "critical", "liquid": True},
            "NVDA": {"name": "NVIDIA Corp", "category": "Mega Cap", "priority": "high", "liquid": True},
            "TSLA": {"name": "Tesla Inc", "category": "Mega Cap", "priority": "high", "liquid": True},
            
            # Macro/Sentiment symbols
            "GLD": {"name": "SPDR Gold ETF", "category": "Macro", "priority": "high", "liquid": True},
            "USO": {"name": "US Oil Fund", "category": "Macro", "priority": "medium", "liquid": False},
            
            # High-beta tech (volatility plays)
            "AMD": {"name": "Advanced Micro Devices", "category": "Tech", "priority": "medium", "liquid": True},
            "META": {"name": "Meta Platforms", "category": "Tech", "priority": "medium", "liquid": True}
        }
        
        # Load existing data
        self.existing_data = self.load_existing_data()
        
        print("🚀 Integrated Options Daily Updater")
        print(f"📊 Symbols: {len(self.symbols_config)} (liquid options focus)")
        liquid_count = sum(1 for s in self.symbols_config.values() if s['liquid'])
        print(f"   🎯 High liquidity: {liquid_count} symbols")
        print("💡 Single workflow → All outputs (Long, Market Features, Wide)")
        print(f"💾 Output 1: {os.path.basename(self.long_csv)}")
        print(f"💾 Output 2: {os.path.basename(self.market_features_csv)}")
        print(f"💾 Output 3: {os.path.basename(self.wide_optimized_csv)}")
    
    def load_existing_data(self):
        """Load existing options data for incremental updates"""
        if os.path.exists(self.long_csv):
            print(f"📂 Loading existing options data...")
            df = pd.read_csv(self.long_csv, parse_dates=['Date'])
            if not df.empty:
                latest_date = df['Date'].max()
                print(f"   ✅ Found {len(df):,} records, latest: {latest_date.strftime('%Y-%m-%d')}")
                
                # Create backup
                df.to_csv(self.backup_path, index=False)
                print(f"   💾 Backup created")
            return df
        else:
            print(f"📂 No existing options data - starting fresh")
            return pd.DataFrame()
    
    def get_update_date_range(self, force_full_history=False):
        """Determine date range for options data collection"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        if force_full_history:
            # Full historical backfill to match market data (2015-05-18)
            start_date = '2015-05-18'
            print(f"📅 FULL HISTORICAL BACKFILL: {start_date} to {end_date}")
            print(f"   ⚠️  Note: Options data may be sparse before ~2020 (expiration issue)")
            return start_date, end_date
        
        if not self.existing_data.empty:
            last_date = self.existing_data['Date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if start_date > end_date:
                print(f"📅 Options data is current (latest: {last_date.strftime('%Y-%m-%d')})")
                return None, None
            else:
                print(f"📅 Incremental update: {start_date} to {end_date}")
        else:
            # Initial load: get 60 days for options (shorter than futures due to expiry)
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            print(f"📅 Initial options load: {start_date} to {end_date}")
        
        return start_date, end_date
    
    def get_current_stock_price(self, symbol):
        """Get current stock price for strike selection"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            
            aggs = list(self.client.get_aggs(
                symbol, 1, "day", start_date, end_date
            ))
            
            if aggs:
                return aggs[-1].close
            return None
            
        except Exception as e:
            print(f"      ❌ Price error: {str(e)[:30]}...")
            return None
    
    def get_option_expirations(self, symbol):
        """Get near-term option expirations"""
        try:
            url = f"https://api.polygon.io/v3/reference/options/contracts"
            params = {
                'underlying_ticker': symbol,
                'limit': 1000,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'results' in data:
                expirations = set()
                for contract in data['results']:
                    if 'expiration_date' in contract:
                        expirations.add(contract['expiration_date'])
                
                # Convert and filter
                exp_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
                exp_dates.sort()
                
                # Focus on near-term (next 90 days)
                today = datetime.now().date()
                max_date = today + timedelta(days=90)
                
                valid_exps = [exp for exp in exp_dates if today < exp <= max_date]
                return valid_exps[:4]  # First 4 expirations
            
            return []
            
        except Exception as e:
            print(f"      ❌ Exp error: {str(e)[:30]}...")
            return []
    
    def get_atm_strikes(self, current_price, num_strikes=3):
        """Get at-the-money strikes (focused approach)"""
        if not current_price:
            return []
        
        # Determine strike increment based on price
        if current_price > 500:
            increment = 10
        elif current_price > 200:
            increment = 5
        elif current_price > 100:
            increment = 2.5
        elif current_price > 50:
            increment = 1
        else:
            increment = 0.5
        
        # ATM strike
        atm_strike = round(current_price / increment) * increment
        
        # Get strikes around ATM
        strikes = []
        for i in range(-num_strikes//2, num_strikes//2 + 1):
            strike = atm_strike + (i * increment)
            if strike > 0:
                strikes.append(strike)
        
        return sorted(strikes)
    
    def build_option_ticker(self, symbol, exp_date, strike, option_type):
        """Build Polygon option ticker format"""
        exp_str = exp_date.strftime('%y%m%d')
        strike_str = f"{int(strike * 1000):08d}"
        return f"O:{symbol}{exp_str}{option_type}{strike_str}"
    
    def fetch_symbol_options(self, symbol, config, start_date, end_date):
        """Fetch options data for a single symbol"""
        print(f"   🎯 {symbol} ({config['name'][:20]})...", end=" ")
        
        # Get current price
        current_price = self.get_current_stock_price(symbol)
        if not current_price:
            print("❌ No price")
            return pd.DataFrame()
        
        # Get expirations and strikes
        expirations = self.get_option_expirations(symbol)
        strikes = self.get_atm_strikes(current_price)
        
        if not expirations or not strikes:
            print("❌ No contracts")
            return pd.DataFrame()
        
        options_data = []
        
        # Focused approach: 2 expirations, 3 strikes, calls and puts
        for exp_date in expirations[:2]:
            for strike in strikes:
                for option_type in ['C', 'P']:
                    
                    ticker = self.build_option_ticker(symbol, exp_date, strike, option_type)
                    
                    try:
                        aggs = list(self.client.get_aggs(
                            ticker, 1, "day", start_date, end_date
                        ))
                        
                        for agg in aggs:
                            date = pd.to_datetime(agg.timestamp, unit='ms')  # Keep as datetime for consistency
                            
                            options_data.append({
                                'Date': date,
                                'Symbol': symbol,
                                'Category': config['category'],
                                'Priority': config['priority'],
                                'Option_Ticker': ticker,
                                'Underlying_Price': current_price,
                                'Strike_Price': strike,
                                'Expiration_Date': exp_date,
                                'Option_Type': option_type,
                                'Days_To_Expiry': (exp_date - date.date()).days,
                                'Moneyness': current_price / strike,
                                'Open': agg.open,
                                'High': agg.high,
                                'Low': agg.low,
                                'Close': agg.close,
                                'Volume': agg.volume,
                                'VWAP': getattr(agg, 'vwap', None),
                                'Transactions': getattr(agg, 'transactions', None)
                            })
                        
                        time.sleep(0.05)  # Light rate limiting
                        
                    except Exception as e:
                        continue  # Skip failed contracts
        
        if options_data:
            df = pd.DataFrame(options_data)
            print(f"✅ {len(df)} records")
            return df
        else:
            print("❌ No data")
            return pd.DataFrame()
    
    def fetch_all_options_data(self, start_date, end_date):
        """Fetch options data for all symbols"""
        print(f"🔄 Fetching options data (focused approach)...")
        print(f"   📊 Period: {start_date} to {end_date}")
        print(f"   🎯 Symbols: {len(self.symbols_config)}")
        
        all_data = []
        successful = 0
        
        for symbol, config in self.symbols_config.items():
            symbol_data = self.fetch_symbol_options(symbol, config, start_date, end_date)
            
            if not symbol_data.empty:
                all_data.append(symbol_data)
                successful += 1
            
            # Rate limiting between symbols
            time.sleep(0.2)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n✅ Options fetch complete: {len(combined_df):,} records from {successful} symbols")
            return combined_df
        else:
            print("❌ No options data collected")
            return pd.DataFrame()
    
    def calculate_options_features(self, df):
        """Calculate comprehensive options features"""
        print("🔧 Calculating options features...")
        
        # Add basic calculated fields
        df['ATM_Flag'] = (abs(df['Moneyness'] - 1) < 0.05).astype(int)
        df['Near_Term_Flag'] = (df['Days_To_Expiry'] <= 30).astype(int)
        df['Intrinsic_Value'] = np.where(
            df['Option_Type'] == 'C',
            np.maximum(df['Underlying_Price'] - df['Strike_Price'], 0),
            np.maximum(df['Strike_Price'] - df['Underlying_Price'], 0)
        )
        df['Time_Value'] = df['Close'] - df['Intrinsic_Value']
        
        print(f"   ✅ Enhanced long format with calculated fields")
        return df
    
    def create_options_market_features(self, long_df):
        """Create market-wide options features for database merge"""
        print("🏗️ Creating options market features (database ready)...")
        
        daily_features = []
        
        for date in sorted(long_df['Date'].unique()):
            date_data = long_df[long_df['Date'] == date]
            features = {'Date': date}
            
            # Process each symbol's options
            for symbol in ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GLD']:
                symbol_data = date_data[date_data['Symbol'] == symbol]
                
                if not symbol_data.empty:
                    # Separate calls and puts
                    calls = symbol_data[symbol_data['Option_Type'] == 'C']
                    puts = symbol_data[symbol_data['Option_Type'] == 'P']
                    
                    # Put/Call Volume Ratio
                    call_volume = calls['Volume'].sum()
                    put_volume = puts['Volume'].sum()
                    if call_volume > 0:
                        features[f'{symbol}_PC_Volume_Ratio'] = put_volume / call_volume
                    
                    # ATM Activity
                    atm_data = symbol_data[symbol_data['ATM_Flag'] == 1]
                    features[f'{symbol}_ATM_Volume'] = atm_data['Volume'].sum()
                    
                    # Near-term activity
                    near_term = symbol_data[symbol_data['Near_Term_Flag'] == 1]
                    features[f'{symbol}_Near_Term_Volume'] = near_term['Volume'].sum()
                    
                    # Average implied activity (using time value as proxy)
                    if symbol_data['Time_Value'].notna().any():
                        features[f'{symbol}_Avg_Time_Value'] = symbol_data['Time_Value'].mean()
            
            # Cross-symbol features
            # Market-wide P/C ratio
            all_calls = date_data[date_data['Option_Type'] == 'C']
            all_puts = date_data[date_data['Option_Type'] == 'P']
            total_call_vol = all_calls['Volume'].sum()
            total_put_vol = all_puts['Volume'].sum()
            
            if total_call_vol > 0:
                features['Market_PC_Ratio'] = total_put_vol / total_call_vol
                features['Risk_Sentiment'] = 1 if features['Market_PC_Ratio'] > 1.0 else 0
            
            # Macro sentiment (GLD fear gauge)
            gld_data = date_data[date_data['Symbol'] == 'GLD']
            if not gld_data.empty:
                gld_puts = gld_data[gld_data['Option_Type'] == 'P']
                features['Gold_Fear_Gauge'] = gld_puts['Volume'].sum() / 1000  # Scaled
            
            daily_features.append(features)
        
        market_df = pd.DataFrame(daily_features)
        
        # Rolling features for regime detection
        if len(market_df) > 10:
            market_df['Market_PC_MA_10'] = market_df['Market_PC_Ratio'].rolling(10).mean()
            market_df['Elevated_Fear'] = (market_df['Market_PC_Ratio'] > market_df['Market_PC_MA_10']).astype(int)
        
        print(f"   ✅ Market features: {len(market_df)} dates × {len(market_df.columns)} features")
        return market_df
    
    def create_options_wide_optimized(self, combined_long_df):
        """Create ML-optimized wide format from complete historical options data"""
        print("📊 Creating options wide optimized format (ML ready)...")
        
        # Use the COMPLETE long format data to create wide format
        print(f"   📊 Processing {len(combined_long_df):,} long format records...")
        
        daily_wide_features = []
        all_dates = sorted(combined_long_df['Date'].unique())
        
        print(f"   📅 Converting {len(all_dates):,} dates to wide format...")
        
        for i, date in enumerate(all_dates):
            if i % 100 == 0 and i > 0:
                print(f"      Progress: {i:,}/{len(all_dates):,} dates")
            
            date_data = combined_long_df[combined_long_df['Date'] == date]
            
            # Initialize daily record
            daily_record = {'Date': date}
            
            # Process each symbol's options
            for symbol in ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GLD']:
                symbol_data = date_data[date_data['Symbol'] == symbol]
                
                if not symbol_data.empty:
                    # Separate calls and puts
                    calls = symbol_data[symbol_data['Option_Type'] == 'C']
                    puts = symbol_data[symbol_data['Option_Type'] == 'P']
                    
                    # Put/Call Volume Ratio
                    call_volume = calls['Volume'].sum()
                    put_volume = puts['Volume'].sum()
                    if call_volume > 0:
                        daily_record[f'{symbol}_PC_Volume_Ratio'] = put_volume / call_volume
                    
                    # ATM Activity
                    atm_data = symbol_data[symbol_data['ATM_Flag'] == 1]
                    daily_record[f'{symbol}_ATM_Volume'] = atm_data['Volume'].sum()
                    
                    # Near-term activity
                    near_term = symbol_data[symbol_data['Near_Term_Flag'] == 1]
                    daily_record[f'{symbol}_Near_Term_Volume'] = near_term['Volume'].sum()
                    
                    # Average implied activity (using time value as proxy)
                    if symbol_data['Time_Value'].notna().any():
                        daily_record[f'{symbol}_Avg_Time_Value'] = symbol_data['Time_Value'].mean()
            
            # Cross-symbol features
            # Market-wide P/C ratio
            all_calls = date_data[date_data['Option_Type'] == 'C']
            all_puts = date_data[date_data['Option_Type'] == 'P']
            total_call_vol = all_calls['Volume'].sum()
            total_put_vol = all_puts['Volume'].sum()
            
            if total_call_vol > 0:
                daily_record['Market_PC_Ratio'] = total_put_vol / total_call_vol
                daily_record['Risk_Sentiment'] = 1 if daily_record['Market_PC_Ratio'] > 1.0 else 0
            
            # Macro sentiment (GLD fear gauge)
            gld_data = date_data[date_data['Symbol'] == 'GLD']
            if not gld_data.empty:
                gld_puts = gld_data[gld_data['Option_Type'] == 'P']
                daily_record['Gold_Fear_Gauge'] = gld_puts['Volume'].sum() / 1000  # Scaled
            
            daily_wide_features.append(daily_record)
        
        wide_df = pd.DataFrame(daily_wide_features)
        wide_df = wide_df.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✅ Wide format: {len(wide_df):,} dates × {len(wide_df.columns)} features")
        print(f"   📅 Date range: {wide_df['Date'].min()} to {wide_df['Date'].max()}")
        return wide_df
    
    def save_all_outputs(self, long_df, market_df):
        """Save all options output formats"""
        print("💾 Saving all options outputs...")
        
        # Combine with existing data
        if not self.existing_data.empty:
            # Ensure Date columns are same type
            long_df['Date'] = pd.to_datetime(long_df['Date'])
            self.existing_data['Date'] = pd.to_datetime(self.existing_data['Date'])
            
            combined_long = pd.concat([self.existing_data, long_df], ignore_index=True)
            combined_long = combined_long.drop_duplicates(
                subset=['Date', 'Symbol', 'Option_Ticker'], keep='last'
            )
        else:
            combined_long = long_df
            combined_long['Date'] = pd.to_datetime(combined_long['Date'])
        
        # Sort data
        combined_long = combined_long.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        market_df = market_df.sort_values('Date').reset_index(drop=True)
        
        # Create proper wide format from complete historical data
        wide_df = self.create_options_wide_optimized(combined_long)
        
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
        """Main daily options update workflow"""
        print("🚀 OPTIONS DAILY UPDATE - Integrated Workflow")
        print("=" * 60)
        
        try:
            # Check update date range
            start_date, end_date = self.get_update_date_range(force_full_history)
            if start_date is None:
                print("✅ No options update needed - data is current")
                return
            
            # Fetch options data
            raw_data = self.fetch_all_options_data(start_date, end_date)
            if raw_data.empty:
                print("📊 No new options data to process")
                return
            
            # Calculate features
            enhanced_long = self.calculate_options_features(raw_data)
            
            # Combine with existing data to get complete historical dataset
            if not self.existing_data.empty:
                enhanced_long['Date'] = pd.to_datetime(enhanced_long['Date'])
                self.existing_data['Date'] = pd.to_datetime(self.existing_data['Date'])
                
                combined_long = pd.concat([self.existing_data, enhanced_long], ignore_index=True)
                combined_long = combined_long.drop_duplicates(
                    subset=['Date', 'Symbol', 'Option_Ticker'], keep='last'
                )
                combined_long = combined_long.sort_values(['Symbol', 'Date']).reset_index(drop=True)
            else:
                combined_long = enhanced_long
                combined_long['Date'] = pd.to_datetime(combined_long['Date'])
                combined_long = combined_long.sort_values(['Symbol', 'Date']).reset_index(drop=True)
            
            print(f"   📊 Complete dataset: {len(combined_long):,} records across {len(combined_long['Date'].unique())} dates")
            
            # Create market features using COMPLETE historical data
            market_features = self.create_options_market_features(combined_long)
            
            # Save all outputs
            summary = self.save_all_outputs(enhanced_long, market_features)
            
            print(f"\n🎯 OPTIONS UPDATE COMPLETE!")
            print(f"   📈 Enhanced long format: {summary['long']:,} records")
            print(f"   🗄️ Market features (DB ready): {summary['market']:,} dates")
            print(f"   🤖 Wide optimized (ML ready): {summary['wide']:,} dates")
            print(f"   📅 Date range: {summary['date_range']}")
            print(f"\n✅ Ready for SQLite database integration!")
            
        except Exception as e:
            print(f"❌ Error in options update: {str(e)}")
            raise

def main():
    """
    Daily options update - single command
    
    Usage:
        Daily updates (incremental):
            python 00_05_options_daily_updater.py
        
        Full historical backfill (one-time):
            python 00_05_options_daily_updater.py --full-history
    
    Note: Options historical data may be sparse before ~2020 due to:
    - Most options have expired (no historical pricing)
    - Limited Polygon data coverage for older contracts
    - Focus on recent 1-2 years is typical for options analysis
    """
    import sys
    
    print("🎯 INTEGRATED OPTIONS DAILY UPDATER")
    print("   Focused approach → Liquid symbols, ATM strikes, near-term expirations")
    print("   Output: Long format + Market features for database integration")
    
    updater = IntegratedOptionsUpdater()
    
    # Check for full history flag
    if len(sys.argv) > 1 and sys.argv[1] == '--full-history':
        print("=" * 80)
        print("FULL HISTORICAL BACKFILL MODE")
        print("Attempting to fetch options data from 2015-05-18")
        print("⚠️  WARNING: Options data will be sparse/empty before ~2020")
        print("   (Most old contracts have expired, limited historical data)")
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