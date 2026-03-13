#!/usr/bin/env python3
"""
Enhanced Treasury Yields Fetcher
Improved version of 00_03_fetch_treasuryyields_yahoo.py with:
- Better error handling and validation
- Progress tracking and logging
- Flexible date ranges
- Backup functionality
- Data quality checks

OUTPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_03_fetch_treasuries\\00_03_treasuryyields_fred.csv
"""

import pandas as pd
from pandas_datareader.data import DataReader
from datetime import datetime, timedelta
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TreasuryYieldsFetcher:
    """Enhanced treasury yields data fetcher"""
    
    def __init__(self):
        # Hardcoded output file path
        self.csv_path = CONFIG_BASE_PATH  # Set in config.py
        
        # Create backup path in same directory
        backup_dir = os.path.dirname(self.csv_path)
        self.backup_path = os.path.join(backup_dir, f"00_03_treasuryyields_fred_backup_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # FRED series mapping
        self.fred_series = {
            "DGS3MO": "Treasury_3M",    # 3-Month Treasury
            "DGS6MO": "Treasury_6M",    # 6-Month Treasury
            "DGS1": "Treasury_1Y",      # 1-Year Treasury
            "DGS2": "Treasury_2Y",      # 2-Year Treasury
            "DGS5": "Treasury_5Y",      # 5-Year Treasury
            "DGS10": "Treasury_10Y",    # 10-Year Treasury
            "DGS30": "Treasury_30Y",    # 30-Year Treasury
            "DFII10": "TIPS_10Y",       # 10-Year TIPS
            "T5YIE": "Breakeven_5Y",    # 5-Year Breakeven Inflation
            "T10YIE": "Breakeven_10Y"   # 10-Year Breakeven Inflation
        }
        
        print("🏦 Enhanced Treasury Yields Fetcher Initialized")
        print(f" Tracking {len(self.fred_series)} yield series")
        print(f"💾 Output file: {self.csv_path}")
        print(f"💾 Backup location: {backup_dir}")
    
    def load_existing_data(self):
        """Load existing treasury yields data"""
        
        if os.path.exists(self.csv_path):
            print(f"📂 Loading existing data from {self.csv_path}...")
            df = pd.read_csv(self.csv_path, parse_dates=["Date"])
            print(f"   ✅ Loaded {len(df):,} records")
            print(f"   📅 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            return df
        else:
            print(f"📂 No existing data found, starting fresh...")
            return pd.DataFrame()
    
    def fetch_fred_data(self, start_date, end_date, series_list=None):
        """Fetch data from FRED with error handling"""
        
        if series_list is None:
            series_list = list(self.fred_series.keys())
        
        print(f"🔄 Fetching FRED data from {start_date} to {end_date}...")
        print(f"   📊 Series: {len(series_list)} yield curves")
        
        successful_data = []
        failed_series = []
        
        for i, series in enumerate(series_list):
            try:
                print(f"   📈 Fetching {series} ({self.fred_series.get(series, series)}) [{i+1}/{len(series_list)}]")
                
                # Fetch individual series to handle errors better
                data = DataReader(series, "fred", start_date, end_date)
                
                if not data.empty:
                    data.columns = [self.fred_series.get(series, series)]
                    successful_data.append(data)
                    print(f"      ✅ Success: {len(data)} records")
                else:
                    failed_series.append(f"{series}: No data returned")
                    print(f"      ⚠️ No data for {series}")
                
            except Exception as e:
                failed_series.append(f"{series}: {str(e)}")
                print(f"      ❌ Error fetching {series}: {str(e)}")
                continue
        
        # Combine successful data
        if successful_data:
            print(f"🔗 Combining {len(successful_data)} successful series...")
            combined_df = pd.concat(successful_data, axis=1, sort=True)
            combined_df.reset_index(inplace=True)
            combined_df.rename(columns={"DATE": "Date"}, inplace=True)
            
            print(f"   ✅ Combined data: {len(combined_df)} records")
            print(f"   📊 Columns: {list(combined_df.columns)}")
            
            if failed_series:
                print(f"   ⚠️ Failed series ({len(failed_series)}):")
                for failure in failed_series:
                    print(f"      - {failure}")
            
            return combined_df
        else:
            print(f"❌ No data could be fetched!")
            return pd.DataFrame()
    
    def validate_data_quality(self, df):
        """Validate and check data quality"""
        
        print(f"🔍 Validating data quality...")
        
        if df.empty:
            print(f"   ❌ Empty dataset!")
            return False
        
        # Check for reasonable yield ranges
        quality_issues = []
        
        for col in df.columns:
            if col != 'Date' and 'Treasury' in col:
                series_data = df[col].dropna()
                
                if series_data.empty:
                    quality_issues.append(f"{col}: No valid data")
                    continue
                
                # Check for reasonable ranges (yields should be 0-20%)
                min_val = series_data.min()
                max_val = series_data.max()
                
                if min_val < -5 or max_val > 25:
                    quality_issues.append(f"{col}: Extreme values ({min_val:.2f}% to {max_val:.2f}%)")
                
                # Check for too many zeros
                zero_pct = (series_data == 0).sum() / len(series_data) * 100
                if zero_pct > 10:
                    quality_issues.append(f"{col}: {zero_pct:.1f}% zero values")
        
        if quality_issues:
            print(f"   ⚠️ Quality issues found:")
            for issue in quality_issues:
                print(f"      - {issue}")
        else:
            print(f"   ✅ Data quality looks good")
        
        return len(quality_issues) == 0
    
    def fill_missing_dates(self, df):
        """Fill missing dates and forward-fill values"""
        
        if df.empty:
            return df
        
        print(f"📅 Filling missing dates...")
        
        # Create complete date range
        start_date = df['Date'].min()
        end_date = max(df['Date'].max(), datetime.now())
        
        all_dates = pd.date_range(start_date, end_date, freq="D")
        
        print(f"   📊 Original: {len(df)} records")
        print(f"   📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Reindex to fill missing dates
        df_filled = df.set_index("Date").reindex(all_dates).rename_axis("Date").reset_index()
        
        # Forward-fill missing values (treasury yields change slowly)
        df_filled.ffill(inplace=True)
        
        print(f"   ✅ After filling: {len(df_filled)} records")
        print(f"   📈 Added: {len(df_filled) - len(df)} missing dates")
        
        return df_filled
    
    def create_summary_stats(self, df):
        """Create summary statistics for the data"""
        
        print(f"\n📊 Treasury Yields Summary Statistics:")
        print("=" * 50)
        
        if df.empty:
            print(f"   ❌ No data to summarize")
            return
        
        # Current levels (latest date)
        latest_date = df['Date'].max()
        latest_data = df[df['Date'] == latest_date].iloc[0]
        
        print(f"📅 Latest Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"📈 Current Yield Levels:")
        
        for col in df.columns:
            if col != 'Date' and not latest_data[col] != latest_data[col]:  # Not NaN
                value = latest_data[col]
                print(f"   {col}: {value:.2f}%")
        
        # Yield curve analysis
        treasury_cols = [col for col in df.columns if 'Treasury' in col and col != 'Date']
        if len(treasury_cols) >= 2:
            latest_treasury = latest_data[treasury_cols].dropna()
            if len(latest_treasury) >= 2:
                min_yield = latest_treasury.min()
                max_yield = latest_treasury.max()
                spread = max_yield - min_yield
                
                print(f"\n📊 Yield Curve Analysis:")
                print(f"   Range: {min_yield:.2f}% to {max_yield:.2f}%")
                print(f"   Spread: {spread:.2f}%")
                
                # Yield curve shape
                if len(latest_treasury) >= 3:
                    if latest_treasury.iloc[-1] > latest_treasury.iloc[0]:
                        curve_shape = "Normal (Upward Sloping)"
                    elif latest_treasury.iloc[-1] < latest_treasury.iloc[0]:
                        curve_shape = "Inverted (Downward Sloping)"
                    else:
                        curve_shape = "Flat"
                    
                    print(f"   Shape: {curve_shape}")
    
    def update_treasury_data(self, days_back=30, full_refresh=False):
        """Main update function"""
        
        print("🚀 Treasury Yields Data Update")
        print(f"⏰ Start time: {datetime.now()}")
        print("=" * 50)
        
        try:
            # Load existing data
            existing_df = self.load_existing_data()
            
            # Determine date range for update
            if full_refresh or existing_df.empty:
                # Full refresh - get 15 years of data
                start_date = datetime.now() - timedelta(days=15*365)
                print(f"🔄 Full refresh mode: fetching {15} years of data")
            else:
                # Incremental update
                latest_date = existing_df['Date'].max()
                start_date = latest_date - timedelta(days=days_back)  # Overlap for validation
                print(f"🔄 Incremental update: from {start_date.strftime('%Y-%m-%d')}")
            
            end_date = datetime.now()
            
            # Fetch new data
            new_df = self.fetch_fred_data(start_date, end_date)
            
            if new_df.empty:
                print(f"ℹ️ No new data available")
                return existing_df
            
            # Validate data quality
            is_valid = self.validate_data_quality(new_df)
            if not is_valid:
                print(f"⚠️ Data quality issues detected, but proceeding...")
            
            # Merge with existing data
            if not existing_df.empty:
                print(f"🔗 Merging with existing data...")
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Remove duplicates (keep latest)
                before_count = len(combined_df)
                combined_df.drop_duplicates(subset="Date", keep="last", inplace=True)
                after_count = len(combined_df)
                
                print(f"   📊 Before merge: {len(existing_df):,} + {len(new_df):,} records")
                print(f"   📊 After dedup: {after_count:,} records")
                print(f"   ✅ Net added: {after_count - len(existing_df):,} records")
            else:
                combined_df = new_df
                print(f"   📊 New dataset: {len(combined_df):,} records")
            
            # Sort by date
            combined_df.sort_values("Date", inplace=True)
            
            # Fill missing dates
            final_df = self.fill_missing_dates(combined_df)
            
            # Create backup of existing file
            if os.path.exists(self.csv_path):
                existing_df.to_csv(self.backup_path, index=False)
                print(f"💾 Created backup: {self.backup_path}")
            
            # Save updated data
            final_df.to_csv(self.csv_path, index=False)
            print(f"💾 Saved updated data: {self.csv_path}")
            
            # Generate summary
            self.create_summary_stats(final_df)
            
            print(f"\n🎉 Treasury yields update complete!")
            print(f"📁 Main file: {self.csv_path}")
            print(f"📁 Backup: {self.backup_path}")
            
            return final_df
            
        except Exception as e:
            print(f"❌ Error in treasury yields update: {str(e)}")
            raise

def main():
    """Main execution function"""
    
    fetcher = TreasuryYieldsFetcher()
    
    # Default: incremental update (30 days)
    # For full refresh, use: fetcher.update_treasury_data(full_refresh=True)
    updated_df = fetcher.update_treasury_data(days_back=30)
    
    if updated_df is not None and not updated_df.empty:
        print(f"✅ Success! Updated treasury yields data with {len(updated_df):,} total records")
    else:
        print(f"❌ Failed to update treasury yields data")

if __name__ == "__main__":
    main()