#!/usr/bin/env python3
"""
INTEGRATED MACRO EVENTS DAILY UPDATER
Clean CSV-based approach following futures/options pattern

INPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_06_fetch_events\\00_06_macroevents_fomc.csv
- Historical macro events data (FOMC meetings, jobs reports, CPI, inflation data)

OUTPUT FILES (hardcoded):
1. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_06_fetch_events\\00_06_events_long.csv
   - Enhanced long format (detailed events with impact scores)
2. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_06_fetch_events\\00_06_events_market_features.csv
   - Market features (database merge)
3. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_06_fetch_events\\00_06_events_wide_optimized.csv
   - Event counts and flags with forward-looking indicators (ML ready)

Strategy:
- Economic calendar events (BLS, FOMC, etc.)
- Impact scoring for market sensitivity
- Forward-looking event flags (market anticipation)
- Rolling event density for regime detection

Daily Update Workflow:
1. Load and clean existing historical data
2. Generate market-wide event features
3. Create forward-looking event calendars
4. Output all formats for database integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class IntegratedEventsUpdater:
    """Clean macro events processor following futures/options pattern"""
    
    def __init__(self):
        """Initialize with comprehensive events strategy (hardcoded paths)"""
        # Hardcoded file paths
        base_path = CONFIG_BASE_PATH  # Set in config.py
        
        # Input file
        self.input_csv = os.path.join(base_path, "00_06_macroevents_fomc.csv")
        
        # Output files (parallel to futures/options structure)
        self.long_csv = os.path.join(base_path, "00_06_events_long.csv")
        self.market_features_csv = os.path.join(base_path, "00_06_events_market_features.csv")
        self.wide_optimized_csv = os.path.join(base_path, "00_06_events_wide_optimized.csv")
        self.backup_path = os.path.join(base_path, f"00_06_events_backup_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # Impact scoring configuration
        self.impact_weights = {
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        
        # Category priorities for market impact
        self.category_priorities = {
            'FOMC': 10,      # Highest market impact
            'Jobs': 9,       # Employment data critical
            'Inflation': 8,  # CPI/PPI major market movers
            'Wages': 6,      # Secondary employment indicator
            'Productivity': 5  # Lower frequency, medium impact
        }
        
        print("🚀 Integrated Macro Events Daily Updater")
        print("📊 Processing: Economic calendar events with impact scoring")
        print("💡 Single workflow → All outputs (Long, Market Features, Wide)")
        print(f"📂 Input file: {os.path.basename(self.input_csv)}")
        print(f"💾 Output 1: {os.path.basename(self.long_csv)}")
        print(f"💾 Output 2: {os.path.basename(self.market_features_csv)}")
        print(f"💾 Output 3: {os.path.basename(self.wide_optimized_csv)}")
    
    def load_and_validate_data(self):
        """Load existing macro events data with validation"""
        print(f"📂 Loading macro events data...")
        
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"❌ Events file not found: {self.input_csv}")
        
        df = pd.read_csv(self.input_csv, parse_dates=['date'])
        
        # Create backup
        df.to_csv(self.backup_path, index=False)
        print(f"   💾 Backup created")
        
        # Validate required columns
        required_cols = ['date', 'release_title', 'category', 'impact']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")
        
        # Basic validation
        print(f"   ✅ Loaded {len(df):,} events")
        print(f"   📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        # Show categories
        category_counts = df['category'].value_counts()
        print(f"   📋 Categories: {', '.join(category_counts.index.tolist())}")
        
        return df
    
    def enhance_events_data(self, df):
        """Add calculated fields and clean data"""
        print("🔧 Enhancing events data...")
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['date', 'release_title'], keep='first')
        removed = original_len - len(df)
        
        if removed > 0:
            print(f"   🧹 Removed {removed:,} duplicate events")
        
        # Add calculated fields
        df['impact_score'] = df['impact'].map(self.impact_weights)
        df['category_priority'] = df['category'].map(self.category_priorities).fillna(1)
        df['weighted_impact'] = df['impact_score'] * df['category_priority']
        
        # Add timing flags
        df['is_fomc'] = (df['category'] == 'FOMC').astype(int)
        df['is_jobs_report'] = (df['release_title'].str.contains('Employment Situation', case=False)).astype(int)
        df['is_cpi'] = (df['release_title'].str.contains('Consumer Price Index', case=False)).astype(int)
        df['is_high_impact'] = (df['impact'] == 'High').astype(int)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"   ✅ Enhanced events with calculated fields")
        return df
    
    def create_forward_looking_events(self, df, days_ahead=30):
        """Create forward-looking event flags for market anticipation"""
        print(f"🔮 Creating forward-looking event flags ({days_ahead} days)...")
        
        # Get all dates in range
        min_date = df['date'].min()
        max_date = df['date'].max() + timedelta(days=days_ahead)
        
        # Create daily date range
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        daily_df = pd.DataFrame({'Date': date_range})
        
        forward_features = []
        
        for date in daily_df['Date']:
            features = {'Date': date}
            
            # Look ahead for upcoming events
            future_events = df[
                (df['date'] > date) & 
                (df['date'] <= date + timedelta(days=days_ahead))
            ]
            
            if not future_events.empty:
                # Days to next high-impact event
                high_impact_events = future_events[future_events['is_high_impact'] == 1]
                if not high_impact_events.empty:
                    features['days_to_next_high_impact'] = (high_impact_events['date'].min() - date).days
                else:
                    features['days_to_next_high_impact'] = days_ahead + 1
                
                # Days to next FOMC
                fomc_events = future_events[future_events['is_fomc'] == 1]
                if not fomc_events.empty:
                    features['days_to_next_fomc'] = (fomc_events['date'].min() - date).days
                else:
                    features['days_to_next_fomc'] = days_ahead + 1
                
                # Days to next jobs report
                jobs_events = future_events[future_events['is_jobs_report'] == 1]
                if not jobs_events.empty:
                    features['days_to_next_jobs'] = (jobs_events['date'].min() - date).days
                else:
                    features['days_to_next_jobs'] = days_ahead + 1
                
                # Total upcoming impact score
                features['upcoming_impact_score'] = future_events['weighted_impact'].sum()
                
                # Event density (events in next 7 days)
                week_events = future_events[future_events['date'] <= date + timedelta(days=7)]
                features['events_next_7_days'] = len(week_events)
                features['high_impact_next_7_days'] = len(week_events[week_events['is_high_impact'] == 1])
                
            else:
                # No upcoming events
                features.update({
                    'days_to_next_high_impact': days_ahead + 1,
                    'days_to_next_fomc': days_ahead + 1,
                    'days_to_next_jobs': days_ahead + 1,
                    'upcoming_impact_score': 0,
                    'events_next_7_days': 0,
                    'high_impact_next_7_days': 0
                })
            
            forward_features.append(features)
        
        forward_df = pd.DataFrame(forward_features)
        print(f"   ✅ Forward-looking features: {len(forward_df):,} dates")
        return forward_df
    
    def create_events_market_features(self, df):
        """Create market-wide event features for database merge"""
        print("🏗️ Creating events market features (database ready)...")
        
        # Create daily event features
        daily_features = []
        
        # Get all trading days (business days only)
        min_date = df['date'].min()
        max_date = df['date'].max()
        trading_days = pd.bdate_range(start=min_date, end=max_date)
        
        for date in trading_days:
            features = {'Date': date}
            
            # Events on this specific date
            day_events = df[df['date'].dt.date == date.date()]
            
            if not day_events.empty:
                # Event counts by type
                features['total_events'] = len(day_events)
                features['high_impact_events'] = len(day_events[day_events['is_high_impact'] == 1])
                features['fomc_events'] = len(day_events[day_events['is_fomc'] == 1])
                features['jobs_events'] = len(day_events[day_events['is_jobs_report'] == 1])
                features['cpi_events'] = len(day_events[day_events['is_cpi'] == 1])
                
                # Impact scoring
                features['daily_impact_score'] = day_events['weighted_impact'].sum()
                features['max_event_priority'] = day_events['category_priority'].max()
                
                # Event flags
                features['is_event_day'] = 1
                features['is_major_event_day'] = 1 if features['high_impact_events'] > 0 else 0
                
            else:
                # No events this day
                features.update({
                    'total_events': 0,
                    'high_impact_events': 0,
                    'fomc_events': 0,
                    'jobs_events': 0,
                    'cpi_events': 0,
                    'daily_impact_score': 0,
                    'max_event_priority': 0,
                    'is_event_day': 0,
                    'is_major_event_day': 0
                })
            
            daily_features.append(features)
        
        market_df = pd.DataFrame(daily_features)
        
        # Add rolling features for regime detection
        if len(market_df) > 10:
            market_df['events_rolling_7d'] = market_df['total_events'].rolling(7, min_periods=1).sum()
            market_df['high_impact_rolling_7d'] = market_df['high_impact_events'].rolling(7, min_periods=1).sum()
            market_df['impact_score_ma_10d'] = market_df['daily_impact_score'].rolling(10, min_periods=1).mean()
            
            # Event intensity flags
            market_df['high_event_period'] = (market_df['events_rolling_7d'] > market_df['events_rolling_7d'].quantile(0.75)).astype(int)
            market_df['major_event_cluster'] = (market_df['high_impact_rolling_7d'] >= 2).astype(int)
        
        print(f"   ✅ Market features: {len(market_df)} dates × {len(market_df.columns)} features")
        return market_df
    
    def save_all_outputs(self, long_df, market_df, forward_df):
        """Save all events output formats"""
        print("💾 Saving all events outputs...")
        
        # Sort data
        long_df = long_df.sort_values('date').reset_index(drop=True)
        market_df = market_df.sort_values('Date').reset_index(drop=True)
        
        # Merge market features with forward-looking features
        wide_df = market_df.merge(forward_df, on='Date', how='left')
        wide_df = wide_df.sort_values('Date').reset_index(drop=True)
        
        # Save all formats
        long_df.to_csv(self.long_csv, index=False)
        market_df.to_csv(self.market_features_csv, index=False)
        wide_df.to_csv(self.wide_optimized_csv, index=False)
        
        print(f"   ✅ Long format: {len(long_df):,} events → {os.path.basename(self.long_csv)}")
        print(f"   ✅ Market features: {len(market_df):,} dates → {os.path.basename(self.market_features_csv)}")
        print(f"   ✅ Wide optimized: {len(wide_df):,} dates → {os.path.basename(self.wide_optimized_csv)}")
        
        return {
            'long': len(long_df),
            'market': len(market_df),
            'wide': len(wide_df),
            'date_range': f"{long_df['date'].min().date()} to {long_df['date'].max().date()}"
        }
    
    def run_daily_update(self):
        """Main daily events update workflow"""
        print("🚀 EVENTS DAILY UPDATE - Integrated Workflow")
        print("=" * 60)
        
        try:
            # Load and validate
            raw_data = self.load_and_validate_data()
            
            # Enhance with calculated fields
            enhanced_long = self.enhance_events_data(raw_data)
            
            # Create market features
            market_features = self.create_events_market_features(enhanced_long)
            
            # Create forward-looking features
            forward_features = self.create_forward_looking_events(enhanced_long)
            
            # Save all outputs
            summary = self.save_all_outputs(enhanced_long, market_features, forward_features)
            
            print(f"\n🎯 EVENTS UPDATE COMPLETE!")
            print(f"   📈 Enhanced long format: {summary['long']:,} events")
            print(f"   🗄️ Market features (DB ready): {summary['market']:,} dates")
            print(f"   🔮 Wide optimized (w/ forward-looking): {summary['wide']:,} dates")
            print(f"   📅 Date range: {summary['date_range']}")
            print(f"\n✅ Ready for SQLite database integration!")
            
        except Exception as e:
            print(f"❌ Error in events update: {str(e)}")
            raise

def main():
    """Daily events update - single command"""
    print("🎯 INTEGRATED EVENTS DAILY UPDATER")
    print("   Historical events → Market features + Forward-looking indicators")
    print("   Output: Long format + Market features + Event anticipation")
    print()
    
    updater = IntegratedEventsUpdater()
    updater.run_daily_update()

if __name__ == "__main__":
    main()