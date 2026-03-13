#!/usr/bin/env python3
"""
INTEGRATED NEWS & SENTIMENT DAILY UPDATER
Consolidated pipeline: Headlines → Sentiment → Aggregation

INPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_07_news_sentiment\\00_08_headlines_polygon.csv.gz
- Compressed headlines from Polygon API (fetched by separate script)

OUTPUT FILES (hardcoded):
1. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_07_news_sentiment\\00_07_sentiment_long.csv
   - Ticker-level daily sentiment scores
2. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_07_news_sentiment\\00_07_sentiment_market_features.csv
   - Market-wide sentiment features (database merge)
3. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_07_news_sentiment\\00_07_sentiment_wide_optimized.csv
   - ML-ready wide format

Strategy:
- Process headlines in batches through OpenAI GPT-3.5-turbo
- Ticker-level sentiment tracking
- Market-wide aggregations (SPY, Tech sector, Safe haven)
- Sentiment momentum and breadth metrics
- Cost-optimized with caching and incremental updates

Daily Update Workflow:
1. Load existing sentiment scores (skip already processed)
2. Batch process new headlines through OpenAI
3. Aggregate to ticker-level daily scores
4. Create market-wide features (SPY, sectors, breadth)
5. Generate all output formats for database integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class IntegratedNewsSentimentUpdater:
    """Unified news sentiment processor following futures/options pattern"""
    
    def __init__(self):
        """Initialize with comprehensive sentiment strategy (hardcoded paths)"""
        # Hardcoded file paths
        base_path = CONFIG_BASE_PATH  # Set in config.py
        
        # Input file (from separate headlines fetcher)
        self.headlines_input = os.path.join(base_path, "00_08_headlines_polygon.csv.gz")
        
        # Intermediate file (raw sentiment scores)
        self.raw_sentiment_csv = os.path.join(base_path, "00_07_sentiment_raw.csv")
        
        # Output files (parallel to futures/options structure)
        self.long_csv = os.path.join(base_path, "00_07_sentiment_long.csv")
        self.market_features_csv = os.path.join(base_path, "00_07_sentiment_market_features.csv")
        self.wide_optimized_csv = os.path.join(base_path, "00_07_sentiment_wide_optimized.csv")
        self.backup_path = os.path.join(base_path, f"00_07_sentiment_backup_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # OpenAI setup
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.openai_key)
        
        # Sector definitions for market-wide features
        self.sector_tickers = {
            'Tech': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AMD', 'AVGO', 'ORCL'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB'],
            'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'OXY', 'VLO', 'PSX'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'DIS']
        }
        
        # Market indices
        self.market_indices = ['SPY', 'QQQ', 'IWM', 'DIA']
        
        # Safe haven / macro
        self.safe_haven = ['GLD', 'TLT', 'SHY', 'UUP']
        
        print("🚀 Integrated News & Sentiment Daily Updater")
        print("📰 Pipeline: Headlines → Sentiment → Aggregation")
        print("💡 Single workflow → All outputs (Long, Market Features, Wide)")
        print(f"📂 Input: {os.path.basename(self.headlines_input)}")
        print(f"💾 Output 1: {os.path.basename(self.long_csv)}")
        print(f"💾 Output 2: {os.path.basename(self.market_features_csv)}")
        print(f"💾 Output 3: {os.path.basename(self.wide_optimized_csv)}")
    
    def load_headlines(self):
        """Load headlines from compressed file"""
        print(f"📰 Loading headlines...")
        
        if not os.path.exists(self.headlines_input):
            raise FileNotFoundError(f"❌ Headlines file not found: {self.headlines_input}")
        
        df = pd.read_csv(self.headlines_input, compression='gzip', parse_dates=['Date'])
        
        # Standardize date column
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Create unique key for deduplication
        df['unique_key'] = (
            df['Date'].dt.strftime('%Y-%m-%d') + '_' + 
            df['ticker'].astype(str) + '_' + 
            df['title'].astype(str).str.slice(0, 50)
        )
        
        # Create combined text for sentiment analysis
        df['text'] = (df['title'].fillna('') + ' ' + df['summary'].fillna('')).str.strip()
        
        print(f"   ✅ Loaded {len(df):,} headlines")
        print(f"   📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"   🎯 Unique tickers: {df['ticker'].nunique()}")
        
        return df
    
    def load_existing_sentiment(self):
        """Load existing sentiment scores to avoid reprocessing"""
        if os.path.exists(self.raw_sentiment_csv):
            print(f"📊 Loading existing sentiment scores...")
            df = pd.read_csv(self.raw_sentiment_csv, parse_dates=['Date'])
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Create backup
            df.to_csv(self.backup_path, index=False)
            print(f"   💾 Backup created")
            
            already_scored = set(df['unique_key'].dropna())
            print(f"   ✅ Found {len(df):,} existing scores ({len(already_scored):,} unique)")
            return df, already_scored
        else:
            print(f"📊 No existing sentiment data - starting fresh")
            return pd.DataFrame(), set()
    
    def score_sentiment_batch(self, headlines_batch):
        """Score sentiment for a batch of headlines using OpenAI"""
        
        results = []
        
        for idx, row in headlines_batch.iterrows():
            try:
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analyzer. Score the sentiment of news headlines on a scale from -1 (very negative) to +1 (very positive). Return ONLY a number between -1 and +1."},
                        {"role": "user", "content": f"Score this headline: {row['text'][:500]}"}  # Limit to 500 chars
                    ],
                    max_tokens=10,
                    temperature=0
                )
                
                # Parse sentiment score
                sentiment_str = response.choices[0].message.content.strip()
                try:
                    sentiment = float(sentiment_str)
                    sentiment = max(-1, min(1, sentiment))  # Clamp to [-1, 1]
                except:
                    sentiment = 0.0  # Default to neutral if parsing fails
                
                results.append({
                    'Date': row['Date'],
                    'ticker': row['ticker'],
                    'title': row['title'],
                    'text': row['text'],
                    'unique_key': row['unique_key'],
                    'sentiment': sentiment
                })
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"      ⚠️ Error scoring headline: {str(e)[:50]}...")
                results.append({
                    'Date': row['Date'],
                    'ticker': row['ticker'],
                    'title': row['title'],
                    'text': row['text'],
                    'unique_key': row['unique_key'],
                    'sentiment': 0.0  # Default to neutral on error
                })
        
        return pd.DataFrame(results)
    
    def process_sentiment_scoring(self, headlines_df, already_scored):
        """Process sentiment scoring with incremental updates"""
        
        # Filter to headlines that need scoring
        need_scoring = headlines_df[~headlines_df['unique_key'].isin(already_scored)].copy()
        
        if need_scoring.empty:
            print("✅ All headlines already scored - skipping sentiment analysis")
            return pd.DataFrame()
        
        print(f"🔄 Processing sentiment for {len(need_scoring):,} new headlines...")
        print(f"   💰 Estimated cost: ${len(need_scoring) * 0.0001:.2f} (at $0.0001/headline)")
        
        # Process in batches
        batch_size = 100
        all_results = []
        
        total_batches = (len(need_scoring) + batch_size - 1) // batch_size
        
        for i in range(0, len(need_scoring), batch_size):
            batch = need_scoring.iloc[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"   📊 Batch {batch_num}/{total_batches} ({len(batch)} headlines)...", end=" ")
            
            batch_results = self.score_sentiment_batch(batch)
            all_results.append(batch_results)
            
            print(f"✅")
            
            # Save progress periodically
            if batch_num % 10 == 0:
                progress_df = pd.concat(all_results, ignore_index=True)
                progress_df.to_csv(self.raw_sentiment_csv + '.progress', index=False)
                print(f"      💾 Progress saved ({len(progress_df):,} scores)")
        
        if all_results:
            new_scores_df = pd.concat(all_results, ignore_index=True)
            print(f"   ✅ Scored {len(new_scores_df):,} new headlines")
            return new_scores_df
        else:
            return pd.DataFrame()
    
    def create_sentiment_long_format(self, sentiment_df):
        """Create ticker-level daily sentiment features"""
        print("🔧 Creating ticker-level sentiment features...")
        
        # Convert datetime to date-only (remove timestamps for daily aggregation)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
        
        # Daily aggregation by ticker
        daily_ticker = sentiment_df.groupby(['Date', 'ticker']).agg({
            'sentiment': ['mean', 'std', 'count', 'min', 'max'],
            'unique_key': 'count'  # News volume
        }).reset_index()
        
        # Flatten columns
        daily_ticker.columns = [
            'Date', 'ticker', 'sentiment_mean', 'sentiment_std', 
            'sentiment_count', 'sentiment_min', 'sentiment_max', 'news_volume'
        ]
        
        # Convert Date back to datetime for consistency
        daily_ticker['Date'] = pd.to_datetime(daily_ticker['Date'])
        
        # Sort by ticker and date for rolling calculations
        daily_ticker = daily_ticker.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        # Calculate rolling features per ticker
        for ticker in daily_ticker['ticker'].unique():
            mask = daily_ticker['ticker'] == ticker
            ticker_data = daily_ticker[mask].copy()
            
            # Rolling averages
            ticker_data['sentiment_ma_3d'] = ticker_data['sentiment_mean'].rolling(3, min_periods=1).mean()
            ticker_data['sentiment_ma_7d'] = ticker_data['sentiment_mean'].rolling(7, min_periods=1).mean()
            ticker_data['sentiment_ma_30d'] = ticker_data['sentiment_mean'].rolling(30, min_periods=1).mean()
            
            # Momentum
            ticker_data['sentiment_momentum_3d'] = ticker_data['sentiment_mean'].diff(3)
            ticker_data['sentiment_momentum_7d'] = ticker_data['sentiment_mean'].diff(7)
            
            # Volatility
            ticker_data['sentiment_volatility_7d'] = ticker_data['sentiment_mean'].rolling(7, min_periods=1).std()
            
            # Update main dataframe
            daily_ticker.loc[mask, ticker_data.columns] = ticker_data
        
        # Extreme sentiment flags
        daily_ticker['extreme_positive'] = (daily_ticker['sentiment_mean'] > 0.5).astype(int)
        daily_ticker['extreme_negative'] = (daily_ticker['sentiment_mean'] < -0.5).astype(int)
        
        print(f"   ✅ Ticker-level features: {len(daily_ticker):,} records")
        return daily_ticker
    
    def create_market_features(self, sentiment_df):
        """Create market-wide sentiment features for database merge"""
        print("🏗️ Creating market-wide sentiment features...")
        
        # Convert datetime to date-only (remove timestamps for daily aggregation)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
        
        daily_features = []
        
        for date in sorted(sentiment_df['Date'].unique()):
            date_data = sentiment_df[sentiment_df['Date'] == date]
            features = {'Date': date}
            
            # Overall market sentiment
            features['market_sentiment_mean'] = date_data['sentiment'].mean()
            features['market_sentiment_std'] = date_data['sentiment'].std()
            features['total_news_volume'] = len(date_data)
            
            # Market indices sentiment
            for index in self.market_indices:
                index_data = date_data[date_data['ticker'] == index]
                if not index_data.empty:
                    features[f'{index}_sentiment'] = index_data['sentiment'].mean()
                    features[f'{index}_news_volume'] = len(index_data)
            
            # Sector sentiment
            for sector, tickers in self.sector_tickers.items():
                sector_data = date_data[date_data['ticker'].isin(tickers)]
                if not sector_data.empty:
                    features[f'{sector}_sentiment'] = sector_data['sentiment'].mean()
                    features[f'{sector}_news_volume'] = len(sector_data)
            
            # Safe haven sentiment
            safe_data = date_data[date_data['ticker'].isin(self.safe_haven)]
            if not safe_data.empty:
                features['safe_haven_sentiment'] = safe_data['sentiment'].mean()
                features['safe_haven_news_volume'] = len(safe_data)
            
            # Sentiment breadth
            total_tickers = date_data['ticker'].nunique()
            positive_tickers = (date_data.groupby('ticker')['sentiment'].mean() > 0).sum()
            negative_tickers = (date_data.groupby('ticker')['sentiment'].mean() < 0).sum()
            
            features['sentiment_breadth_positive_pct'] = (positive_tickers / total_tickers * 100) if total_tickers > 0 else 0
            features['sentiment_breadth_negative_pct'] = (negative_tickers / total_tickers * 100) if total_tickers > 0 else 0
            features['sentiment_breadth_ratio'] = positive_tickers / negative_tickers if negative_tickers > 0 else 10
            
            # Extreme sentiment counts
            features['extreme_positive_count'] = (date_data['sentiment'] > 0.5).sum()
            features['extreme_negative_count'] = (date_data['sentiment'] < -0.5).sum()
            
            daily_features.append(features)
        
        market_df = pd.DataFrame(daily_features)
        
        # Add rolling features for regime detection
        if len(market_df) > 10:
            market_df['market_sentiment_ma_10d'] = market_df['market_sentiment_mean'].rolling(10, min_periods=1).mean()
            market_df['market_sentiment_trend'] = market_df['market_sentiment_mean'] - market_df['market_sentiment_ma_10d']
            market_df['sentiment_regime_positive'] = (market_df['market_sentiment_mean'] > market_df['market_sentiment_ma_10d']).astype(int)
        
        print(f"   ✅ Market features: {len(market_df)} dates × {len(market_df.columns)} features")
        return market_df
    
    def create_wide_optimized_format(self, long_df):
        """Create ML-ready wide format"""
        print("📊 Creating wide optimized format (ML ready)...")
        
        daily_wide = []
        
        all_dates = sorted(long_df['Date'].unique())
        
        for i, date in enumerate(all_dates):
            if i % 100 == 0 and i > 0:
                print(f"      Progress: {i:,}/{len(all_dates):,} dates")
            
            date_data = long_df[long_df['Date'] == date]
            record = {'Date': date}
            
            # Key market indices
            for index in self.market_indices:
                index_data = date_data[date_data['ticker'] == index]
                if not index_data.empty:
                    record[f'{index}_sentiment'] = index_data['sentiment_mean'].iloc[0]
                    record[f'{index}_news_volume'] = index_data['news_volume'].iloc[0]
            
            # Key individual stocks (top 20 by market cap)
            key_stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK.B', 
                         'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'WMT', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV']
            for stock in key_stocks:
                stock_data = date_data[date_data['ticker'] == stock]
                if not stock_data.empty:
                    record[f'{stock}_sentiment'] = stock_data['sentiment_mean'].iloc[0]
            
            # Sector aggregates
            for sector, tickers in self.sector_tickers.items():
                sector_data = date_data[date_data['ticker'].isin(tickers)]
                if not sector_data.empty:
                    record[f'{sector}_sentiment'] = sector_data['sentiment_mean'].mean()
            
            # Market breadth
            positive_count = (date_data['sentiment_mean'] > 0).sum()
            negative_count = (date_data['sentiment_mean'] < 0).sum()
            total_count = len(date_data)
            
            record['sentiment_breadth_positive'] = positive_count / total_count if total_count > 0 else 0
            record['sentiment_breadth_negative'] = negative_count / total_count if total_count > 0 else 0
            record['total_news_volume'] = date_data['news_volume'].sum()
            
            daily_wide.append(record)
        
        wide_df = pd.DataFrame(daily_wide)
        wide_df = wide_df.sort_values('Date').reset_index(drop=True)
        
        print(f"   ✅ Wide format: {len(wide_df):,} dates × {len(wide_df.columns)} features")
        return wide_df
    
    def save_all_outputs(self, raw_sentiment, long_df, market_df, wide_df):
        """Save all sentiment output formats"""
        print("💾 Saving all sentiment outputs...")
        
        # Combine with existing raw sentiment
        if os.path.exists(self.raw_sentiment_csv):
            existing = pd.read_csv(self.raw_sentiment_csv, parse_dates=['Date'])
            combined_raw = pd.concat([existing, raw_sentiment], ignore_index=True)
            combined_raw = combined_raw.drop_duplicates(subset=['unique_key'], keep='last')
        else:
            combined_raw = raw_sentiment
        
        # Sort data
        combined_raw = combined_raw.sort_values(['Date', 'ticker']).reset_index(drop=True)
        long_df = long_df.sort_values(['Date', 'ticker']).reset_index(drop=True)
        market_df = market_df.sort_values('Date').reset_index(drop=True)
        wide_df = wide_df.sort_values('Date').reset_index(drop=True)
        
        # Format dates as YYYY-MM-DD only (remove timestamps for database consistency)
        long_df['Date'] = pd.to_datetime(long_df['Date']).dt.strftime('%Y-%m-%d')
        market_df['Date'] = pd.to_datetime(market_df['Date']).dt.strftime('%Y-%m-%d')
        wide_df['Date'] = pd.to_datetime(wide_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Save all formats
        combined_raw.to_csv(self.raw_sentiment_csv, index=False)
        long_df.to_csv(self.long_csv, index=False)
        market_df.to_csv(self.market_features_csv, index=False)
        wide_df.to_csv(self.wide_optimized_csv, index=False)
        
        print(f"   ✅ Raw sentiment: {len(combined_raw):,} scores → {os.path.basename(self.raw_sentiment_csv)}")
        print(f"   ✅ Long format: {len(long_df):,} records → {os.path.basename(self.long_csv)}")
        print(f"   ✅ Market features: {len(market_df):,} dates → {os.path.basename(self.market_features_csv)}")
        print(f"   ✅ Wide optimized: {len(wide_df):,} dates → {os.path.basename(self.wide_optimized_csv)}")
        
        return {
            'raw': len(combined_raw),
            'long': len(long_df),
            'market': len(market_df),
            'wide': len(wide_df),
            'date_range': f"{long_df['Date'].min()} to {long_df['Date'].max()}"
        }
    
    def run_daily_update(self):
        """Main daily sentiment update workflow"""
        print("🚀 NEWS & SENTIMENT DAILY UPDATE - Integrated Workflow")
        print("=" * 60)
        
        try:
            # Load headlines
            headlines_df = self.load_headlines()
            
            # Load existing sentiment scores
            existing_sentiment, already_scored = self.load_existing_sentiment()
            
            # Process new sentiment scoring
            new_sentiment = self.process_sentiment_scoring(headlines_df, already_scored)
            
            # Combine all sentiment data for aggregation
            if not new_sentiment.empty and not existing_sentiment.empty:
                all_sentiment = pd.concat([existing_sentiment, new_sentiment], ignore_index=True)
            elif not new_sentiment.empty:
                all_sentiment = new_sentiment
            elif not existing_sentiment.empty:
                all_sentiment = existing_sentiment
            else:
                print("❌ No sentiment data available")
                return
            
            # Create all output formats
            long_format = self.create_sentiment_long_format(all_sentiment)
            market_features = self.create_market_features(all_sentiment)
            wide_format = self.create_wide_optimized_format(long_format)
            
            # Save outputs
            summary = self.save_all_outputs(new_sentiment, long_format, market_features, wide_format)
            
            print(f"\n🎯 SENTIMENT UPDATE COMPLETE!")
            print(f"   📊 Raw sentiment scores: {summary['raw']:,}")
            print(f"   📈 Ticker-level daily: {summary['long']:,} records")
            print(f"   🗄️ Market features (DB ready): {summary['market']:,} dates")
            print(f"   🤖 Wide optimized (ML ready): {summary['wide']:,} dates")
            print(f"   📅 Date range: {summary['date_range']}")
            print(f"\n✅ Ready for SQLite database integration!")
            
        except Exception as e:
            print(f"❌ Error in sentiment update: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Daily news & sentiment update - single command"""
    print("🎯 INTEGRATED NEWS & SENTIMENT DAILY UPDATER")
    print("   Headlines → Sentiment Analysis → Aggregation")
    print("   Output: Ticker-level + Market-wide features for database")
    print()
    
    updater = IntegratedNewsSentimentUpdater()
    updater.run_daily_update()

if __name__ == "__main__":
    main()
