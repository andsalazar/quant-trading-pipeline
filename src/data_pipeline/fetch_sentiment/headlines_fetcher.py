#!/usr/bin/env python3
"""
HEADLINES FETCHER (Step 7 - News & Sentiment Pipeline)
Fetches news headlines from Polygon API for dynamic ticker universe

OUTPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_07_news_sentiment\\00_08_headlines_polygon.csv.gz
- Compressed CSV with columns: Date, ticker, title, summary, url

Features:
- Uses centralized ticker management (no hardcoded lists)
- Intelligent backfill for new/missing tickers
- Proper timezone handling (New York time)
- Efficient API usage with rate limiting
- Creates daily backups
"""

from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_dynamic_ticker_universe():
    """Get current ticker universe from centralized manager"""
    
    # Hardcoded path to ticker universe
    ticker_universe_file = CONFIG_BASE_PATH  # Set in config.py
    
    if os.path.exists(ticker_universe_file):
        print("📊 Loading ticker universe from centralized manager...")
        df = pd.read_csv(ticker_universe_file)
        
        # Filter by priority (1 = highest priority, 2 = medium, 3 = lower)
        # For headlines, we want priority 1 and 2 (avoid too many low-priority symbols)
        priority_symbols = df[df['priority'] <= 2]['symbol'].tolist()
        
        print(f"   ✅ Loaded {len(priority_symbols)} priority symbols")
        print(f"   📈 Total available: {len(df)} symbols")
        
        return sorted(priority_symbols)
    else:
        print("⚠️ Dynamic ticker universe not found. Creating it...")
        
        # Import and run ticker manager
        try:
            from dynamic_ticker_manager import DynamicTickerManager
            manager = DynamicTickerManager()
            manager.build_complete_ticker_universe()
            return get_dynamic_ticker_universe()  # Recursive call after creation
        except Exception as e:
            print(f"❌ Error creating ticker universe: {e}")
            return get_fallback_ticker_list()

def get_fallback_ticker_list():
    """Fallback ticker list if dynamic system fails"""
    
    print("📋 Using fallback ticker list...")
    
    # Essential symbols for market coverage
    fallback_symbols = [
        # Mega cap tech
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
        
        # Major financials
        "JPM", "BAC", "WFC", "BRK.B", "V", "MA", "GS", "MS",
        
        # Healthcare
        "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "DHR",
        
        # Consumer
        "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "DIS",
        
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB",
        
        # Essential ETFs
        "SPY", "QQQ", "IWM", "GLD", "TLT", "VXX", "USO"
    ]
    
    print(f"   📊 Fallback list: {len(fallback_symbols)} symbols")
    return sorted(fallback_symbols)

def setup_polygon_client():
    """Setup Polygon client with secure API key"""
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ POLYGON_API_KEY not found in environment variables.\n"
            "Please add POLYGON_API_KEY=your_key_here to your .env file"
        )
    
    print(f"🔑 Polygon API key loaded: {api_key[:8]}...{api_key[-4:]}")
    return RESTClient(api_key)

def identify_missing_tickers_and_dates(current_tickers, existing_headlines_df):
    """Identify what tickers and date ranges need fetching"""
    
    ny_tz = pytz.timezone('America/New_York')
    today_ny = datetime.now(ny_tz).date()
    
    if existing_headlines_df.empty:
        print("📰 No existing headlines - starting fresh from Oct 1, 2024")
        return {
            'regular_update': {
                'tickers': current_tickers,
                'start_date': datetime(2024, 10, 1).date(),
                'end_date': today_ny
            },
            'backfill': None
        }
    
    # Normalize date column with proper timezone handling
    if 'Date' in existing_headlines_df.columns:
        existing_headlines_df['Date'] = pd.to_datetime(existing_headlines_df['Date'], utc=True)
    
    # Find existing tickers and date range
    existing_tickers = set(existing_headlines_df['ticker'].dropna().unique())
    
    # Regular update: get latest date for existing tickers
    if not existing_headlines_df.empty:
        latest_timestamp = existing_headlines_df['Date'].max()
        if pd.isna(latest_timestamp):
            latest_date = datetime(2024, 10, 1).date()
        else:
            # Ensure timezone-aware handling
            if latest_timestamp.tz is None:
                latest_timestamp = latest_timestamp.tz_localize('UTC')
            latest_date = latest_timestamp.tz_convert(ny_tz).date()
        
        # Start from day after latest date
        regular_start = latest_date + timedelta(days=1)
    else:
        regular_start = datetime(2024, 10, 1).date()
    
    # Find new tickers needing backfill
    new_tickers = [t for t in current_tickers if t not in existing_tickers]
    
    # Determine backfill range for new tickers
    if new_tickers and not existing_headlines_df.empty:
        earliest_timestamp = existing_headlines_df['Date'].min()
        if pd.isna(earliest_timestamp):
            backfill_start = datetime(2024, 10, 1).date()
        else:
            if earliest_timestamp.tz is None:
                earliest_timestamp = earliest_timestamp.tz_localize('UTC')
            backfill_start = earliest_timestamp.tz_convert(ny_tz).date()
    else:
        backfill_start = datetime(2024, 10, 1).date()
    
    strategy = {
        'regular_update': None,
        'backfill': None
    }
    
    # Regular update needed?
    if regular_start <= today_ny:
        strategy['regular_update'] = {
            'tickers': current_tickers,
            'start_date': regular_start,
            'end_date': today_ny
        }
        print(f"📅 Regular update: {regular_start} to {today_ny} ({len(current_tickers)} tickers)")
    
    # Backfill needed?
    if new_tickers:
        strategy['backfill'] = {
            'tickers': new_tickers,
            'start_date': backfill_start,
            'end_date': today_ny
        }
        print(f"🔄 Backfill needed: {len(new_tickers)} new tickers from {backfill_start}")
        print(f"   New tickers: {new_tickers[:10]}{'...' if len(new_tickers) > 10 else ''}")
    
    return strategy

def fetch_headlines_for_tickers(client, tickers, start_date, end_date, description=""):
    """Fetch headlines for specific tickers and date range"""
    
    ny_tz = pytz.timezone('America/New_York')
    all_news = []
    
    print(f"📰 Fetching {description}: {len(tickers)} tickers, {start_date} to {end_date}")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"   {description} ({i}/{len(tickers)}): {ticker}")
        
        try:
            response = client.list_ticker_news(
                ticker=ticker,
                published_utc_gte=start_date.strftime("%Y-%m-%d"),
                published_utc_lt=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                order="asc",
                limit=1000
            )
            
            for item in response:
                # Proper timezone handling
                date_utc = pd.to_datetime(item.published_utc, utc=True)
                date_local = date_utc.tz_convert(ny_tz)
                
                all_news.append({
                    "Date": date_local,
                    "ticker": ticker,
                    "title": item.title,
                    "summary": getattr(item, "summary", ""),
                    "url": item.article_url,
                })
                
        except Exception as e:
            print(f"      ⚠️ Error with {ticker}: {str(e)[:50]}")
        
        # Rate limiting - respect Polygon's limits
        time.sleep(0.12)  # ~8 requests per second (well under 12 req/sec limit)
    
    print(f"   ✅ Collected {len(all_news)} headlines")
    return all_news

def smart_headlines_fetch():
    """Main headlines fetching with smart ticker management"""
    
    print("🚀 HEADLINES FETCHER (Step 7 - News & Sentiment)")
    print(f"⏰ Start time: {datetime.now()}")
    print("=" * 60)
    
    # Setup
    client = setup_polygon_client()
    
    # Hardcoded file paths
    base_path = CONFIG_BASE_PATH  # Set in config.py
    news_file = os.path.join(base_path, "00_08_headlines_polygon.csv.gz")
    backup_file = os.path.join(base_path, f"00_08_headlines_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.csv.gz")
    
    # Load existing headlines
    print("📰 Loading existing headlines...")
    try:
        if os.path.exists(news_file):
            historical_news = pd.read_csv(news_file, compression="gzip")
            
            # Create backup
            historical_news.to_csv(backup_file, index=False, compression="gzip")
            print(f"💾 Backup created: {backup_file}")
            print(f"📊 Existing headlines: {len(historical_news):,}")
        else:
            historical_news = pd.DataFrame(columns=["Date", "ticker", "title", "summary", "url"])
            print("📊 Starting fresh headlines collection")
    except Exception as e:
        print(f"❌ Error loading existing headlines: {e}")
        historical_news = pd.DataFrame(columns=["Date", "ticker", "title", "summary", "url"])
    
    # Get current ticker universe
    current_tickers = get_dynamic_ticker_universe()
    print(f"🎯 Current ticker universe: {len(current_tickers)} symbols")
    
    # Determine fetch strategy
    strategy = identify_missing_tickers_and_dates(current_tickers, historical_news)
    
    all_new_news = []
    
    # Execute regular update
    if strategy['regular_update']:
        update_info = strategy['regular_update']
        news = fetch_headlines_for_tickers(
            client,
            update_info['tickers'],
            update_info['start_date'],
            update_info['end_date'],
            "Regular Update"
        )
        all_new_news.extend(news)
    
    # Execute backfill
    if strategy['backfill']:
        backfill_info = strategy['backfill']
        news = fetch_headlines_for_tickers(
            client,
            backfill_info['tickers'],
            backfill_info['start_date'],
            backfill_info['end_date'],
            "Backfill"
        )
        all_new_news.extend(news)
    
    if not all_new_news:
        print("✅ No new headlines to fetch - everything up to date")
        return
    
    # Process and save results
    print(f"\n💾 Processing {len(all_new_news):,} new headlines...")
    
    new_news_df = pd.DataFrame(all_new_news)
    
    # Ensure consistent timezone handling for new data
    if not new_news_df.empty:
        new_news_df["Date"] = pd.to_datetime(new_news_df["Date"], utc=True)
    
    # Combine with historical data
    if not historical_news.empty:
        # Ensure consistent timezone handling for historical data
        print("🔄 Normalizing timezone data...")
        historical_news["Date"] = pd.to_datetime(historical_news["Date"], utc=True)
        combined = pd.concat([historical_news, new_news_df], ignore_index=True)
    else:
        combined = new_news_df
    
    # Remove duplicates and sort
    print("🔄 Removing duplicates and sorting...")
    combined = combined.drop_duplicates(
        subset=["Date", "ticker", "title"]
    ).sort_values("Date").reset_index(drop=True)
    
    # Save compressed
    combined.to_csv(news_file, index=False, compression="gzip")
    
    # Final summary
    print(f"\n📊 Smart Headlines Fetch Complete:")
    print(f"   📈 Total headlines: {len(combined):,}")
    print(f"   🆕 New headlines: {len(new_news_df):,}")
    print(f"   📅 Date range: {combined['Date'].min().date()} to {combined['Date'].max().date()}")
    print(f"   🎯 Unique tickers: {combined['ticker'].nunique()}")
    print(f"   💾 Saved to: {news_file}")
    
    # Show top tickers by headline count
    ticker_counts = combined['ticker'].value_counts().head(10)
    print(f"\n🏆 Top tickers by headline count:")
    for ticker, count in ticker_counts.items():
        print(f"   {ticker}: {count:,} headlines")
    
    print(f"\n🎉 Smart headlines fetching complete!")

if __name__ == "__main__":
    smart_headlines_fetch()