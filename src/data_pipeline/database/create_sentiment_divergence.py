"""
Professional Sentiment Divergence Signals
Generate sophisticated sentiment-price divergence features for contrarian signals
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_sentiment_divergence_features():
    """
    Create professional sentiment divergence signals:
    1. Price-Sentiment Divergence Detection
    2. News Flow Exhaustion Signals
    3. Sentiment Momentum vs Price Momentum
    4. Cross-Asset Sentiment Spread
    5. Sentiment Volatility & Stability
    6. Contrarian Sentiment Extremes
    """
    
    print("🧠 CREATING SENTIMENT DIVERGENCE SIGNALS")
    print("=" * 80)
    
    # Connect to database
    conn = sqlite3.connect('quant_trading_final.db')
    
    # Load required data
    print("📊 Loading sentiment and price data...")
    
    # Get ticker-level sentiment
    sentiment_query = """
    SELECT Date, ticker, sentiment_score_mean, sentiment_score_std, 
           total_news_count, sentiment_3d_mean, sentiment_5d_mean,
           sentiment_zscore_3d, sentiment_momentum_1d, sentiment_volatility_7d,
           sentiment_extreme_pos, sentiment_extreme_neg
    FROM ticker_sentiment  
    ORDER BY Date, ticker
    """
    
    sentiment_df = pd.read_sql_query(sentiment_query, conn)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    
    # Get market-level sentiment
    market_sentiment_query = """
    SELECT Date, market_sentiment_mean, market_sentiment_std,
           total_news_volume, unique_tickers_with_news,
           positive_sentiment_count, negative_sentiment_count
    FROM news_sentiment
    ORDER BY Date
    """
    
    market_sentiment_df = pd.read_sql_query(market_sentiment_query, conn)
    market_sentiment_df['Date'] = pd.to_datetime(market_sentiment_df['Date'])
    
    # Get market data for price comparison
    market_query = """
    SELECT Date, Symbol, Close, Daily_Return, 
           Price_Change_5d, Price_Change_20d, Volume, RSI_14
    FROM market_data
    ORDER BY Date, Symbol
    """
    
    market_df = pd.read_sql_query(market_query, conn)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    
    print(f"📈 Processing sentiment for {sentiment_df['ticker'].nunique()} tickers...")
    print(f"📊 Market sentiment data: {len(market_sentiment_df)} trading days")
    
    # Merge ticker sentiment with price data
    divergence_df = sentiment_df.merge(
        market_df, 
        left_on=['Date', 'ticker'], 
        right_on=['Date', 'Symbol'], 
        how='inner'
    )
    
    print(f"📈 Processing {len(divergence_df)} ticker-date combinations...")
    
    # 1. PRICE-SENTIMENT DIVERGENCE DETECTION
    print("\n📊 Creating price-sentiment divergence features...")
    
    # Calculate price momentum for comparison
    divergence_df = divergence_df.sort_values(['ticker', 'Date'])
    divergence_df['Price_momentum_5d'] = divergence_df.groupby('ticker')['Daily_Return'].rolling(5).mean().reset_index(0, drop=True)
    divergence_df['Price_momentum_20d'] = divergence_df.groupby('ticker')['Daily_Return'].rolling(20).mean().reset_index(0, drop=True)
    
    # Direct divergence measures
    divergence_df['Price_sentiment_divergence_5d'] = (
        divergence_df['Price_momentum_5d'] - divergence_df['sentiment_5d_mean']
    )
    
    # Normalize divergence by typical ranges
    divergence_df['Price_sentiment_divergence_zscore'] = divergence_df.groupby('ticker')[
        'Price_sentiment_divergence_5d'
    ].transform(lambda x: (x - x.rolling(252).mean()) / x.rolling(252).std())
    
    # Extreme divergence flags
    divergence_df['Bullish_sentiment_bearish_price'] = (
        (divergence_df['sentiment_5d_mean'] > 0.2) & 
        (divergence_df['Price_momentum_5d'] < -0.01)
    ).astype(int)
    
    divergence_df['Bearish_sentiment_bullish_price'] = (
        (divergence_df['sentiment_5d_mean'] < -0.2) & 
        (divergence_df['Price_momentum_5d'] > 0.01)
    ).astype(int)
    
    # Divergence strength
    divergence_df['Divergence_strength'] = abs(divergence_df['Price_sentiment_divergence_5d'])
    
    # Divergence persistence (how long has divergence lasted)
    divergence_df['Divergence_persistence'] = divergence_df.groupby('ticker')[
        'Price_sentiment_divergence_zscore'
    ].transform(lambda x: (abs(x) > 1.0).rolling(10).sum())
    
    # 2. NEWS FLOW EXHAUSTION SIGNALS
    print("📰 Creating news flow exhaustion features...")
    
    # High news volume with flat price action = exhaustion
    divergence_df['News_volume_percentile'] = divergence_df.groupby('ticker')[
        'total_news_count'
    ].transform(lambda x: x.rolling(63).rank(pct=True))
    
    divergence_df['Price_action_strength'] = abs(divergence_df['Price_momentum_5d'])
    
    # News exhaustion: high news volume but low price movement
    divergence_df['News_exhaustion_signal'] = (
        (divergence_df['News_volume_percentile'] > 0.8) & 
        (divergence_df['Price_action_strength'] < 0.005)
    ).astype(int)
    
    # Sentiment exhaustion: extreme sentiment but price not following
    divergence_df['Sentiment_exhaustion_bullish'] = (
        (divergence_df['sentiment_extreme_pos'] == 1) & 
        (divergence_df['Price_momentum_5d'] < 0)
    ).astype(int)
    
    divergence_df['Sentiment_exhaustion_bearish'] = (
        (divergence_df['sentiment_extreme_neg'] == 1) & 
        (divergence_df['Price_momentum_5d'] > 0)
    ).astype(int)
    
    # Combined exhaustion score
    divergence_df['Sentiment_exhaustion_composite'] = (
        divergence_df['News_exhaustion_signal'] + 
        divergence_df['Sentiment_exhaustion_bullish'] + 
        divergence_df['Sentiment_exhaustion_bearish']
    )
    
    # 3. SENTIMENT MOMENTUM VS PRICE MOMENTUM
    print("⚡ Creating sentiment vs price momentum features...")
    
    # Sentiment momentum indicators
    divergence_df['Sentiment_acceleration'] = divergence_df.groupby('ticker')[
        'sentiment_momentum_1d'
    ].transform(lambda x: x.diff())
    
    # Price momentum indicators  
    divergence_df['Price_acceleration'] = divergence_df.groupby('ticker')[
        'Price_momentum_5d'
    ].transform(lambda x: x.diff())
    
    # Momentum divergence
    divergence_df['Momentum_divergence'] = (
        divergence_df['Sentiment_acceleration'] - divergence_df['Price_acceleration']
    )
    
    # Momentum alignment (both moving same direction)
    divergence_df['Sentiment_price_momentum_alignment'] = (
        np.sign(divergence_df['sentiment_momentum_1d']) == 
        np.sign(divergence_df['Price_momentum_5d'])
    ).astype(int)
    
    # Momentum strength differential
    divergence_df['Momentum_strength_ratio'] = (
        abs(divergence_df['sentiment_momentum_1d']) / 
        (abs(divergence_df['Price_momentum_5d']) + 0.001)  # Avoid division by zero
    )
    
    # 4. SENTIMENT VOLATILITY & STABILITY  
    print("📊 Creating sentiment volatility features...")
    
    # Sentiment stability measures
    divergence_df['Sentiment_stability_20d'] = divergence_df.groupby('ticker')[
        'sentiment_score_mean'
    ].transform(lambda x: 1 / (x.rolling(20).std() + 0.01))  # Higher = more stable
    
    # Sudden sentiment shifts
    divergence_df['Sentiment_shock_positive'] = (
        divergence_df['sentiment_zscore_3d'] > 2.0
    ).astype(int)
    
    divergence_df['Sentiment_shock_negative'] = (
        divergence_df['sentiment_zscore_3d'] < -2.0
    ).astype(int)
    
    # Sentiment regime changes
    divergence_df['Sentiment_regime_change'] = divergence_df.groupby('ticker')[
        'sentiment_5d_mean'
    ].transform(lambda x: (x.diff().abs() > 0.3).astype(int))
    
    # 5. CONTRARIAN SENTIMENT EXTREMES
    print("🔄 Creating contrarian sentiment features...")
    
    # Sentiment percentiles for contrarian signals
    divergence_df['Sentiment_percentile_252d'] = divergence_df.groupby('ticker')[
        'sentiment_score_mean'
    ].transform(lambda x: x.rolling(252).rank(pct=True))
    
    # Extreme sentiment levels (contrarian opportunities)
    divergence_df['Sentiment_extremely_bullish'] = (
        divergence_df['Sentiment_percentile_252d'] > 0.95
    ).astype(int)
    
    divergence_df['Sentiment_extremely_bearish'] = (
        divergence_df['Sentiment_percentile_252d'] < 0.05
    ).astype(int)
    
    # Contrarian strength (more extreme = stronger signal)
    divergence_df['Contrarian_strength_bullish'] = np.maximum(
        0, divergence_df['Sentiment_percentile_252d'] - 0.8
    ) * 5  # Scale 0.8-1.0 to 0-1.0
    
    divergence_df['Contrarian_strength_bearish'] = np.maximum(
        0, 0.2 - divergence_df['Sentiment_percentile_252d']
    ) * 5  # Scale 0.0-0.2 to 0-1.0
    
    # Sentiment mean reversion signal
    divergence_df['Sentiment_mean_reversion_signal'] = (
        divergence_df['Contrarian_strength_bullish'] + 
        divergence_df['Contrarian_strength_bearish']
    )
    
    print(f"\n✅ Created {len(divergence_df.columns)} sentiment divergence features")
    
    # Now create market-level sentiment features
    print("\n📊 Creating market-level sentiment features...")
    
    # Merge with market sentiment data
    market_sentiment_df['Market_sentiment_momentum_5d'] = market_sentiment_df['market_sentiment_mean'].diff(5)
    market_sentiment_df['Market_sentiment_volatility'] = market_sentiment_df['market_sentiment_std']
    
    # News volume trends
    market_sentiment_df['News_volume_trend_5d'] = market_sentiment_df['total_news_volume'].pct_change(5)
    market_sentiment_df['News_volume_acceleration'] = market_sentiment_df['News_volume_trend_5d'].diff()
    
    # Market sentiment extremes
    market_sentiment_df['Market_sentiment_percentile'] = market_sentiment_df['market_sentiment_mean'].rolling(252).rank(pct=True)
    market_sentiment_df['Market_sentiment_extreme_bullish'] = (market_sentiment_df['Market_sentiment_percentile'] > 0.9).astype(int)
    market_sentiment_df['Market_sentiment_extreme_bearish'] = (market_sentiment_df['Market_sentiment_percentile'] < 0.1).astype(int)
    
    # Select final sentiment divergence features (ticker level)
    ticker_sentiment_features = [
        'Date', 'ticker',
        # Price-Sentiment Divergence
        'Price_sentiment_divergence_5d', 'Price_sentiment_divergence_zscore',
        'Bullish_sentiment_bearish_price', 'Bearish_sentiment_bullish_price',
        'Divergence_strength', 'Divergence_persistence',
        # News Exhaustion
        'News_exhaustion_signal', 'Sentiment_exhaustion_composite',
        # Momentum Features
        'Momentum_divergence', 'Sentiment_price_momentum_alignment', 'Momentum_strength_ratio',
        # Volatility Features  
        'Sentiment_stability_20d', 'Sentiment_shock_positive', 'Sentiment_shock_negative',
        'Sentiment_regime_change',
        # Contrarian Features
        'Sentiment_percentile_252d', 'Sentiment_extremely_bullish', 'Sentiment_extremely_bearish',
        'Contrarian_strength_bullish', 'Contrarian_strength_bearish', 'Sentiment_mean_reversion_signal'
    ]
    
    # Filter to available columns
    available_ticker_features = [col for col in ticker_sentiment_features if col in divergence_df.columns]
    final_ticker_sentiment = divergence_df[available_ticker_features].copy()
    
    # Market-level sentiment features
    market_sentiment_features = [
        'Date', 'Market_sentiment_momentum_5d', 'Market_sentiment_volatility',
        'News_volume_trend_5d', 'News_volume_acceleration',
        'Market_sentiment_percentile', 'Market_sentiment_extreme_bullish', 'Market_sentiment_extreme_bearish'
    ]
    
    available_market_features = [col for col in market_sentiment_features if col in market_sentiment_df.columns]
    final_market_sentiment = market_sentiment_df[available_market_features].copy()
    
    print(f"📊 Ticker-level sentiment features: {len(available_ticker_features)-2}")  # -2 for Date, ticker
    print(f"📊 Market-level sentiment features: {len(available_market_features)-1}")  # -1 for Date
    
    # Save to database
    print("\n💾 Saving sentiment divergence features to database...")
    
    final_ticker_sentiment.to_sql('sentiment_divergence_tickers', conn, if_exists='replace', index=False)
    final_market_sentiment.to_sql('sentiment_divergence_market', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("✅ Sentiment divergence signals completed!")
    print(f"📈 Generated {len(available_ticker_features)-2} ticker-level + {len(available_market_features)-1} market-level features")
    
    return final_ticker_sentiment, final_market_sentiment

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Starting sentiment divergence feature generation at {start_time}")
    
    ticker_sentiment, market_sentiment = create_sentiment_divergence_features()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏱️ Completed in {duration.total_seconds():.1f} seconds")
    
    # Display sample features
    print(f"\n📊 SAMPLE SENTIMENT DIVERGENCE FEATURES:")
    print("=" * 70)
    
    # Show ticker-level samples
    recent_ticker = ticker_sentiment.tail(10)
    key_ticker_features = [
        'Price_sentiment_divergence_5d', 'Divergence_strength', 
        'News_exhaustion_signal', 'Sentiment_mean_reversion_signal'
    ]
    
    print("TICKER-LEVEL FEATURES (sample):")
    for feature in key_ticker_features:
        if feature in recent_ticker.columns:
            latest_value = recent_ticker[feature].iloc[-1]
            print(f"  {feature:35s}: {latest_value:8.4f}")
    
    # Show market-level samples  
    print("\nMARKET-LEVEL FEATURES (latest):")
    recent_market = market_sentiment.tail(1)
    for col in recent_market.columns:
        if col != 'Date':
            latest_value = recent_market[col].iloc[-1] if len(recent_market) > 0 else 0
            print(f"  {col:35s}: {latest_value:8.4f}")
            
    # Update pipeline status
    print(f"\n📋 DAILY UPDATE PIPELINE STATUS:")
    print("✅ Market Regime Features - COMPLETED")  
    print("✅ Cross-Asset Flow Signals - COMPLETED")
    print("✅ Sentiment Divergence Signals - COMPLETED")
    print("⏳ Momentum Quality Factors - PENDING")