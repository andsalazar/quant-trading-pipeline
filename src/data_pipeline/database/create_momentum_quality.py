"""
Professional Momentum Quality Factors
Generate sophisticated momentum quality and factor timing features
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_momentum_quality_features():
    """
    Create professional momentum quality factors:
    1. Risk-Adjusted Momentum
    2. Momentum Consistency & Quality
    3. Cross-Asset Momentum Confirmation
    4. Momentum Breadth Analysis
    5. Sector Rotation Momentum
    6. Factor Timing Signals
    """
    
    print("⚡ CREATING MOMENTUM QUALITY FACTORS")
    print("=" * 80)
    
    # Connect to database
    conn = sqlite3.connect('quant_trading_final.db')
    
    # Load required data
    print("📊 Loading market data for momentum analysis...")
    
    # Get individual stock data
    market_query = """
    SELECT Date, Symbol, Close, Daily_Return, Volume, 
           Price_Change_5d, Price_Change_20d, Volatility_20d,
           RSI_14, SMA_20, SMA_50, SMA_200, ATR_14
    FROM market_data
    ORDER BY Date, Symbol
    """
    
    market_df = pd.read_sql_query(market_query, conn)
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    
    # Get market features for breadth
    market_features_query = """
    SELECT Date, Market_Return_Mean, Market_Return_Std,
           Stocks_Above_SMA20, Stocks_Below_SMA20, Average_RSI,
           High_RSI_Count, Low_RSI_Count
    FROM market_features
    ORDER BY Date
    """
    
    market_features_df = pd.read_sql_query(market_features_query, conn)
    market_features_df['Date'] = pd.to_datetime(market_features_df['Date'])
    
    # Get regime features for confirmation
    regime_query = """
    SELECT Date, VIX_regime_numeric, Risk_regime_numeric,
           Trend_regime_numeric, Dollar_regime_numeric
    FROM regime_features
    ORDER BY Date
    """
    
    regime_df = pd.read_sql_query(regime_query, conn)
    regime_df['Date'] = pd.to_datetime(regime_df['Date'])
    
    # Get cross-asset data
    cross_asset_query = """
    SELECT Date, Bond_equity_rotation_5d, Risk_asset_breadth_5d,
           Asset_rotation_strength
    FROM cross_asset_flows
    ORDER BY Date
    """
    
    cross_asset_df = pd.read_sql_query(cross_asset_query, conn)
    cross_asset_df['Date'] = pd.to_datetime(cross_asset_df['Date'])
    
    print(f"📈 Processing {market_df['Symbol'].nunique()} stocks across {market_df['Date'].nunique()} trading days...")
    
    # Merge all data
    momentum_df = market_df.copy()
    momentum_df = momentum_df.merge(market_features_df, on='Date', how='left')
    momentum_df = momentum_df.merge(regime_df, on='Date', how='left')
    momentum_df = momentum_df.merge(cross_asset_df, on='Date', how='left')
    
    # 1. RISK-ADJUSTED MOMENTUM
    print("\n📊 Creating risk-adjusted momentum features...")
    
    # Sort for time series operations
    momentum_df = momentum_df.sort_values(['Symbol', 'Date'])
    
    # Calculate momentum returns over different periods
    for period in [5, 10, 20, 63]:
        momentum_df[f'Return_{period}d'] = momentum_df.groupby('Symbol')['Daily_Return'].rolling(period).sum().reset_index(0, drop=True)
        momentum_df[f'Volatility_{period}d'] = momentum_df.groupby('Symbol')['Daily_Return'].rolling(period).std().reset_index(0, drop=True) * np.sqrt(252)
    
    # Risk-adjusted momentum (Sharpe-like ratios)
    for period in [5, 10, 20, 63]:
        momentum_df[f'Risk_adj_momentum_{period}d'] = (
            momentum_df[f'Return_{period}d'] / (momentum_df[f'Volatility_{period}d'] + 0.01)
        )
    
    # Momentum quality score (consistency + magnitude)
    momentum_df['Momentum_quality_score'] = (
        momentum_df['Risk_adj_momentum_20d'] * 0.4 +
        momentum_df['Risk_adj_momentum_63d'] * 0.6
    )
    
    # Downside-adjusted momentum (only penalize downside volatility)
    momentum_df['Downside_volatility_20d'] = momentum_df.groupby('Symbol')['Daily_Return'].rolling(20).apply(
        lambda x: x[x < 0].std() * np.sqrt(252) if len(x[x < 0]) > 0 else 0.01
    ).reset_index(0, drop=True)
    
    momentum_df['Sortino_momentum_20d'] = (
        momentum_df['Return_20d'] / (momentum_df['Downside_volatility_20d'] + 0.01)
    )
    
    # 2. MOMENTUM CONSISTENCY & QUALITY
    print("🎯 Creating momentum consistency features...")
    
    # Momentum consistency (% of positive days within trend)
    for period in [10, 20]:
        momentum_df[f'Momentum_consistency_{period}d'] = momentum_df.groupby('Symbol')['Daily_Return'].rolling(period).apply(
            lambda x: (x > 0).sum() / len(x)
        ).reset_index(0, drop=True)
    
    # Momentum acceleration (2nd derivative)
    momentum_df['Return_5d_acceleration'] = momentum_df.groupby('Symbol')['Return_5d'].diff().reset_index(0, drop=True)
    momentum_df['Return_20d_acceleration'] = momentum_df.groupby('Symbol')['Return_20d'].diff().reset_index(0, drop=True)
    
    # Momentum persistence (how long has momentum lasted)
    momentum_df['Positive_momentum_streak'] = momentum_df.groupby('Symbol')['Return_5d'].transform(
        lambda x: (x > 0).groupby((x <= 0).cumsum()).cumsum()
    )
    
    momentum_df['Negative_momentum_streak'] = momentum_df.groupby('Symbol')['Return_5d'].transform(
        lambda x: (x < 0).groupby((x >= 0).cumsum()).cumsum()
    )
    
    # Momentum strength relative to historical
    momentum_df['Momentum_percentile_252d'] = momentum_df.groupby('Symbol')['Return_20d'].transform(
        lambda x: x.rolling(252).rank(pct=True)
    )
    
    # Quality flags
    momentum_df['High_quality_momentum'] = (
        (momentum_df['Risk_adj_momentum_20d'] > 1.0) & 
        (momentum_df['Momentum_consistency_20d'] > 0.6) &
        (momentum_df['Momentum_percentile_252d'] > 0.8)
    ).astype(int)
    
    momentum_df['Low_quality_momentum'] = (
        (momentum_df['Risk_adj_momentum_20d'] < -1.0) & 
        (momentum_df['Momentum_consistency_20d'] < 0.4) &
        (momentum_df['Momentum_percentile_252d'] < 0.2)
    ).astype(int)
    
    # 3. CROSS-ASSET MOMENTUM CONFIRMATION
    print("🌍 Creating cross-asset momentum confirmation...")
    
    # Align stock momentum with broader market trends
    momentum_df['Stock_vs_market_momentum'] = (
        momentum_df['Return_20d'] - momentum_df['Market_Return_Mean'] * 20
    )
    
    # Cross-asset confirmation scores
    momentum_df['Cross_asset_confirmation'] = (
        # Positive if stock momentum aligns with cross-asset flows
        np.sign(momentum_df['Return_20d']) == np.sign(momentum_df['Bond_equity_rotation_5d'])
    ).astype(int)
    
    # Regime-adjusted momentum (stronger in favorable regimes)
    momentum_df['Regime_adjusted_momentum'] = momentum_df['Return_20d'] * (
        1 + 0.2 * (momentum_df['Trend_regime_numeric'] >= 3) +  # Bull markets
        0.1 * (momentum_df['Risk_regime_numeric'] >= 3) -       # Risk-on
        0.2 * (momentum_df['VIX_regime_numeric'] == 2)          # High vol penalty
    )
    
    # 4. MOMENTUM BREADTH ANALYSIS
    print("📊 Creating momentum breadth features...")
    
    # Individual stock contribution to market breadth
    momentum_df['Above_SMA20'] = (momentum_df['Close'] > momentum_df['SMA_20']).astype(int)
    momentum_df['Above_SMA50'] = (momentum_df['Close'] > momentum_df['SMA_50']).astype(int)
    momentum_df['Above_SMA200'] = (momentum_df['Close'] > momentum_df['SMA_200']).astype(int)
    
    # Momentum relative to market breadth
    momentum_df['Momentum_vs_breadth'] = momentum_df['Return_20d'] - (
        momentum_df['Stocks_Above_SMA20'] / 500  # Normalize breadth
    )
    
    # Leadership vs laggard classification
    momentum_df['Momentum_leadership'] = (
        momentum_df['Momentum_percentile_252d'] > 0.9
    ).astype(int)
    
    momentum_df['Momentum_laggard'] = (
        momentum_df['Momentum_percentile_252d'] < 0.1
    ).astype(int)
    
    # 5. SECTOR ROTATION MOMENTUM (Simplified)
    print("🔄 Creating sector rotation proxies...")
    
    # Use symbol patterns to identify sector proxies
    # This is simplified - in practice you'd have sector classifications
    momentum_df['Tech_proxy'] = momentum_df['Symbol'].str.contains('AAPL|MSFT|GOOGL|AMZN|NVDA', na=False).astype(int)
    momentum_df['Finance_proxy'] = momentum_df['Symbol'].str.contains('JPM|BAC|WFC|GS|MS', na=False).astype(int)
    momentum_df['Healthcare_proxy'] = momentum_df['Symbol'].str.contains('JNJ|PFE|UNH|MRK|ABT', na=False).astype(int)
    
    # Sector momentum relative performance (simplified)
    for sector in ['Tech_proxy', 'Finance_proxy', 'Healthcare_proxy']:
        sector_mask = momentum_df[sector] == 1
        if sector_mask.sum() > 0:
            sector_return = momentum_df.loc[sector_mask].groupby('Date')['Return_20d'].mean()
            momentum_df[f'{sector}_sector_momentum'] = momentum_df['Date'].map(sector_return).fillna(0)
            
            # Individual stock vs sector momentum
            momentum_df[f'Stock_vs_{sector}_momentum'] = np.where(
                momentum_df[sector] == 1,
                momentum_df['Return_20d'] - momentum_df[f'{sector}_sector_momentum'],
                0
            )
    
    # 6. FACTOR TIMING SIGNALS
    print("⏰ Creating factor timing features...")
    
    # Value factor proxy (low P/E like behavior - using RSI as proxy)
    momentum_df['Value_factor_proxy'] = 100 - momentum_df['RSI_14']  # Oversold = value
    momentum_df['Value_momentum_divergence'] = (
        momentum_df['Return_20d'] * momentum_df['Value_factor_proxy'] / 100
    )
    
    # Size factor proxy (assuming smaller symbols = small cap, simplified)
    momentum_df['Size_factor_proxy'] = 1 / (momentum_df['Close'] + 1)  # Lower price = smaller
    momentum_df['Size_momentum_interaction'] = (
        momentum_df['Return_20d'] * momentum_df['Size_factor_proxy']
    )
    
    # Quality factor (using volatility and consistency)
    momentum_df['Quality_factor_score'] = (
        1 / (momentum_df['Volatility_20d'] + 0.1) * momentum_df['Momentum_consistency_20d']
    )
    
    momentum_df['Quality_momentum_interaction'] = (
        momentum_df['Return_20d'] * momentum_df['Quality_factor_score']
    )
    
    # Low volatility factor timing
    momentum_df['Low_vol_factor_strength'] = 1 / (momentum_df['Volatility_20d'] + 0.1)
    momentum_df['Low_vol_outperformance'] = (
        momentum_df['Return_20d'] * momentum_df['Low_vol_factor_strength']
    )
    
    # Factor crowding detection (when factor loadings are extreme)
    momentum_df['Momentum_factor_crowding'] = (
        momentum_df['Momentum_percentile_252d'] > 0.95
    ).astype(int) + (
        momentum_df['Momentum_percentile_252d'] < 0.05
    ).astype(int)
    
    print(f"\n✅ Created {len(momentum_df.columns)} momentum quality features")
    
    # Select final momentum quality features
    momentum_features = [
        'Date', 'Symbol',
        # Risk-Adjusted Momentum
        'Risk_adj_momentum_20d', 'Risk_adj_momentum_63d', 'Momentum_quality_score', 'Sortino_momentum_20d',
        # Momentum Consistency
        'Momentum_consistency_20d', 'Return_20d_acceleration', 'Positive_momentum_streak',
        'Momentum_percentile_252d', 'High_quality_momentum', 'Low_quality_momentum',
        # Cross-Asset Confirmation
        'Stock_vs_market_momentum', 'Cross_asset_confirmation', 'Regime_adjusted_momentum',
        # Breadth Analysis
        'Above_SMA20', 'Above_SMA50', 'Above_SMA200', 'Momentum_vs_breadth',
        'Momentum_leadership', 'Momentum_laggard',
        # Factor Timing
        'Value_momentum_divergence', 'Quality_momentum_interaction', 'Low_vol_outperformance',
        'Momentum_factor_crowding'
    ]
    
    # Add sector features if they exist (avoid duplicates)
    sector_features = [col for col in momentum_df.columns if 'sector_momentum' in col or ('Stock_vs_' in col and col not in momentum_features)]
    momentum_features.extend(sector_features)
    
    # Filter to available columns
    available_features = [col for col in momentum_features if col in momentum_df.columns]
    final_momentum_df = momentum_df[available_features].copy()
    
    print(f"📊 Final momentum quality features: {len(available_features)-2}")  # -2 for Date, Symbol
    
    # Save to database
    print("\n💾 Saving momentum quality features to database...")
    
    final_momentum_df.to_sql('momentum_quality', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("✅ Momentum quality factors completed!")
    print(f"📈 Generated {len(available_features)-2} professional momentum features")
    
    return final_momentum_df

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Starting momentum quality feature generation at {start_time}")
    
    momentum_df = create_momentum_quality_features()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏱️ Completed in {duration.total_seconds():.1f} seconds")
    
    # Display sample features
    print(f"\n📊 SAMPLE MOMENTUM QUALITY FEATURES:")
    print("=" * 70)
    
    recent_data = momentum_df.tail(10)
    key_features = [
        'Risk_adj_momentum_20d', 'Momentum_quality_score', 
        'High_quality_momentum', 'Cross_asset_confirmation',
        'Momentum_leadership'
    ]
    
    for feature in key_features:
        if feature in recent_data.columns:
            latest_value = recent_data[feature].iloc[-1]
            print(f"{feature:30s}: {latest_value:8.4f}")
            
    # Final pipeline status
    print(f"\n📋 PROFESSIONAL FEATURES PIPELINE - COMPLETED!")
    print("=" * 60)
    print("✅ Market Regime Features - COMPLETED")
    print("✅ Cross-Asset Flow Signals - COMPLETED") 
    print("✅ Sentiment Divergence Signals - COMPLETED")
    print("✅ Momentum Quality Factors - COMPLETED")
    print("⏳ Update Final Database - READY TO EXECUTE")
    
    # Show feature summary
    print(f"\n📊 PROFESSIONAL FEATURES SUMMARY:")
    print(f"🌡️  Market Regime Features: 19 features")
    print(f"🔄 Cross-Asset Flow Signals: 25 features") 
    print(f"🧠 Sentiment Divergence: 21 ticker + 7 market features")
    print(f"⚡ Momentum Quality Factors: {len(momentum_df.columns)-2} features")
    print(f"📈 TOTAL PROFESSIONAL FEATURES: ~72+ features")
    
    print(f"\n🎯 READY FOR ML TRAINING WITH PROFESSIONAL FEATURES!")
    print("Next step: Integrate all features into final database")