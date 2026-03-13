"""
Professional Market Regime Detection Features
Generate sophisticated regime classification features for quantitative trading
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_market_regime_features():
    """
    Create professional market regime detection features:
    1. VIX Regime Classification
    2. Yield Curve Regime 
    3. Risk-On/Risk-Off Composite
    4. Dollar Strength Regime
    5. Trend Regime Classification
    6. Volatility Persistence
    """
    
    print("🌡️ CREATING MARKET REGIME DETECTION FEATURES")
    print("=" * 80)
    
    # Connect to database
    conn = sqlite3.connect('quant_trading_final.db')
    
    # Get base data for regime analysis
    print("📊 Loading cross-asset data for regime analysis...")
    
    # Get futures data (VIX proxy, yields, trends)
    futures_query = """
    SELECT Date, SPY_Close, SPY_Return_1d, SPY_Volatility_5d,
           QQQ_Close, QQQ_Return_1d, QQQ_Volatility_5d,
           ES_Close, ES_Return_1d, ES_Volatility_5d,
           TY_Close, TY_Return_1d, FV_Close, FV_Return_1d,
           TLT_Close, TLT_Return_1d, Yield_Curve_Spread,
           VXX_Close, VXX_Return_1d, DX_Close, DX_Return_1d,
           GLD_Close, GLD_Return_1d, CL_Close, CL_Return_1d
    FROM futures_data
    ORDER BY Date
    """
    
    futures_df = pd.read_sql_query(futures_query, conn)
    futures_df['Date'] = pd.to_datetime(futures_df['Date'])
    
    # Get treasury data for yield curve analysis
    treasury_query = """
    SELECT Date, Treasury_2Y, Treasury_5Y, Treasury_10Y, Treasury_30Y,
           TIPS_10Y, Breakeven_5Y, Breakeven_10Y
    FROM treasury_data
    ORDER BY Date
    """
    
    treasury_df = pd.read_sql_query(treasury_query, conn)
    treasury_df['Date'] = pd.to_datetime(treasury_df['Date'])
    
    # Get currency data for dollar regime
    currency_query = """
    SELECT date as Date, EUR_USD, USD_JPY, USD_CHF, AUD_USD,
           USD_Strength_Index, Currency_Risk_Sentiment
    FROM currency_data
    ORDER BY date
    """
    
    currency_df = pd.read_sql_query(currency_query, conn)
    currency_df['Date'] = pd.to_datetime(currency_df['Date'])
    
    # Merge all data
    regime_df = futures_df.merge(treasury_df, on='Date', how='left')
    regime_df = regime_df.merge(currency_df, on='Date', how='left')
    
    print(f"📈 Processing {len(regime_df)} trading days for regime analysis...")
    
    # 1. VIX REGIME CLASSIFICATION
    print("\n🎯 Creating VIX regime features...")
    
    # Use VXX as VIX proxy (actual VIX not in database)
    regime_df['VIX_proxy'] = regime_df['VXX_Close']
    
    # Calculate rolling percentiles for regime classification
    regime_df['VIX_percentile_252d'] = regime_df['VIX_proxy'].rolling(252).rank(pct=True)
    regime_df['VIX_percentile_63d'] = regime_df['VIX_proxy'].rolling(63).rank(pct=True)
    
    # Regime classification
    def classify_vix_regime(row):
        if pd.isna(row['VIX_percentile_252d']):
            return 'unknown'
        elif row['VIX_percentile_252d'] < 0.25:
            return 'low_vol'
        elif row['VIX_percentile_252d'] < 0.75:
            return 'medium_vol'
        else:
            return 'high_vol'
    
    regime_df['VIX_regime'] = regime_df.apply(classify_vix_regime, axis=1)
    
    # Numeric regime for ML
    regime_mapping = {'low_vol': 0, 'medium_vol': 1, 'high_vol': 2, 'unknown': -1}
    regime_df['VIX_regime_numeric'] = regime_df['VIX_regime'].map(regime_mapping)
    
    # VIX momentum and persistence
    regime_df['VIX_momentum_5d'] = regime_df['VIX_proxy'].pct_change(5)
    regime_df['VIX_momentum_20d'] = regime_df['VIX_proxy'].pct_change(20)
    regime_df['VIX_expansion'] = (regime_df['VIX_momentum_5d'] > 0.1).astype(int)
    
    # 2. YIELD CURVE REGIME
    print("📈 Creating yield curve regime features...")
    
    # Enhanced yield curve analysis
    regime_df['Yield_2Y_10Y_Spread'] = regime_df['Treasury_10Y'] - regime_df['Treasury_2Y']
    regime_df['Yield_5Y_30Y_Spread'] = regime_df['Treasury_30Y'] - regime_df['Treasury_5Y']
    
    # Curve shape analysis
    regime_df['Curve_steepness'] = regime_df['Yield_2Y_10Y_Spread']
    regime_df['Curve_steepness_change_20d'] = regime_df['Curve_steepness'].diff(20)
    
    def classify_yield_curve_regime(row):
        if pd.isna(row['Curve_steepness']):
            return 'unknown'
        elif row['Curve_steepness'] < -0.5:  # Deeply inverted
            return 'deeply_inverted'
        elif row['Curve_steepness'] < 0:     # Inverted
            return 'inverted'
        elif row['Curve_steepness'] < 1.0:   # Flat
            return 'flat'
        elif row['Curve_steepness'] < 2.0:   # Normal
            return 'normal'
        else:                                 # Steep
            return 'steep'
    
    regime_df['Yield_curve_regime'] = regime_df.apply(classify_yield_curve_regime, axis=1)
    
    # Numeric mapping for yield curve
    curve_mapping = {'deeply_inverted': 0, 'inverted': 1, 'flat': 2, 'normal': 3, 'steep': 4, 'unknown': -1}
    regime_df['Yield_curve_regime_numeric'] = regime_df['Yield_curve_regime'].map(curve_mapping)
    
    # 3. RISK-ON/RISK-OFF COMPOSITE
    print("⚖️ Creating risk-on/off composite features...")
    
    # Individual risk indicators
    regime_df['SPY_vs_TLT_ratio'] = regime_df['SPY_Close'] / regime_df['TLT_Close']
    regime_df['SPY_TLT_momentum_5d'] = regime_df['SPY_vs_TLT_ratio'].pct_change(5)
    
    # Gold vs dollar strength
    regime_df['GLD_vs_DX_ratio'] = regime_df['GLD_Close'] / regime_df['DX_Close']
    regime_df['GLD_DX_momentum_5d'] = regime_df['GLD_vs_DX_ratio'].pct_change(5)
    
    # Credit risk proxy (using yield spreads)
    regime_df['Credit_risk_proxy'] = regime_df['Treasury_10Y'] - regime_df['Treasury_2Y']
    
    # Risk composite score
    # Normalize components to z-scores for combination
    for col in ['SPY_TLT_momentum_5d', 'GLD_DX_momentum_5d', 'VIX_momentum_5d']:
        if col in regime_df.columns:
            mean_val = regime_df[col].rolling(252).mean()
            std_val = regime_df[col].rolling(252).std()
            regime_df[f'{col}_zscore'] = (regime_df[col] - mean_val) / std_val
    
    # Composite risk score (positive = risk-on, negative = risk-off)
    regime_df['Risk_on_off_composite'] = (
        regime_df['SPY_TLT_momentum_5d_zscore'].fillna(0) * 0.4 +
        -regime_df['VIX_momentum_5d_zscore'].fillna(0) * 0.3 +  # Negative because VIX up = risk off
        regime_df['GLD_DX_momentum_5d_zscore'].fillna(0) * 0.3
    )
    
    # Risk regime classification
    def classify_risk_regime(row):
        if pd.isna(row['Risk_on_off_composite']):
            return 'unknown'
        elif row['Risk_on_off_composite'] > 1.0:
            return 'strong_risk_on'
        elif row['Risk_on_off_composite'] > 0.5:
            return 'moderate_risk_on'
        elif row['Risk_on_off_composite'] > -0.5:
            return 'neutral'
        elif row['Risk_on_off_composite'] > -1.0:
            return 'moderate_risk_off'
        else:
            return 'strong_risk_off'
    
    regime_df['Risk_regime'] = regime_df.apply(classify_risk_regime, axis=1)
    
    # Numeric risk regime
    risk_mapping = {
        'strong_risk_off': 0, 'moderate_risk_off': 1, 'neutral': 2, 
        'moderate_risk_on': 3, 'strong_risk_on': 4, 'unknown': -1
    }
    regime_df['Risk_regime_numeric'] = regime_df['Risk_regime'].map(risk_mapping)
    
    # 4. DOLLAR STRENGTH REGIME
    print("💵 Creating dollar strength regime features...")
    
    if 'USD_Strength_Index' in regime_df.columns:
        regime_df['Dollar_strength_percentile'] = regime_df['USD_Strength_Index'].rolling(252).rank(pct=True)
        
        def classify_dollar_regime(row):
            if pd.isna(row['Dollar_strength_percentile']):
                return 'unknown'
            elif row['Dollar_strength_percentile'] < 0.33:
                return 'weak_dollar'
            elif row['Dollar_strength_percentile'] < 0.67:
                return 'neutral_dollar'
            else:
                return 'strong_dollar'
        
        regime_df['Dollar_regime'] = regime_df.apply(classify_dollar_regime, axis=1)
        
        # Numeric dollar regime
        dollar_mapping = {'weak_dollar': 0, 'neutral_dollar': 1, 'strong_dollar': 2, 'unknown': -1}
        regime_df['Dollar_regime_numeric'] = regime_df['Dollar_regime'].map(dollar_mapping)
    
    # 5. TREND REGIME CLASSIFICATION
    print("📊 Creating trend regime features...")
    
    # SPY trend analysis
    regime_df['SPY_SMA_50'] = regime_df['SPY_Close'].rolling(50).mean()
    regime_df['SPY_SMA_200'] = regime_df['SPY_Close'].rolling(200).mean()
    
    regime_df['SPY_vs_SMA50'] = (regime_df['SPY_Close'] / regime_df['SPY_SMA_50'] - 1) * 100
    regime_df['SPY_vs_SMA200'] = (regime_df['SPY_Close'] / regime_df['SPY_SMA_200'] - 1) * 100
    
    # Trend strength
    regime_df['SPY_trend_strength'] = regime_df['SPY_SMA_50'] / regime_df['SPY_SMA_200']
    
    def classify_trend_regime(row):
        if pd.isna(row['SPY_vs_SMA50']) or pd.isna(row['SPY_vs_SMA200']):
            return 'unknown'
        
        if row['SPY_vs_SMA50'] > 2 and row['SPY_vs_SMA200'] > 5:
            return 'strong_bull'
        elif row['SPY_vs_SMA50'] > 0 and row['SPY_vs_SMA200'] > 0:
            return 'bull'
        elif row['SPY_vs_SMA50'] < -2 and row['SPY_vs_SMA200'] < -5:
            return 'strong_bear'
        elif row['SPY_vs_SMA50'] < 0 and row['SPY_vs_SMA200'] < 0:
            return 'bear'
        else:
            return 'sideways'
    
    regime_df['Trend_regime'] = regime_df.apply(classify_trend_regime, axis=1)
    
    # Numeric trend regime
    trend_mapping = {'strong_bear': 0, 'bear': 1, 'sideways': 2, 'bull': 3, 'strong_bull': 4, 'unknown': -1}
    regime_df['Trend_regime_numeric'] = regime_df['Trend_regime'].map(trend_mapping)
    
    # 6. VOLATILITY PERSISTENCE
    print("📈 Creating volatility persistence features...")
    
    # Calculate realized volatility for multiple assets
    for asset in ['SPY', 'QQQ', 'TLT']:
        if f'{asset}_Return_1d' in regime_df.columns:
            regime_df[f'{asset}_realized_vol_20d'] = regime_df[f'{asset}_Return_1d'].rolling(20).std() * np.sqrt(252)
            regime_df[f'{asset}_vol_percentile'] = regime_df[f'{asset}_realized_vol_20d'].rolling(252).rank(pct=True)
    
    # Volatility regime persistence (autocorrelation)
    regime_df['Vol_persistence_SPY'] = regime_df['SPY_realized_vol_20d'].rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
    )
    
    # Cross-asset volatility correlation
    regime_df['Vol_correlation_stocks_bonds'] = regime_df['SPY_realized_vol_20d'].rolling(63).corr(
        regime_df['TLT_realized_vol_20d']
    )
    
    print(f"\n✅ Created {len(regime_df.columns)} regime features")
    
    # Select final regime features for database
    regime_features = [
        'Date', 
        # VIX Regime
        'VIX_regime_numeric', 'VIX_percentile_252d', 'VIX_momentum_5d', 'VIX_expansion',
        # Yield Curve Regime  
        'Yield_curve_regime_numeric', 'Curve_steepness', 'Curve_steepness_change_20d',
        # Risk Regime
        'Risk_regime_numeric', 'Risk_on_off_composite', 'SPY_TLT_momentum_5d',
        # Dollar Regime
        'Dollar_regime_numeric', 'Dollar_strength_percentile',
        # Trend Regime
        'Trend_regime_numeric', 'SPY_vs_SMA50', 'SPY_vs_SMA200', 'SPY_trend_strength',
        # Volatility Features
        'Vol_persistence_SPY', 'Vol_correlation_stocks_bonds', 'SPY_realized_vol_20d'
    ]
    
    # Filter to available columns
    available_features = [col for col in regime_features if col in regime_df.columns]
    final_regime_df = regime_df[available_features].copy()
    
    print(f"📊 Final regime features: {len(available_features)-1}")  # -1 for Date
    
    # Save to database
    print("\n💾 Saving regime features to database...")
    
    # Create regime features table
    final_regime_df.to_sql('regime_features', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("✅ Market regime detection features completed!")
    print(f"📈 Generated {len(available_features)-1} professional regime features")
    
    return final_regime_df

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Starting market regime feature generation at {start_time}")
    
    regime_df = create_market_regime_features()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏱️ Completed in {duration.total_seconds():.1f} seconds")
    
    # Display sample features
    print(f"\n📊 SAMPLE REGIME FEATURES:")
    print("=" * 50)
    recent_data = regime_df.tail(10)
    for col in ['VIX_regime_numeric', 'Yield_curve_regime_numeric', 'Risk_regime_numeric', 'Trend_regime_numeric']:
        if col in recent_data.columns:
            print(f"{col:25s}: {recent_data[col].iloc[-1]}")