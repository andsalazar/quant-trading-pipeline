"""
Professional Cross-Asset Flow Signals
Generate sophisticated cross-asset capital flow detection features
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_cross_asset_flow_features():
    """
    Create professional cross-asset flow signals:
    1. Bond-Equity Rotation Signals
    2. Safe Haven Flow Detection  
    3. Commodity-Currency Alignment
    4. Credit-Equity Divergence
    5. Risk Asset Momentum Breadth
    6. Asset Class Rotation Strength
    """
    
    print("🔄 CREATING CROSS-ASSET FLOW SIGNALS")
    print("=" * 80)
    
    # Connect to database
    conn = sqlite3.connect('quant_trading_final.db')
    
    # Load required data
    print("📊 Loading cross-asset data for flow analysis...")
    
    # Get futures data (multiple asset classes)
    futures_query = """
    SELECT Date, SPY_Close, SPY_Return_1d, SPY_Volatility_5d,
           QQQ_Close, QQQ_Return_1d, QQQ_Volatility_5d,
           TLT_Close, TLT_Return_1d, TY_Close, TY_Return_1d,
           GLD_Close, GLD_Return_1d, CL_Close, CL_Return_1d,
           DX_Close, DX_Return_1d, VXX_Close, VXX_Return_1d,
           HG_Close, HG_Return_1d, NG_Close, NG_Return_1d
    FROM futures_data
    ORDER BY Date
    """
    
    futures_df = pd.read_sql_query(futures_query, conn)
    futures_df['Date'] = pd.to_datetime(futures_df['Date'])
    
    # Get currency data for flow analysis
    currency_query = """
    SELECT date as Date, EUR_USD, USD_JPY, AUD_USD, 
           USD_Strength_Index, Currency_Risk_Sentiment,
           Safe_Haven_Strength, Risk_On_Strength
    FROM currency_data
    ORDER BY date
    """
    
    currency_df = pd.read_sql_query(currency_query, conn)
    currency_df['Date'] = pd.to_datetime(currency_df['Date'])
    
    # Get regime features (dependencies)
    regime_query = """
    SELECT Date, VIX_regime_numeric, Risk_regime_numeric, 
           Trend_regime_numeric, Dollar_regime_numeric
    FROM regime_features
    ORDER BY Date
    """
    
    regime_df = pd.read_sql_query(regime_query, conn)
    regime_df['Date'] = pd.to_datetime(regime_df['Date'])
    
    # Merge all data
    flow_df = futures_df.merge(currency_df, on='Date', how='left')
    flow_df = flow_df.merge(regime_df, on='Date', how='left')
    
    print(f"📈 Processing {len(flow_df)} trading days for flow analysis...")
    
    # 1. BOND-EQUITY ROTATION SIGNALS
    print("\n🔄 Creating bond-equity rotation features...")
    
    # Core rotation ratios
    flow_df['SPY_TLT_ratio'] = flow_df['SPY_Close'] / flow_df['TLT_Close']
    flow_df['QQQ_TLT_ratio'] = flow_df['QQQ_Close'] / flow_df['TLT_Close']
    
    # Rotation momentum (short and medium term)
    flow_df['Bond_equity_rotation_1d'] = flow_df['SPY_TLT_ratio'].pct_change(1)
    flow_df['Bond_equity_rotation_5d'] = flow_df['SPY_TLT_ratio'].pct_change(5)
    flow_df['Bond_equity_rotation_20d'] = flow_df['SPY_TLT_ratio'].pct_change(20)
    
    # Rotation strength (how strong is the move)
    flow_df['Rotation_strength_5d'] = abs(flow_df['Bond_equity_rotation_5d'])
    flow_df['Rotation_persistence'] = (
        flow_df['Bond_equity_rotation_5d'].rolling(5).apply(
            lambda x: (x > 0).sum() if len(x) > 0 else 0
        ) / 5.0
    )
    
    # Historical rotation percentile
    flow_df['Rotation_percentile_252d'] = flow_df['SPY_TLT_ratio'].rolling(252).rank(pct=True)
    
    # Regime-adjusted rotation (stronger signal in certain regimes)
    flow_df['Rotation_regime_adjusted'] = flow_df['Bond_equity_rotation_5d'] * (
        1 + 0.3 * (flow_df['VIX_regime_numeric'] == 2)  # Boost during high vol
    )
    
    # 2. SAFE HAVEN FLOW DETECTION
    print("🏦 Creating safe haven flow features...")
    
    # Core safe haven assets performance
    flow_df['USD_JPY_safe_haven'] = -flow_df['USD_JPY'].pct_change(5)  # JPY strength
    flow_df['Gold_safe_haven'] = flow_df['GLD_Return_1d']
    flow_df['Bond_safe_haven'] = flow_df['TLT_Return_1d']
    
    # Composite safe haven flow (when all move together)
    safe_haven_components = ['USD_JPY_safe_haven', 'Gold_safe_haven', 'Bond_safe_haven']
    
    # Normalize components for combination
    for component in safe_haven_components:
        if component in flow_df.columns:
            mean_val = flow_df[component].rolling(63).mean()
            std_val = flow_df[component].rolling(63).std()
            flow_df[f'{component}_zscore'] = (flow_df[component] - mean_val) / std_val
    
    # Safe haven composite (average z-score)
    flow_df['Safe_haven_flow_composite'] = flow_df[[f'{c}_zscore' for c in safe_haven_components]].mean(axis=1, skipna=True)
    
    # Safe haven flow strength (when flows are synchronized)
    flow_df['Safe_haven_flow_strength'] = flow_df['Safe_haven_flow_composite'].abs()
    
    # Safe haven regime (persistent safe haven flows)
    flow_df['Safe_haven_regime'] = (
        flow_df['Safe_haven_flow_composite'].rolling(10).mean() > 0.5
    ).astype(int)
    
    # 3. COMMODITY-CURRENCY ALIGNMENT
    print("🌾 Creating commodity-currency alignment features...")
    
    # Resource currency performance (AUD is commodity-linked)
    flow_df['AUD_commodity_align'] = flow_df['AUD_USD'].pct_change(5)
    
    # Oil-dollar relationship
    flow_df['Oil_dollar_flow'] = flow_df['CL_Return_1d'] - flow_df['DX_Return_1d']
    flow_df['Oil_dollar_alignment'] = flow_df['Oil_dollar_flow'].rolling(20).mean()
    
    # Gold-dollar inverse relationship
    flow_df['Gold_dollar_divergence'] = flow_df['GLD_Return_1d'] + flow_df['DX_Return_1d']
    flow_df['Gold_dollar_alignment'] = -flow_df['Gold_dollar_divergence'].rolling(20).mean()
    
    # Commodity complex momentum
    commodity_assets = ['CL_Return_1d', 'GLD_Return_1d', 'HG_Return_1d']
    flow_df['Commodity_momentum_breadth'] = flow_df[commodity_assets].apply(
        lambda row: (row > 0).sum() / len(commodity_assets), axis=1
    )
    
    # Resource flow composite
    flow_df['Resource_flow_composite'] = (
        flow_df['AUD_commodity_align'].fillna(0) * 0.3 +
        flow_df['Oil_dollar_alignment'].fillna(0) * 0.4 +
        flow_df['Gold_dollar_alignment'].fillna(0) * 0.3
    )
    
    # 4. CREDIT-EQUITY DIVERGENCE
    print("💳 Creating credit-equity divergence features...")
    
    # Use treasury spreads as credit proxy
    # Credit risk increases when long-term yields rise relative to short-term
    # This is approximated by yield curve steepening
    
    # Equity momentum
    flow_df['Equity_momentum_5d'] = (
        flow_df['SPY_Return_1d'].rolling(5).mean() + 
        flow_df['QQQ_Return_1d'].rolling(5).mean()
    ) / 2
    
    # Credit stress proxy (bond underperformance)
    flow_df['Credit_stress_proxy'] = -flow_df['TLT_Return_1d'].rolling(5).mean()
    
    # Credit-equity divergence (equity up, credit down = divergence)
    flow_df['Credit_equity_divergence'] = (
        flow_df['Equity_momentum_5d'] - flow_df['Credit_stress_proxy']
    )
    
    # Divergence strength
    flow_df['Credit_equity_divergence_strength'] = abs(flow_df['Credit_equity_divergence'])
    
    # Divergence regime (persistent divergence)
    flow_df['Credit_equity_divergence_regime'] = (
        flow_df['Credit_equity_divergence_strength'].rolling(20).mean() > 
        flow_df['Credit_equity_divergence_strength'].rolling(63).quantile(0.75)
    ).astype(int)
    
    # 5. RISK ASSET MOMENTUM BREADTH
    print("📊 Creating risk asset momentum breadth...")
    
    # Define risk assets
    risk_assets = ['SPY_Return_1d', 'QQQ_Return_1d', 'CL_Return_1d', 'HG_Return_1d']
    
    # Daily breadth (% of risk assets with positive returns)
    flow_df['Risk_asset_breadth_1d'] = flow_df[risk_assets].apply(
        lambda row: (row > 0).sum() / len(risk_assets), axis=1
    )
    
    # 5-day breadth  
    flow_df['Risk_asset_breadth_5d'] = flow_df[risk_assets].apply(
        lambda row: (row.rolling(5).mean() > 0).sum() / len(risk_assets), axis=1
    )
    
    # Breadth momentum (improving vs deteriorating breadth)
    flow_df['Risk_breadth_momentum'] = (
        flow_df['Risk_asset_breadth_5d'] - flow_df['Risk_asset_breadth_5d'].shift(5)
    )
    
    # Breadth regime (broad-based vs narrow leadership)
    flow_df['Broad_risk_regime'] = (flow_df['Risk_asset_breadth_5d'] > 0.75).astype(int)
    flow_df['Narrow_risk_regime'] = (flow_df['Risk_asset_breadth_5d'] < 0.25).astype(int)
    
    # 6. ASSET CLASS ROTATION STRENGTH
    print("🔄 Creating asset class rotation strength...")
    
    # Performance differences between asset classes
    flow_df['Stocks_vs_bonds_5d'] = (
        flow_df['SPY_Return_1d'].rolling(5).sum() - 
        flow_df['TLT_Return_1d'].rolling(5).sum()
    )
    
    flow_df['Stocks_vs_commodities_5d'] = (
        flow_df['SPY_Return_1d'].rolling(5).sum() - 
        flow_df['GLD_Return_1d'].rolling(5).sum()
    )
    
    flow_df['Bonds_vs_commodities_5d'] = (
        flow_df['TLT_Return_1d'].rolling(5).sum() - 
        flow_df['GLD_Return_1d'].rolling(5).sum()
    )
    
    # Rotation strength (how much performance is dispersed)
    rotation_components = [
        'Stocks_vs_bonds_5d', 'Stocks_vs_commodities_5d', 'Bonds_vs_commodities_5d'
    ]
    
    flow_df['Asset_rotation_strength'] = flow_df[rotation_components].std(axis=1, skipna=True)
    
    # Rotation persistence (consistent rotation vs random)
    flow_df['Asset_rotation_persistence'] = flow_df['Asset_rotation_strength'].rolling(10).mean()
    
    # Current rotation winner
    def identify_rotation_leader(row):
        stocks = row.get('Stocks_vs_bonds_5d', 0) + row.get('Stocks_vs_commodities_5d', 0)
        bonds = -row.get('Stocks_vs_bonds_5d', 0) + row.get('Bonds_vs_commodities_5d', 0)
        commodities = -row.get('Stocks_vs_commodities_5d', 0) - row.get('Bonds_vs_commodities_5d', 0)
        
        if stocks > bonds and stocks > commodities:
            return 2  # Stocks leading
        elif bonds > stocks and bonds > commodities:
            return 1  # Bonds leading  
        elif commodities > stocks and commodities > bonds:
            return 0  # Commodities leading
        else:
            return -1  # No clear leader
    
    flow_df['Rotation_leader'] = flow_df.apply(identify_rotation_leader, axis=1)
    
    print(f"\n✅ Created {len(flow_df.columns)} cross-asset flow features")
    
    # Select final cross-asset flow features
    flow_features = [
        'Date',
        # Bond-Equity Rotation
        'Bond_equity_rotation_5d', 'Bond_equity_rotation_20d', 'Rotation_strength_5d',
        'Rotation_persistence', 'Rotation_percentile_252d', 'Rotation_regime_adjusted',
        # Safe Haven Flows
        'Safe_haven_flow_composite', 'Safe_haven_flow_strength', 'Safe_haven_regime',
        # Commodity-Currency Alignment
        'Oil_dollar_alignment', 'Gold_dollar_alignment', 'Commodity_momentum_breadth',
        'Resource_flow_composite',
        # Credit-Equity Divergence
        'Credit_equity_divergence', 'Credit_equity_divergence_strength', 'Credit_equity_divergence_regime',
        # Risk Asset Breadth
        'Risk_asset_breadth_1d', 'Risk_asset_breadth_5d', 'Risk_breadth_momentum',
        'Broad_risk_regime', 'Narrow_risk_regime',
        # Asset Class Rotation
        'Stocks_vs_bonds_5d', 'Asset_rotation_strength', 'Asset_rotation_persistence',
        'Rotation_leader'
    ]
    
    # Filter to available columns
    available_features = [col for col in flow_features if col in flow_df.columns]
    final_flow_df = flow_df[available_features].copy()
    
    print(f"📊 Final cross-asset flow features: {len(available_features)-1}")  # -1 for Date
    
    # Save to database
    print("\n💾 Saving cross-asset flow features to database...")
    
    final_flow_df.to_sql('cross_asset_flows', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("✅ Cross-asset flow signals completed!")
    print(f"📈 Generated {len(available_features)-1} professional flow features")
    
    return final_flow_df

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Starting cross-asset flow feature generation at {start_time}")
    
    flow_df = create_cross_asset_flow_features()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏱️ Completed in {duration.total_seconds():.1f} seconds")
    
    # Display sample features
    print(f"\n📊 SAMPLE CROSS-ASSET FLOW FEATURES:")
    print("=" * 60)
    recent_data = flow_df.tail(5)
    key_features = [
        'Bond_equity_rotation_5d', 'Safe_haven_flow_composite', 
        'Risk_asset_breadth_5d', 'Asset_rotation_strength'
    ]
    
    for feature in key_features:
        if feature in recent_data.columns:
            latest_value = recent_data[feature].iloc[-1]
            print(f"{feature:30s}: {latest_value:8.4f}")
            
    # Update daily pipeline tracking
    print(f"\n📋 DAILY UPDATE PIPELINE STATUS:")
    print("✅ Market Regime Features - COMPLETED")  
    print("✅ Cross-Asset Flow Signals - COMPLETED")
    print("⏳ Sentiment Divergence Signals - PENDING")
    print("⏳ Momentum Quality Factors - PENDING")