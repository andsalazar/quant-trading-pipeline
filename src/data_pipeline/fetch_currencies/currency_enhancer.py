#!/usr/bin/env python3
"""
ENHANCED CURRENCY FEATURES GENERATOR
Builds on existing wide format currency data with calculated features

INPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_02_fetch_currencies\\00_02_currency_polygon_clean.csv

OUTPUT FILES (hardcoded):
1. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_02_fetch_currencies\\00_02_currency_enhanced.csv
   - Original + calculated features (database ready)
2. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_02_fetch_currencies\\00_02_currency_market_features.csv
   - Market-wide currency features

Strategy:
- Currency volatility and momentum indicators
- Dollar strength index and cross-currency signals
- Risk-on/risk-off currency indicators
- Central bank policy divergence signals
- Date column standardized to "Date" (capitalized) to match market data

Enhanced Features:
- Daily currency returns and volatility
- Dollar strength vs basket of currencies
- Risk sentiment indicators (JPY, CHF strength)
- Carry trade signals (AUD, NZD)
- Major currency pair momentum
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedCurrencyProcessor:
    """Enhanced currency features following established pattern"""
    
    def __init__(self):
        """Initialize with comprehensive currency strategy"""
        # Hardcoded file paths
        self.input_csv = CONFIG_BASE_PATH  # Set in config.py
        self.enhanced_csv = CONFIG_BASE_PATH  # Set in config.py
        self.market_features_csv = CONFIG_BASE_PATH  # Set in config.py
        
        # Create backup path
        backup_dir = os.path.dirname(self.input_csv)
        self.backup_path = os.path.join(backup_dir, f"00_02_currency_backup_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # Currency pair configurations
        self.currency_pairs = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD',
            'AUD_USD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY'
        ]
        
        # Major currencies for basket calculations
        self.major_usd_pairs = ['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD']  # USD is quote currency
        self.usd_base_pairs = ['USD_JPY', 'USD_CHF', 'USD_CAD']  # USD is base currency
        
        # Risk sentiment currencies
        self.safe_haven_pairs = ['USD_JPY', 'USD_CHF']  # Higher = USD stronger (risk-off)
        self.risk_on_pairs = ['AUD_USD', 'NZD_USD']     # Higher = risk-on
        
        print("🚀 Enhanced Currency Features Generator")
        print("💱 Processing: Major currency pairs → Market features")
        print("💡 Enhanced wide format → Database-ready features")
        print(f"📂 Input file: {self.input_csv}")
        print(f"💾 Output file 1: {self.enhanced_csv}")
        print(f"💾 Output file 2: {self.market_features_csv}")
    
    def load_and_validate_data(self):
        """Load currency data with validation"""
        print(f"📂 Loading currency data...")
        
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"❌ Currency file not found: {self.input_csv}")
        
        df = pd.read_csv(self.input_csv)
        
        # Standardize date column to "Date" (capitalized) to match market data
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
            print("   ✅ Standardized 'date' column to 'Date' (matches market data format)")
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create backup
        df.to_csv(self.backup_path, index=False)
        print(f"   💾 Backup created")
        
        # Validate currency pairs
        missing_pairs = [pair for pair in self.currency_pairs if pair not in df.columns]
        if missing_pairs:
            print(f"   ⚠️ Missing currency pairs: {missing_pairs}")
        
        available_pairs = [pair for pair in self.currency_pairs if pair in df.columns]
        
        print(f"   ✅ Loaded {len(df):,} records")
        print(f"   📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"   💱 Available pairs: {len(available_pairs)}")
        
        return df, available_pairs
    
    def calculate_currency_returns_and_volatility(self, df, pairs):
        """Calculate returns and volatility for all currency pairs"""
        print("📈 Calculating currency returns and volatility...")
        
        for pair in pairs:
            if pair in df.columns:
                # Daily returns (log returns for better properties)
                df[f'{pair}_return'] = np.log(df[pair] / df[pair].shift(1))
                
                # Rolling volatility (5-day and 20-day)
                df[f'{pair}_vol_5d'] = df[f'{pair}_return'].rolling(5, min_periods=2).std() * np.sqrt(252)  # Annualized
                df[f'{pair}_vol_20d'] = df[f'{pair}_return'].rolling(20, min_periods=5).std() * np.sqrt(252)
                
                # Price momentum (5-day and 20-day)
                df[f'{pair}_momentum_5d'] = (df[pair] / df[pair].shift(5) - 1) * 100
                df[f'{pair}_momentum_20d'] = (df[pair] / df[pair].shift(20) - 1) * 100
                
                # Relative strength (vs 60-day average)
                df[f'{pair}_ma_60d'] = df[pair].rolling(60, min_periods=10).mean()
                df[f'{pair}_vs_ma60'] = (df[pair] / df[f'{pair}_ma_60d'] - 1) * 100
        
        print(f"   ✅ Calculated returns and volatility for {len(pairs)} pairs")
        return df
    
    def calculate_dollar_strength_index(self, df):
        """Calculate Dollar Strength Index (DXY-like)"""
        print("💪 Calculating Dollar Strength Index...")
        
        # Create USD strength components
        usd_components = []
        
        # For USD base pairs (USD_JPY, USD_CHF, USD_CAD) - higher = stronger USD
        for pair in self.usd_base_pairs:
            if pair in df.columns:
                # Normalize to percentage change from start
                base_value = df[pair].iloc[0]
                normalized = (df[pair] / base_value - 1) * 100
                usd_components.append(normalized)
        
        # For USD quote pairs (EUR_USD, GBP_USD, etc.) - lower = stronger USD
        for pair in self.major_usd_pairs:
            if pair in df.columns:
                # Invert because lower value = stronger USD
                base_value = df[pair].iloc[0]
                normalized = -((df[pair] / base_value - 1) * 100)
                usd_components.append(normalized)
        
        if usd_components:
            # Average all components
            df['USD_Strength_Index'] = pd.concat(usd_components, axis=1).mean(axis=1)
            
            # USD strength momentum
            df['USD_Strength_5d'] = df['USD_Strength_Index'].diff(5)
            df['USD_Strength_20d'] = df['USD_Strength_Index'].diff(20)
            
            # USD strength volatility
            df['USD_Strength_Vol_20d'] = df['USD_Strength_Index'].rolling(20, min_periods=5).std()
            
            print(f"   ✅ USD Strength Index created")
        else:
            print(f"   ⚠️ Insufficient data for USD Strength Index")
        
        return df
    
    def calculate_risk_sentiment_indicators(self, df):
        """Calculate risk-on/risk-off indicators from currencies"""
        print("🎯 Calculating risk sentiment indicators...")
        
        risk_indicators = []
        
        # Safe haven strength (JPY, CHF)
        safe_haven_strength = []
        for pair in self.safe_haven_pairs:
            if pair in df.columns:
                # Higher USD_JPY, USD_CHF = stronger safe havens (risk-off)
                base_value = df[pair].iloc[0]
                strength = (df[pair] / base_value - 1) * 100
                safe_haven_strength.append(strength)
        
        if safe_haven_strength:
            df['Safe_Haven_Strength'] = pd.concat(safe_haven_strength, axis=1).mean(axis=1)
            risk_indicators.append('Safe_Haven_Strength')
        
        # Risk-on currencies strength (AUD, NZD)
        risk_on_strength = []
        for pair in self.risk_on_pairs:
            if pair in df.columns:
                # Higher AUD_USD, NZD_USD = risk-on
                base_value = df[pair].iloc[0]
                strength = (df[pair] / base_value - 1) * 100
                risk_on_strength.append(strength)
        
        if risk_on_strength:
            df['Risk_On_Strength'] = pd.concat(risk_on_strength, axis=1).mean(axis=1)
            risk_indicators.append('Risk_On_Strength')
        
        # Overall risk sentiment (Risk-on minus Safe-haven)
        if 'Risk_On_Strength' in df.columns and 'Safe_Haven_Strength' in df.columns:
            df['Currency_Risk_Sentiment'] = df['Risk_On_Strength'] - df['Safe_Haven_Strength']
            
            # Risk sentiment momentum
            df['Risk_Sentiment_5d'] = df['Currency_Risk_Sentiment'].diff(5)
            df['Risk_Sentiment_20d'] = df['Currency_Risk_Sentiment'].diff(20)
            
            # Risk regime indicator (positive = risk-on, negative = risk-off)
            df['Risk_On_Regime'] = (df['Currency_Risk_Sentiment'] > 0).astype(int)
            
            print(f"   ✅ Risk sentiment indicators created")
        
        return df
    
    def calculate_carry_trade_indicators(self, df):
        """Calculate carry trade strength indicators"""
        print("💰 Calculating carry trade indicators...")
        
        # Carry trade currencies (typically high-yield)
        carry_pairs = ['AUD_USD', 'NZD_USD']  # Traditional carry currencies
        funding_pairs = ['USD_JPY', 'USD_CHF']  # Traditional funding currencies
        
        carry_strength = []
        
        # Carry currency strength
        for pair in carry_pairs:
            if pair in df.columns:
                base_value = df[pair].iloc[0]
                strength = (df[pair] / base_value - 1) * 100
                carry_strength.append(strength)
        
        if carry_strength:
            df['Carry_Trade_Strength'] = pd.concat(carry_strength, axis=1).mean(axis=1)
            
            # Carry trade momentum
            df['Carry_Trade_5d'] = df['Carry_Trade_Strength'].diff(5)
            df['Carry_Trade_20d'] = df['Carry_Trade_Strength'].diff(20)
            
            # Carry trade volatility (risk measure)
            df['Carry_Trade_Vol_20d'] = df['Carry_Trade_Strength'].rolling(20, min_periods=5).std()
            
            print(f"   ✅ Carry trade indicators created")
        
        return df
    
    def calculate_cross_currency_signals(self, df):
        """Calculate cross-currency signals"""
        print("🌐 Calculating cross-currency signals...")
        
        # EUR strength vs other majors
        if 'EUR_USD' in df.columns and 'EUR_GBP' in df.columns and 'EUR_JPY' in df.columns:
            eur_components = []
            
            # EUR vs USD
            eur_components.append((df['EUR_USD'] / df['EUR_USD'].iloc[0] - 1) * 100)
            
            # EUR vs GBP
            eur_components.append((df['EUR_GBP'] / df['EUR_GBP'].iloc[0] - 1) * 100)
            
            # EUR vs JPY (normalize)
            eur_components.append((df['EUR_JPY'] / df['EUR_JPY'].iloc[0] - 1) * 100)
            
            df['EUR_Strength_Index'] = pd.concat(eur_components, axis=1).mean(axis=1)
            df['EUR_Strength_5d'] = df['EUR_Strength_Index'].diff(5)
        
        # GBP strength
        if 'GBP_USD' in df.columns and 'EUR_GBP' in df.columns and 'GBP_JPY' in df.columns:
            gbp_components = []
            
            # GBP vs USD
            gbp_components.append((df['GBP_USD'] / df['GBP_USD'].iloc[0] - 1) * 100)
            
            # GBP vs EUR (invert EUR_GBP)
            gbp_components.append(-((df['EUR_GBP'] / df['EUR_GBP'].iloc[0] - 1) * 100))
            
            # GBP vs JPY
            gbp_components.append((df['GBP_JPY'] / df['GBP_JPY'].iloc[0] - 1) * 100)
            
            df['GBP_Strength_Index'] = pd.concat(gbp_components, axis=1).mean(axis=1)
            df['GBP_Strength_5d'] = df['GBP_Strength_Index'].diff(5)
        
        print(f"   ✅ Cross-currency signals created")
        return df
    
    def create_market_wide_features(self, df):
        """Create market-wide currency features"""
        print("🌍 Creating market-wide currency features...")
        
        # Market-wide currency volatility
        volatility_cols = [col for col in df.columns if '_vol_5d' in col]
        if volatility_cols:
            df['Currency_Market_Vol_Avg'] = df[volatility_cols].mean(axis=1)
            df['Currency_Market_Vol_Max'] = df[volatility_cols].max(axis=1)
        
        # Market-wide currency momentum
        momentum_cols = [col for col in df.columns if '_momentum_5d' in col]
        if momentum_cols:
            df['Currency_Market_Momentum_Avg'] = df[momentum_cols].mean(axis=1)
            df['Currency_Market_Momentum_Dispersion'] = df[momentum_cols].std(axis=1)
        
        # Currency correlation breakdown indicator
        # When currencies move together = risk-off, when dispersed = normal markets
        if len(momentum_cols) > 3:
            df['Currency_Correlation_Regime'] = (df['Currency_Market_Momentum_Dispersion'] < df['Currency_Market_Momentum_Dispersion'].rolling(20, min_periods=5).mean()).astype(int)
        
        print(f"   ✅ Market-wide currency features created")
        return df
    
    def save_enhanced_outputs(self, df):
        """Save enhanced currency data"""
        print("💾 Saving enhanced currency outputs...")
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        # Sort by Date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Save enhanced version (all features)
        df.to_csv(self.enhanced_csv, index=False)
        
        # Create market features subset (most important for database merge)
        market_features_cols = ['Date']
        
        # Core currency levels (original data)
        market_features_cols.extend(self.currency_pairs)
        
        # Key calculated features
        key_features = [
            'USD_Strength_Index', 'USD_Strength_5d', 'USD_Strength_Vol_20d',
            'Currency_Risk_Sentiment', 'Risk_Sentiment_5d', 'Risk_On_Regime',
            'Carry_Trade_Strength', 'Carry_Trade_5d', 'Carry_Trade_Vol_20d',
            'Currency_Market_Vol_Avg', 'Currency_Market_Momentum_Avg',
            'Safe_Haven_Strength', 'Risk_On_Strength'
        ]
        
        # Add features that exist
        for feature in key_features:
            if feature in df.columns:
                market_features_cols.append(feature)
        
        market_df = df[market_features_cols].copy()
        market_df.to_csv(self.market_features_csv, index=False)
        
        print(f"   ✅ Enhanced file saved: {self.enhanced_csv}")
        print(f"      {len(df):,} records × {len(df.columns)} features")
        print(f"   ✅ Market features saved: {self.market_features_csv}")
        print(f"      {len(market_df):,} records × {len(market_df.columns)} features")
        
        return {
            'enhanced': len(df.columns),
            'market': len(market_df.columns),
            'records': len(df),
            'date_range': f"{df['Date'].min().date()} to {df['Date'].max().date()}"
        }
    
    def run_enhancement(self):
        """Main currency enhancement workflow"""
        print("🚀 CURRENCY ENHANCEMENT - Enhanced Wide Format")
        print("=" * 60)
        
        try:
            # Load and validate
            df, available_pairs = self.load_and_validate_data()
            
            # Calculate returns and volatility
            df = self.calculate_currency_returns_and_volatility(df, available_pairs)
            
            # Dollar strength index
            df = self.calculate_dollar_strength_index(df)
            
            # Risk sentiment indicators
            df = self.calculate_risk_sentiment_indicators(df)
            
            # Carry trade indicators
            df = self.calculate_carry_trade_indicators(df)
            
            # Cross-currency signals
            df = self.calculate_cross_currency_signals(df)
            
            # Market-wide features
            df = self.create_market_wide_features(df)
            
            # Save outputs
            summary = self.save_enhanced_outputs(df)
            
            print(f"\n🎯 CURRENCY ENHANCEMENT COMPLETE!")
            print(f"   📊 Enhanced features: {summary['enhanced']} total columns")
            print(f"   🗄️ Market features: {summary['market']} key columns")
            print(f"   📈 Records processed: {summary['records']:,}")
            print(f"   📅 Date range: {summary['date_range']}")
            print(f"\n✅ Ready for SQLite database integration!")
            print(f"   💡 Wide format perfect for easy merge on 'Date' column (standardized)")
            print(f"   💱 Enhanced with volatility, momentum, and risk indicators")
            
        except Exception as e:
            print(f"❌ Error in currency enhancement: {str(e)}")
            raise

def main():
    """Enhanced currency processing - single command"""
    print("🎯 ENHANCED CURRENCY FEATURES GENERATOR")
    print("   Original wide format + calculated features → Database-ready")
    print("   Volatility, momentum, USD strength, risk sentiment indicators")
    print()
    
    processor = EnhancedCurrencyProcessor()
    processor.run_enhancement()

if __name__ == "__main__":
    main()