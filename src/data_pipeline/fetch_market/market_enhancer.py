#!/usr/bin/env python3
"""
SIMPLIFIED MARKET DATA PROCESSOR
Reliable technical indicators without complex dependencies

INPUT FILE (hardcoded):
<project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_01_fetch_market\\00_01_market_polygon_converted.csv

OUTPUT FILES (hardcoded):
1. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_01_fetch_market\\00_01_market_enhanced_long.csv
   - Original OHLCV + technical indicators in LONG format
2. <project_root>neDrive\\Documents\\QuantTradingProject\\#_Core_Project\\#_fetch_data\\#_01_fetch_market\\00_01_market_features.csv
   - Market-wide features for database integration

Strategy:
- Focus on essential, reliable technical indicators
- Manual calculations to avoid library dependency issues
- Proper long format output for database integration
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class SimpleMarketProcessor:
    """Simplified but reliable market data enhancement"""
    
    def __init__(self):
        """Initialize with robust processing strategy"""
        # Hardcoded file paths
        self.input_csv = CONFIG_BASE_PATH  # Set in config.py
        self.enhanced_csv = CONFIG_BASE_PATH  # Set in config.py
        self.market_features_csv = CONFIG_BASE_PATH  # Set in config.py
        
        # Key symbols for market features
        self.key_symbols = ['SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'TLT', 'VXX', 'USO']
        
        print("🚀 Simplified Market Data Processor")
        print("📈 Processing: OHLCV → Essential Technical Indicators")
        print("💡 Reliable feature engineering for database integration")
        print(f"📂 Input file: {self.input_csv}")
        print(f"💾 Output file 1: {self.enhanced_csv}")
        print(f"💾 Output file 2: {self.market_features_csv}")

    def load_market_data(self):
        """Load and validate market data"""
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"❌ Market data file not found: {self.input_csv}")
        
        print("📂 Loading market data...")
        df = pd.read_csv(self.input_csv, parse_dates=['Date'])
        print(f"   ✅ Loaded {len(df):,} records")
        print(f"   📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"   🎯 Symbols: {df['Symbol'].nunique()}")
        
        return df

    def calculate_technical_indicators(self, group_df):
        """Calculate essential technical indicators for a symbol"""
        try:
            # Ensure proper sorting
            group_df = group_df.sort_values('Date').copy()
            
            # Basic price metrics
            group_df['Daily_Return'] = group_df['Close'].pct_change()
            group_df['Overnight_Gap'] = (group_df['Open'] / group_df['Close'].shift(1) - 1) * 100
            group_df['Daily_Range'] = ((group_df['High'] - group_df['Low']) / group_df['Close']) * 100
            
            # Moving averages
            group_df['SMA_5'] = group_df['Close'].rolling(window=5).mean()
            group_df['SMA_10'] = group_df['Close'].rolling(window=10).mean()
            group_df['SMA_20'] = group_df['Close'].rolling(window=20).mean()
            group_df['SMA_50'] = group_df['Close'].rolling(window=50).mean()
            group_df['SMA_200'] = group_df['Close'].rolling(window=200).mean()
            
            # Price position vs moving averages
            group_df['Price_vs_SMA20'] = ((group_df['Close'] / group_df['SMA_20']) - 1) * 100
            group_df['Price_vs_SMA50'] = ((group_df['Close'] / group_df['SMA_50']) - 1) * 100
            
            # Momentum indicators
            group_df['RSI_14'] = self.calculate_rsi(group_df['Close'], 14)
            group_df['Price_Change_5d'] = ((group_df['Close'] / group_df['Close'].shift(5)) - 1) * 100
            group_df['Price_Change_20d'] = ((group_df['Close'] / group_df['Close'].shift(20)) - 1) * 100
            
            # Volatility
            group_df['Volatility_20d'] = group_df['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
            group_df['ATR_14'] = self.calculate_atr(group_df, 14)
            
            # Volume indicators
            if group_df['Volume'].sum() > 0:
                group_df['Volume_SMA_20'] = group_df['Volume'].rolling(window=20).mean()
                group_df['Volume_Ratio'] = group_df['Volume'] / group_df['Volume_SMA_20']
                group_df['OBV'] = (group_df['Volume'] * np.sign(group_df['Daily_Return'])).cumsum()
            else:
                group_df['Volume_SMA_20'] = 0
                group_df['Volume_Ratio'] = 1
                group_df['OBV'] = 0
                
            # VWAP calculation
            if 'VWAP' not in group_df.columns or group_df['VWAP'].isna().all():
                if group_df['Volume'].sum() > 0:
                    vwap_num = (group_df['Close'] * group_df['Volume']).cumsum()
                    vwap_den = group_df['Volume'].cumsum()
                    group_df['VWAP'] = vwap_num / vwap_den
                    group_df['Price_vs_VWAP'] = ((group_df['Close'] / group_df['VWAP']) - 1) * 100
                else:
                    group_df['VWAP'] = group_df['Close']
                    group_df['Price_vs_VWAP'] = 0
            else:
                group_df['Price_vs_VWAP'] = ((group_df['Close'] / group_df['VWAP']) - 1) * 100
            
            # Bollinger Bands
            bb_mid = group_df['Close'].rolling(window=20).mean()
            bb_std = group_df['Close'].rolling(window=20).std()
            group_df['BB_Upper'] = bb_mid + (bb_std * 2)
            group_df['BB_Lower'] = bb_mid - (bb_std * 2)
            group_df['BB_Position'] = ((group_df['Close'] - group_df['BB_Lower']) / 
                                      (group_df['BB_Upper'] - group_df['BB_Lower'])) * 100
            
            # Trend strength
            group_df['Trend_Strength'] = np.where(
                group_df['Close'] > group_df['SMA_20'], 1,
                np.where(group_df['Close'] < group_df['SMA_20'], -1, 0)
            )
            
            return group_df
            
        except Exception as e:
            print(f"      ⚠️ Error processing: {str(e)[:50]}...")
            return group_df

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift(1))
        low_close_prev = abs(df['Low'] - df['Close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def create_market_features(self, enhanced_df):
        """Create market-wide features for database integration"""
        print("🌍 Creating market-wide features...")
        
        market_features = []
        
        # Get date range
        dates = enhanced_df['Date'].unique()
        
        for date in dates:
            date_data = enhanced_df[enhanced_df['Date'] == date].copy()
            
            if len(date_data) == 0:
                continue
                
            features = {
                'Date': date,
                'Market_Return_Mean': date_data['Daily_Return'].mean() if 'Daily_Return' in date_data.columns else 0,
                'Market_Return_Std': date_data['Daily_Return'].std() if 'Daily_Return' in date_data.columns else 0,
                'Market_Volume_Total': date_data['Volume'].sum(),
                'Stocks_Above_SMA20': (date_data['Price_vs_SMA20'] > 0).sum() if 'Price_vs_SMA20' in date_data.columns else 0,
                'Stocks_Below_SMA20': (date_data['Price_vs_SMA20'] < 0).sum() if 'Price_vs_SMA20' in date_data.columns else 0,
                'Average_RSI': date_data['RSI_14'].mean() if 'RSI_14' in date_data.columns else 50,
                'High_RSI_Count': (date_data['RSI_14'] > 70).sum() if 'RSI_14' in date_data.columns else 0,
                'Low_RSI_Count': (date_data['RSI_14'] < 30).sum() if 'RSI_14' in date_data.columns else 0,
                'Average_Volatility': date_data['Volatility_20d'].mean() if 'Volatility_20d' in date_data.columns else 0
            }
            
            # Add key symbol features
            for symbol in self.key_symbols:
                symbol_data = date_data[date_data['Symbol'] == symbol]
                if len(symbol_data) > 0:
                    features[f'{symbol}_Close'] = symbol_data['Close'].iloc[0]
                    features[f'{symbol}_Return'] = symbol_data['Daily_Return'].iloc[0] if 'Daily_Return' in symbol_data.columns else 0
                    features[f'{symbol}_Volume'] = symbol_data['Volume'].iloc[0]
                else:
                    features[f'{symbol}_Close'] = None
                    features[f'{symbol}_Return'] = None
                    features[f'{symbol}_Volume'] = None
            
            market_features.append(features)
        
        return pd.DataFrame(market_features)

    def run_enhancement(self):
        """Run the complete enhancement process"""
        try:
            print("🚀 MARKET DATA ENHANCEMENT - Simple & Reliable Processing")
            print("=" * 80)
            
            # Load data
            df = self.load_market_data()
            
            # Process by symbol
            print("🔧 Processing technical indicators by symbol...")
            enhanced_data = []
            
            symbols = df['Symbol'].unique()
            for i, symbol in enumerate(symbols, 1):
                if i % 50 == 0:
                    print(f"      Processing: {symbol} [{i}/{len(symbols)}]")
                
                symbol_data = df[df['Symbol'] == symbol].copy()
                enhanced_symbol = self.calculate_technical_indicators(symbol_data)
                enhanced_data.append(enhanced_symbol)
            
            # Combine all enhanced data
            print("🔄 Combining enhanced data...")
            enhanced_df = pd.concat(enhanced_data, ignore_index=True)
            
            # Create market features
            market_features = self.create_market_features(enhanced_df)
            
            # Save outputs
            print("💾 Saving enhanced data...")
            enhanced_df.to_csv(self.enhanced_csv, index=False)
            print(f"   ✅ Enhanced file saved: {self.enhanced_csv}")
            
            if len(market_features) > 0:
                market_features.to_csv(self.market_features_csv, index=False)
                print(f"   ✅ Market features saved: {self.market_features_csv}")
            
            # Summary
            print("\n✅ MARKET DATA ENHANCEMENT COMPLETE!")
            print(f"   📊 Enhanced records: {len(enhanced_df):,}")
            print(f"   📅 Date range: {enhanced_df['Date'].min().date()} to {enhanced_df['Date'].max().date()}")
            print(f"   🎯 Symbols processed: {enhanced_df['Symbol'].nunique()}")
            print(f"   📈 Technical indicators: {len([col for col in enhanced_df.columns if col not in ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Transactions']])}")
            print(f"   🌍 Market features: {len(market_features)} dates")
            print("\n🎯 Files ready for database integration!")
            
        except Exception as e:
            print(f"❌ Error in market enhancement: {str(e)}")
            raise

def main():
    """Main execution function"""
    processor = SimpleMarketProcessor()
    processor.run_enhancement()

if __name__ == "__main__":
    main()