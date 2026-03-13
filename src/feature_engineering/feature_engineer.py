#!/usr/bin/env python3
"""
STEP 9: RENAISSANCE-LEVEL FEATURE ENGINEERING
==============================================

Purpose:
- LEFT JOIN all 10 database tables
- Generate 400-600 raw ML features
- Create ticker-level, market-level, and cross-dataset features
- Output raw features (normalization happens in Step 10)

Strategy:
- Ticker-Level Features (~200): Momentum, volatility, technicals, sentiment integration
- Market-Level Features (~100): Breadth, currencies, yields, futures, options, macro
- Cross-Dataset Features (~100): Sentiment-price divergence, vol arbitrage, factor orthogonalization

Output:
- ml_features_master.csv (~1.4M rows × ~400 features)
- Features in original units (prices, ratios, percentages)
- Ready for normalization and model training (Step 10)

Created: October 16, 2025
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class RenaissanceFeatureEngineer:
    """Generate institutional-grade ML features from normalized database"""
    
    def __init__(self):
        """Initialize with database paths"""
        self.base_path = CONFIG_BASE_PATH  # Set in config.py
        self.db_path = os.path.join(self.base_path, r'#_master_database\quant_trading_v2.db')
        self.output_csv = os.path.join(self.base_path, r'#_feature_engineering\ml_features_master.csv')
        
        print("=" * 80)
        print("STEP 9: RENAISSANCE-LEVEL FEATURE ENGINEERING")
        print("=" * 80)
        print(f"Database: {os.path.basename(self.db_path)}")
        print(f"Output: {os.path.basename(self.output_csv)}")
        print("Strategy: Raw features (normalization in Step 10)")
        print("=" * 80)
        
        # Check database exists
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    def load_database_tables(self):
        """Load all tables from database"""
        print("\n[1/6] Loading database tables...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load all tables
        self.ticker_universe = pd.read_sql("SELECT * FROM ticker_universe", conn)
        self.market_data = pd.read_sql("SELECT * FROM market_data", conn, parse_dates=['date'])
        self.market_features = pd.read_sql("SELECT * FROM market_features", conn, parse_dates=['date'])
        self.currency_data = pd.read_sql("SELECT * FROM currency_data", conn, parse_dates=['date'])
        self.treasury_data = pd.read_sql("SELECT * FROM treasury_data", conn, parse_dates=['date'])
        
        # Load futures and options, then deduplicate by date (keep first)
        self.futures_features = pd.read_sql("SELECT * FROM futures_features", conn, parse_dates=['date'])
        self.futures_features = self.futures_features.drop_duplicates(subset=['date'], keep='first')
        
        self.options_features = pd.read_sql("SELECT * FROM options_features", conn, parse_dates=['date'])
        self.options_features = self.options_features.drop_duplicates(subset=['date'], keep='first')
        
        self.events_data = pd.read_sql("SELECT * FROM events_data", conn, parse_dates=['date'])
        self.sentiment_data = pd.read_sql("SELECT * FROM sentiment_data", conn, parse_dates=['date'])
        self.sentiment_features = pd.read_sql("SELECT * FROM sentiment_features", conn, parse_dates=['date'])
        
        conn.close()
        
        print(f"  ✓ ticker_universe: {len(self.ticker_universe):,} symbols")
        print(f"  ✓ market_data: {len(self.market_data):,} records (ticker-level)")
        print(f"  ✓ market_features: {len(self.market_features):,} dates (market-level)")
        print(f"  ✓ currency_data: {len(self.currency_data):,} dates")
        print(f"  ✓ treasury_data: {len(self.treasury_data):,} dates")
        print(f"  ✓ futures_features: {len(self.futures_features):,} dates (deduplicated)")
        print(f"  ✓ options_features: {len(self.options_features):,} dates (deduplicated)")
        print(f"  ✓ events_data: {len(self.events_data):,} dates")
        print(f"  ✓ sentiment_data: {len(self.sentiment_data):,} records (ticker-level)")
        print(f"  ✓ sentiment_features: {len(self.sentiment_features):,} dates (market-level)")
    
    def create_base_dataframe(self):
        """Create base dataframe from market_data (ticker-level)"""
        print("\n[2/6] Creating base dataframe...")
        
        # Start with market_data (ticker × date grid)
        self.df = self.market_data.copy()
        
        # Ensure date column is datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"  ✓ Base dataframe: {len(self.df):,} rows (date × symbol)")
        print(f"  ✓ Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"  ✓ Unique symbols: {self.df['symbol'].nunique()}")
    
    def add_ticker_level_features(self):
        """Add ticker-level features (momentum, volatility, technicals)"""
        print("\n[3/6] Generating ticker-level features...")
        
        # Sort for rolling calculations
        self.df = self.df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        feature_count = len(self.df.columns)
        
        print("  → Multi-timeframe momentum...")
        # Price momentum (multiple windows)
        for window in [5, 10, 20, 60, 120, 252]:
            self.df[f'return_{window}d'] = self.df.groupby('symbol')['Close'].pct_change(window)
        
        print("  → Volatility regimes...")
        # Volatility (realized)
        for window in [5, 20, 60]:
            self.df[f'volatility_{window}d'] = self.df.groupby('symbol')['Close'].pct_change().rolling(window).std().reset_index(level=0, drop=True) * np.sqrt(252)
        
        # Volatility of volatility
        self.df['vol_of_vol_20d'] = self.df.groupby('symbol')['volatility_20d'].rolling(20).std().reset_index(level=0, drop=True)
        
        print("  → Technical indicators...")
        # RSI (already in market_data if from enhancer)
        # Bollinger Band position
        for window in [20, 60]:
            rolling_mean = self.df.groupby('symbol')['Close'].rolling(window).mean().reset_index(level=0, drop=True)
            rolling_std = self.df.groupby('symbol')['Close'].rolling(window).std().reset_index(level=0, drop=True)
            self.df[f'bb_position_{window}d'] = (self.df['Close'] - rolling_mean) / (2 * rolling_std)
        
        print("  → Volume analysis...")
        # Volume trends
        self.df['volume_ratio_20d'] = self.df.groupby('symbol')['Volume'].transform(
            lambda x: x / x.rolling(20, min_periods=1).mean()
        )
        
        # Price-volume correlation
        self.df['price_volume_corr_20d'] = self.df.groupby('symbol').apply(
            lambda x: x['Close'].pct_change().rolling(20).corr(x['Volume'].pct_change())
        ).reset_index(level=0, drop=True)
        
        print("  → Momentum quality...")
        # Sharpe ratio (rolling)
        for window in [20, 60]:
            returns = self.df.groupby('symbol')['Close'].pct_change()
            mean_ret = returns.rolling(window).mean().reset_index(level=0, drop=True)
            std_ret = returns.rolling(window).std().reset_index(level=0, drop=True)
            self.df[f'sharpe_{window}d'] = mean_ret / std_ret * np.sqrt(252)
        
        # Maximum drawdown
        self.df['max_drawdown_60d'] = self.df.groupby('symbol')['Close'].transform(
            lambda x: (x / x.rolling(60, min_periods=1).max() - 1)
        )
        
        print("  → Trend strength...")
        # ADX approximation (trend strength) - use existing ATR_14 from market_data if available
        if 'ATR_14' in self.df.columns:
            self.df['atr_14d'] = self.df['ATR_14']
            # Calculate ATR_28 if needed
            high_low = self.df['High'] - self.df['Low']
            high_close = abs(self.df['High'] - self.df.groupby('symbol')['Close'].shift(1))
            low_close = abs(self.df['Low'] - self.df.groupby('symbol')['Close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.df['atr_28d'] = self.df.groupby('symbol')['ATR_14'].transform(lambda x: x.rolling(28).mean())
        
        print("  → Gap analysis features...")
        # Gap magnitude (absolute value - volatility of gaps)
        if 'Overnight_Gap' in self.df.columns:
            self.df['gap_magnitude'] = abs(self.df['Overnight_Gap'])
            
            # Gap direction (1 = gap up, 0 = no gap, -1 = gap down)
            self.df['gap_direction'] = np.sign(self.df['Overnight_Gap'])
            
            # Gap persistence (does intraday close fill the gap?)
            # Positive = gap not filled, Negative = gap filled and reversed
            prev_close = self.df.groupby('symbol')['Close'].shift(1)
            self.df['gap_fill_ratio'] = np.where(
                self.df['Overnight_Gap'] > 0,  # Gap up
                (self.df['Close'] - prev_close) / (self.df['Open'] - prev_close),  # How much of gap remains
                np.where(
                    self.df['Overnight_Gap'] < 0,  # Gap down
                    (prev_close - self.df['Close']) / (prev_close - self.df['Open']),
                    0  # No gap
                )
            )
            
            # Gap vs recent volatility (normalized gap size)
            self.df['gap_vs_volatility_20d'] = self.df['gap_magnitude'] / (self.df['volatility_20d'] + 1e-6)
            
            # Multi-day gap patterns
            self.df['gap_streak'] = self.df.groupby('symbol')['gap_direction'].transform(
                lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
            ) * self.df['gap_direction']  # Preserve sign
            
            # Rolling gap statistics
            self.df['gap_avg_5d'] = self.df.groupby('symbol')['Overnight_Gap'].rolling(5).mean().reset_index(level=0, drop=True)
            self.df['gap_std_20d'] = self.df.groupby('symbol')['Overnight_Gap'].rolling(20).std().reset_index(level=0, drop=True)
            
            # Large gap indicator (>2 std deviations)
            self.df['large_gap_indicator'] = (self.df['gap_magnitude'] > 2 * (self.df['gap_std_20d'] + 1e-6)).astype(int)
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} ticker-level features")
    
    def add_cross_sectional_features(self):
        """Add cross-sectional features (rankings within date)"""
        print("\n[4/6] Generating cross-sectional features...")
        
        feature_count = len(self.df.columns)
        
        print("  → Return rankings...")
        # Rank momentum across stocks (percentile within date)
        for window in [20, 60, 120]:
            self.df[f'return_{window}d_rank'] = self.df.groupby('date')[f'return_{window}d'].rank(pct=True)
        
        print("  → Volatility rankings...")
        # Rank volatility
        for window in [20, 60]:
            self.df[f'volatility_{window}d_rank'] = self.df.groupby('date')[f'volatility_{window}d'].rank(pct=True)
        
        print("  → Volume rankings...")
        # Volume intensity rank
        self.df['volume_ratio_20d_rank'] = self.df.groupby('date')['volume_ratio_20d'].rank(pct=True)
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} cross-sectional features")
    
    def merge_market_level_features(self):
        """Merge market-level features (currencies, yields, futures, options, events, sentiment)"""
        print("\n[5/6] Merging market-level features...")
        
        initial_rows = len(self.df)
        
        # Merge market aggregates
        print("  → Market features (breadth, volatility)...")
        before = len(self.df)
        self.df = self.df.merge(self.market_features, on='date', how='left', suffixes=('', '_mkt'))
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: market_features merge changed rows {before:,} → {after:,}")
            print(f"    Checking for duplicates in market_features...")
            dupes = self.market_features[self.market_features.duplicated(subset=['date'], keep=False)]
            if len(dupes) > 0:
                print(f"    Found {len(dupes)} duplicate dates in market_features")
                # Remove duplicates, keep first
                self.market_features = self.market_features.drop_duplicates(subset=['date'], keep='first')
                # Retry merge
                self.df = self.df.iloc[:before].copy()  # Reset to before merge
                self.df = self.df.merge(self.market_features, on='date', how='left', suffixes=('', '_mkt'))
        
        # Merge currency data
        print("  → Currency data (FX pairs)...")
        before = len(self.df)
        self.df = self.df.merge(self.currency_data, on='date', how='left', suffixes=('', '_fx'))
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: currency_data merge changed rows {before:,} → {after:,}")
        
        # Merge treasury yields
        print("  → Treasury yields (risk-free rates)...")
        before = len(self.df)
        self.df = self.df.merge(self.treasury_data, on='date', how='left', suffixes=('', '_tsy'))
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: treasury_data merge changed rows {before:,} → {after:,}")
        
        # Merge futures features
        print("  → Futures features (ES, TY, CL, etc.)...")
        before = len(self.df)
        self.df = self.df.merge(self.futures_features, on='date', how='left', suffixes=('', '_fut'))
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: futures_features merge changed rows {before:,} → {after:,}")
        
        # Merge options features
        print("  → Options features (P/C ratios, IV)...")
        before = len(self.df)
        self.df = self.df.merge(self.options_features, on='date', how='left', suffixes=('', '_opt'))
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: options_features merge changed rows {before:,} → {after:,}")
        
        # Merge macro events
        print("  → Macro events (FOMC, CPI, jobs)...")
        before = len(self.df)
        self.df = self.df.merge(self.events_data, on='date', how='left', suffixes=('', '_evt'))
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: events_data merge changed rows {before:,} → {after:,}")
        
        # Merge market-wide sentiment
        print("  → Sentiment features (market-wide)...")
        before = len(self.df)
        self.df = self.df.merge(self.sentiment_features, on='date', how='left', suffixes=('', '_sent_mkt'))
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: sentiment_features merge changed rows {before:,} → {after:,}")
        
        # Merge ticker-level sentiment
        print("  → Sentiment data (ticker-level)...")
        before = len(self.df)
        self.df = self.df.merge(
            self.sentiment_data[['date', 'ticker', 'sentiment_mean', 'sentiment_std', 'news_volume']],
            left_on=['date', 'symbol'],
            right_on=['date', 'ticker'],
            how='left',
            suffixes=('', '_sent_tick')
        )
        after = len(self.df)
        if after != before:
            print(f"    ⚠ Warning: sentiment_data merge changed rows {before:,} → {after:,}")
            print(f"    Checking for duplicates in sentiment_data...")
            dupes = self.sentiment_data[self.sentiment_data.duplicated(subset=['date', 'ticker'], keep=False)]
            if len(dupes) > 0:
                print(f"    Found {len(dupes)} duplicate (date, ticker) pairs in sentiment_data")
        
        # Drop duplicate ticker column from sentiment
        if 'ticker' in self.df.columns:
            self.df = self.df.drop(columns=['ticker'])
        
        final_rows = len(self.df)
        if final_rows != initial_rows:
            print(f"\n  ⚠ Row count changed: {initial_rows:,} → {final_rows:,} (diff: {final_rows - initial_rows:,})")
            print(f"  This suggests duplicate dates in one of the market-level tables")
        else:
            print(f"  ✓ Merged all market-level data")
            print(f"  ✓ Total features: {len(self.df.columns)}")
    
    def create_interaction_features(self):
        """Create cross-dataset interaction features"""
        print("\n[6/6] Creating interaction features...")
        
        feature_count = len(self.df.columns)
        
        print("  → Calendar seasonality...")
        # Month and quarter capture earnings season clustering, sell-in-May, Jan effect
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        
        print("  → Sentiment-price divergence...")
        # Sentiment vs return divergence
        if 'sentiment_mean' in self.df.columns:
            self.df['sentiment_return_divergence_20d'] = (
                self.df['sentiment_mean'] - self.df['return_20d']
            )
        
        print("  → Volatility arbitrage signals...")
        # Realized vs implied volatility
        if 'volatility_20d' in self.df.columns and 'SPY_IV_mean' in self.df.columns:
            self.df['vol_arb_signal'] = self.df['volatility_20d'] - self.df['SPY_IV_mean']
        
        print("  → Currency-equity correlations...")
        # DXY correlation with returns
        if 'DXY_close' in self.df.columns:
            self.df['return_dxy_corr_60d'] = self.df.groupby('symbol').apply(
                lambda x: x['return_20d'].rolling(60).corr(x['DXY_close'].pct_change())
            ).reset_index(level=0, drop=True)
        
        print("  → Yield curve interactions...")
        # Slope × momentum
        if 'yield_10y' in self.df.columns and 'yield_2y' in self.df.columns:
            self.df['yield_slope'] = self.df['yield_10y'] - self.df['yield_2y']
            self.df['slope_momentum_interaction'] = self.df['yield_slope'] * self.df['return_60d']
        
        print("  → Macro regime features...")
        # VIX regime × return
        if 'VIX_close' in self.df.columns:
            self.df['vix_regime_high'] = (self.df['VIX_close'] > 20).astype(int)
            self.df['return_20d_vix_interaction'] = self.df['return_20d'] * self.df['vix_regime_high']
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} interaction features")
    
    # =========================================================================
    # PROFESSIONAL FEATURES (adapted from #_master_database/ pipeline)
    # =========================================================================
    
    def add_regime_features(self):
        """Add market regime detection features (market-level, merged by date)
        
        Adapted from: #_master_database/create_regime_features.py
        Source: futures_features, treasury_data, currency_data (v2.db)
        Output: ~20 regime classification and characterization features
        """
        print("\n[PRO 1/5] Adding regime detection features...")
        feature_count = len(self.df.columns)
        
        # Build from raw source tables (one row per date)
        ff = self.futures_features.sort_values('date').reset_index(drop=True)
        tsy = self.treasury_data.sort_values('date').reset_index(drop=True)
        cur = self.currency_data.sort_values('date').reset_index(drop=True)
        
        reg = ff[['date', 'SPY_Close', 'SPY_Return_1d', 'TLT_Close', 'TLT_Return_1d',
                   'VXX_Close', 'VXX_Return_1d', 'GLD_Close', 'DX_Close',
                   'QQQ_Return_1d']].copy()
        reg = reg.merge(tsy[['date', 'Treasury_2Y', 'Treasury_5Y', 'Treasury_10Y', 'Treasury_30Y']],
                         on='date', how='left')
        reg = reg.merge(cur[['date', 'USD_Strength_Index']], on='date', how='left')
        
        # Rolling z-score helper
        def _rz(s, w=252, mp=63):
            m = s.rolling(w, min_periods=mp).mean()
            sd = s.rolling(w, min_periods=mp).std()
            return (s - m) / (sd + 1e-8)
        
        # === VIX Regime (VXX as proxy) ===
        reg['reg_vix_pctile_252d'] = reg['VXX_Close'].rolling(252, min_periods=63).rank(pct=True)
        reg['reg_vix_momentum_5d'] = reg['VXX_Close'].pct_change(5)
        reg['reg_vix_momentum_20d'] = reg['VXX_Close'].pct_change(20)
        pct = reg['reg_vix_pctile_252d']
        reg['reg_vix_regime'] = np.where(pct < 0.25, 0, np.where(pct < 0.75, 1, 2)).astype(float)
        
        # === Yield Curve Regime ===
        reg['reg_curve_steepness'] = reg['Treasury_10Y'] - reg['Treasury_2Y']
        reg['reg_curve_steep_chg_20d'] = reg['reg_curve_steepness'].diff(20)
        reg['reg_yield_5y30y'] = reg['Treasury_30Y'] - reg['Treasury_5Y']
        cs = reg['reg_curve_steepness']
        reg['reg_yield_curve_regime'] = np.select(
            [cs < -0.5, cs < 0, cs < 1.0, cs < 2.0],
            [0, 1, 2, 3], default=4
        ).astype(float)
        
        # === Risk-On/Off Composite ===
        spy_tlt_mom = (reg['SPY_Close'] / reg['TLT_Close']).pct_change(5)
        reg['reg_spy_tlt_mom_5d'] = spy_tlt_mom
        gld_dx_mom = (reg['GLD_Close'] / reg['DX_Close']).pct_change(5)
        vix_mom = reg['reg_vix_momentum_5d']
        
        reg['reg_risk_on_off'] = (
            _rz(spy_tlt_mom).fillna(0) * 0.4
            - _rz(vix_mom).fillna(0) * 0.3
            + _rz(gld_dx_mom).fillna(0) * 0.3
        )
        roc = reg['reg_risk_on_off']
        reg['reg_risk_regime'] = np.select(
            [roc < -1, roc < -0.5, roc < 0.5, roc < 1.0],
            [0, 1, 2, 3], default=4
        ).astype(float)
        
        # === Dollar Strength Regime ===
        reg['reg_dollar_str_pctile'] = reg['USD_Strength_Index'].rolling(252, min_periods=63).rank(pct=True)
        dp = reg['reg_dollar_str_pctile']
        reg['reg_dollar_regime'] = np.where(dp < 0.33, 0, np.where(dp < 0.67, 1, 2)).astype(float)
        
        # === Trend Regime ===
        sma50 = reg['SPY_Close'].rolling(50, min_periods=30).mean()
        sma200 = reg['SPY_Close'].rolling(200, min_periods=100).mean()
        reg['reg_spy_vs_sma50'] = (reg['SPY_Close'] / sma50 - 1) * 100
        reg['reg_spy_vs_sma200'] = (reg['SPY_Close'] / sma200 - 1) * 100
        reg['reg_spy_trend_str'] = sma50 / sma200
        s50, s200 = reg['reg_spy_vs_sma50'], reg['reg_spy_vs_sma200']
        reg['reg_trend_regime'] = np.select(
            [(s50 < -2) & (s200 < -5), (s50 < 0) & (s200 < 0),
             (s50 > 2) & (s200 > 5), (s50 > 0) & (s200 > 0)],
            [0, 1, 4, 3], default=2
        ).astype(float)
        
        # === Volatility Persistence ===
        reg['reg_spy_rvol_20d'] = reg['SPY_Return_1d'].rolling(20, min_periods=10).std() * np.sqrt(252)
        tlt_vol = reg['TLT_Return_1d'].rolling(20, min_periods=10).std() * np.sqrt(252)
        reg['reg_vol_persistence'] = reg['reg_spy_rvol_20d'].rolling(20, min_periods=10).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )
        reg['reg_vol_corr_stk_bnd'] = reg['reg_spy_rvol_20d'].rolling(63, min_periods=30).corr(tlt_vol)
        
        # Collect regime columns and merge into main df
        regime_cols = [c for c in reg.columns if c.startswith('reg_')]
        regime_out = reg[['date'] + regime_cols].drop_duplicates(subset=['date'], keep='first')
        
        before = len(self.df)
        self.df = self.df.merge(regime_out, on='date', how='left')
        if len(self.df) != before:
            print(f"  ⚠ Regime merge changed rows {before:,} → {len(self.df):,}")
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} regime detection features")
    
    def add_cross_asset_flow_features(self):
        """Add cross-asset capital flow signals (market-level, merged by date)
        
        Adapted from: #_master_database/create_cross_asset_flows.py
        Source: futures_features, currency_data (v2.db)
        Output: ~17 cross-asset flow and rotation features
        """
        print("\n[PRO 2/5] Adding cross-asset flow features...")
        feature_count = len(self.df.columns)
        
        ff = self.futures_features.sort_values('date').reset_index(drop=True)
        cur = self.currency_data.sort_values('date').reset_index(drop=True)
        
        fl = ff[['date', 'SPY_Close', 'SPY_Return_1d', 'QQQ_Return_1d',
                  'TLT_Close', 'TLT_Return_1d', 'GLD_Return_1d',
                  'CL_Return_1d', 'DX_Return_1d', 'HG_Return_1d',
                  'VXX_Return_1d']].copy()
        fl = fl.merge(cur[['date', 'USD_JPY', 'AUD_USD']], on='date', how='left')
        
        # === Bond-Equity Rotation ===
        spy_tlt = fl['SPY_Close'] / fl['TLT_Close']
        fl['flow_bond_eq_rot_5d'] = spy_tlt.pct_change(5)
        fl['flow_bond_eq_rot_20d'] = spy_tlt.pct_change(20)
        fl['flow_rot_str_5d'] = fl['flow_bond_eq_rot_5d'].abs()
        fl['flow_rot_pctile_252d'] = spy_tlt.rolling(252, min_periods=63).rank(pct=True)
        
        # === Safe Haven Flows ===
        jpy_safe = -fl['USD_JPY'].pct_change(5)  # JPY strength = safe haven demand
        gld_safe = fl['GLD_Return_1d']
        bnd_safe = fl['TLT_Return_1d']
        
        def _rz63(s):
            m = s.rolling(63, min_periods=21).mean()
            sd = s.rolling(63, min_periods=21).std()
            return (s - m) / (sd + 1e-8)
        
        fl['flow_safe_haven'] = (_rz63(jpy_safe) + _rz63(gld_safe) + _rz63(bnd_safe)) / 3
        fl['flow_safe_haven_str'] = fl['flow_safe_haven'].abs()
        
        # === Commodity-Currency Alignment ===
        fl['flow_oil_dollar_align'] = (fl['CL_Return_1d'] - fl['DX_Return_1d']).rolling(20, min_periods=10).mean()
        fl['flow_gold_dollar_align'] = -(fl['GLD_Return_1d'] + fl['DX_Return_1d']).rolling(20, min_periods=10).mean()
        comm = pd.DataFrame({'a': fl['CL_Return_1d'], 'b': fl['GLD_Return_1d'], 'c': fl['HG_Return_1d']})
        fl['flow_commodity_breadth'] = (comm > 0).sum(axis=1) / 3.0
        
        # === Credit-Equity Divergence ===
        eq_mom = (fl['SPY_Return_1d'].rolling(5).mean() + fl['QQQ_Return_1d'].rolling(5).mean()) / 2
        fl['flow_credit_eq_div'] = eq_mom + fl['TLT_Return_1d'].rolling(5).mean()
        
        # === Risk Asset Breadth ===
        risk = pd.DataFrame({
            'a': fl['SPY_Return_1d'], 'b': fl['QQQ_Return_1d'],
            'c': fl['CL_Return_1d'], 'd': fl['HG_Return_1d']
        })
        fl['flow_risk_breadth_1d'] = (risk > 0).sum(axis=1) / 4.0
        fl['flow_risk_breadth_5d'] = (risk.rolling(5).mean() > 0).sum(axis=1) / 4.0
        fl['flow_risk_brdth_mom'] = fl['flow_risk_breadth_5d'] - fl['flow_risk_breadth_5d'].shift(5)
        
        # === Asset Rotation Strength ===
        stk_v_bnd = fl['SPY_Return_1d'].rolling(5).sum() - fl['TLT_Return_1d'].rolling(5).sum()
        stk_v_cmd = fl['SPY_Return_1d'].rolling(5).sum() - fl['GLD_Return_1d'].rolling(5).sum()
        bnd_v_cmd = fl['TLT_Return_1d'].rolling(5).sum() - fl['GLD_Return_1d'].rolling(5).sum()
        fl['flow_stk_vs_bnd_5d'] = stk_v_bnd
        fl['flow_rot_strength'] = pd.DataFrame({'a': stk_v_bnd, 'b': stk_v_cmd, 'c': bnd_v_cmd}).std(axis=1)
        
        # Rotation leader: 2=stocks, 1=bonds, 0=commodities
        s_score = stk_v_bnd + stk_v_cmd
        b_score = -stk_v_bnd + bnd_v_cmd
        c_score = -stk_v_cmd - bnd_v_cmd
        fl['flow_rot_leader'] = np.select(
            [(s_score > b_score) & (s_score > c_score),
             (b_score > s_score) & (b_score > c_score),
             (c_score > s_score) & (c_score > b_score)],
            [2, 1, 0], default=-1
        ).astype(float)
        
        # Collect and merge
        flow_cols = [c for c in fl.columns if c.startswith('flow_')]
        flow_out = fl[['date'] + flow_cols].drop_duplicates(subset=['date'], keep='first')
        
        before = len(self.df)
        self.df = self.df.merge(flow_out, on='date', how='left')
        if len(self.df) != before:
            print(f"  ⚠ Flow merge changed rows {before:,} → {len(self.df):,}")
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} cross-asset flow features")
    
    def add_momentum_quality_features(self):
        """Add momentum quality factors (ticker-level, computed on self.df)
        
        Adapted from: #_master_database/create_momentum_quality_optimized.py
        Uses existing: return_*d, volatility_*d, Close, SMA_*, Market_Return_Mean
        Output: ~15 risk-adjusted momentum, consistency, and positioning features
        """
        print("\n[PRO 3/5] Adding momentum quality features...")
        feature_count = len(self.df.columns)
        
        self.df = self.df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # --- Risk-Adjusted Momentum ---
        print("  → Risk-adjusted momentum...")
        for period, ret_col, vol_col in [(5, 'return_5d', 'volatility_5d'),
                                          (20, 'return_20d', 'volatility_20d'),
                                          (60, 'return_60d', 'volatility_60d')]:
            if ret_col in self.df.columns and vol_col in self.df.columns:
                self.df[f'mqual_risk_adj_{period}d'] = self.df[ret_col] / (self.df[vol_col] + 0.01)
        
        # Composite quality score
        if 'mqual_risk_adj_20d' in self.df.columns and 'mqual_risk_adj_60d' in self.df.columns:
            self.df['mqual_quality_score'] = (
                self.df['mqual_risk_adj_20d'] * 0.4 + self.df['mqual_risk_adj_60d'] * 0.6
            )
        
        # --- Sortino Ratio ---
        print("  → Downside risk metrics...")
        self.df['_neg_ret'] = self.df['Daily_Return'].clip(upper=0)
        neg_vol = self.df.groupby('symbol')['_neg_ret'].transform(
            lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252)
        )
        if 'return_20d' in self.df.columns:
            self.df['mqual_sortino_20d'] = self.df['return_20d'] / (neg_vol + 0.01)
        self.df.drop(columns=['_neg_ret'], inplace=True)
        
        # --- Momentum Consistency ---
        print("  → Momentum consistency...")
        self.df['_pos_day'] = (self.df['Daily_Return'] > 0).astype(float)
        self.df['mqual_consistency_20d'] = self.df.groupby('symbol')['_pos_day'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        self.df.drop(columns=['_pos_day'], inplace=True)
        
        # Momentum acceleration
        if 'return_20d' in self.df.columns:
            self.df['mqual_accel_20d'] = self.df.groupby('symbol')['return_20d'].diff()
        
        # Historical momentum percentile
        if 'return_20d' in self.df.columns:
            self.df['mqual_pctile_252d'] = self.df.groupby('symbol')['return_20d'].transform(
                lambda x: x.rolling(252, min_periods=63).rank(pct=True)
            )
        
        # High quality momentum flag
        if all(c in self.df.columns for c in ['mqual_risk_adj_20d', 'mqual_consistency_20d', 'mqual_pctile_252d']):
            self.df['mqual_high_quality'] = (
                (self.df['mqual_risk_adj_20d'] > 1.0) &
                (self.df['mqual_consistency_20d'] > 0.6) &
                (self.df['mqual_pctile_252d'] > 0.8)
            ).astype(int)
        
        # --- Cross-Market Momentum ---
        print("  → Cross-market positioning...")
        if 'return_20d' in self.df.columns and 'Market_Return_Mean' in self.df.columns:
            self.df['mqual_vs_market'] = self.df['return_20d'] - self.df['Market_Return_Mean'] * 20
        
        # Technical positioning (binary above/below key SMAs)
        for sma, name in [('SMA_20', 'sma20'), ('SMA_50', 'sma50'), ('SMA_200', 'sma200')]:
            if sma in self.df.columns:
                self.df[f'mqual_above_{name}'] = (self.df['Close'] > self.df[sma]).astype(int)
        
        # Leadership / laggard classification
        if 'mqual_pctile_252d' in self.df.columns:
            self.df['mqual_leader'] = (self.df['mqual_pctile_252d'] > 0.9).astype(int)
            self.df['mqual_laggard'] = (self.df['mqual_pctile_252d'] < 0.1).astype(int)
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} momentum quality features")
    
    def add_sentiment_divergence_features(self):
        """Add sentiment divergence signals (ticker-level + market-level)
        
        Adapted from: #_master_database/create_sentiment_divergence.py
        Source: sentiment_data (ticker), sentiment_features (market) from v2.db
        Output: ~15 divergence, exhaustion, and contrarian features
        """
        print("\n[PRO 4/5] Adding sentiment divergence features...")
        feature_count = len(self.df.columns)
        
        # Merge additional sentiment columns not already in self.df
        extra_cols = ['sentiment_ma_7d', 'sentiment_momentum_3d', 'sentiment_volatility_7d',
                      'extreme_positive', 'extreme_negative']
        available_extra = [c for c in extra_cols if c in self.sentiment_data.columns]
        
        if available_extra:
            merge_cols = ['date', 'ticker'] + available_extra
            extra_sent = self.sentiment_data[merge_cols].drop_duplicates(
                subset=['date', 'ticker'], keep='first'
            )
            before_rows = len(self.df)
            self.df = self.df.merge(
                extra_sent,
                left_on=['date', 'symbol'],
                right_on=['date', 'ticker'],
                how='left',
                suffixes=('', '_xsent')
            )
            # Clean up duplicate columns from merge
            for col in ['ticker', 'ticker_xsent']:
                if col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)
            if len(self.df) != before_rows:
                print(f"  ⚠ Sentiment merge changed rows, deduplicating...")
                self.df = self.df.drop_duplicates(subset=['date', 'symbol'], keep='first')
                self.df.reset_index(drop=True, inplace=True)
        
        # --- Price-Sentiment Divergence ---
        if 'sentiment_mean' in self.df.columns and 'return_5d' in self.df.columns:
            # Scale return to sentiment range (sentiment ~[-1,1], 5d return ~[-0.1,0.1])
            self.df['sdiv_price_sent_div'] = self.df['return_5d'] * 10 - self.df['sentiment_mean']
            self.df['sdiv_div_strength'] = self.df['sdiv_price_sent_div'].abs()
        
        # Extreme divergence flags
        sent_col = 'sentiment_ma_7d' if 'sentiment_ma_7d' in self.df.columns else 'sentiment_mean'
        if sent_col in self.df.columns and 'return_5d' in self.df.columns:
            self.df['sdiv_bull_sent_bear_px'] = (
                (self.df[sent_col] > 0.2) & (self.df['return_5d'] < -0.02)
            ).astype(int)
            self.df['sdiv_bear_sent_bull_px'] = (
                (self.df[sent_col] < -0.2) & (self.df['return_5d'] > 0.02)
            ).astype(int)
        
        # --- News Exhaustion ---
        if 'news_volume' in self.df.columns:
            self.df['sdiv_news_vol_pctile'] = self.df.groupby('symbol')['news_volume'].transform(
                lambda x: x.rolling(63, min_periods=21).rank(pct=True)
            )
            if 'return_5d' in self.df.columns:
                self.df['sdiv_news_exhaustion'] = (
                    (self.df['sdiv_news_vol_pctile'] > 0.8) & (self.df['return_5d'].abs() < 0.01)
                ).astype(int)
        
        # --- Sentiment Stability ---
        if 'sentiment_mean' in self.df.columns:
            sent_std = self.df.groupby('symbol')['sentiment_mean'].transform(
                lambda x: x.rolling(20, min_periods=10).std()
            )
            self.df['sdiv_sent_stability'] = 1.0 / (sent_std + 0.01)
        
        # --- Sentiment Shocks ---
        if 'sentiment_momentum_3d' in self.df.columns:
            self.df['sdiv_shock_pos'] = (self.df['sentiment_momentum_3d'] > 0.3).astype(int)
            self.df['sdiv_shock_neg'] = (self.df['sentiment_momentum_3d'] < -0.3).astype(int)
        
        # --- Contrarian Signals ---
        if 'sentiment_mean' in self.df.columns:
            self.df['sdiv_sent_pctile_252d'] = self.df.groupby('symbol')['sentiment_mean'].transform(
                lambda x: x.rolling(252, min_periods=63).rank(pct=True)
            )
            sp = self.df['sdiv_sent_pctile_252d']
            self.df['sdiv_contrarian_bull'] = np.maximum(0, sp - 0.8) * 5
            self.df['sdiv_contrarian_bear'] = np.maximum(0, 0.2 - sp) * 5
            self.df['sdiv_mean_reversion'] = self.df['sdiv_contrarian_bull'] + self.df['sdiv_contrarian_bear']
        
        # --- Market-Level Sentiment ---
        sf = self.sentiment_features.sort_values('date').copy()
        mkt_sent = sf[['date']].copy()
        if 'total_news_volume' in sf.columns:
            mkt_sent['sdiv_mkt_news_trend_5d'] = sf['total_news_volume'].pct_change(5).values
        if 'sentiment_breadth_positive' in sf.columns:
            mkt_sent['sdiv_mkt_breadth_mom_5d'] = sf['sentiment_breadth_positive'].diff(5).values
        
        if len(mkt_sent.columns) > 1:
            mkt_sent = mkt_sent.drop_duplicates(subset=['date'], keep='first')
            self.df = self.df.merge(mkt_sent, on='date', how='left')
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} sentiment divergence features")
    
    def add_lag_features(self):
        """Add explicit lag features for key signals
        
        Tree models cannot capture temporal dependencies natively.
        Explicit lags provide the same information LSTM learns implicitly.
        """
        print("\n[PRO 5/5] Adding explicit lag features...")
        feature_count = len(self.df.columns)
        
        self.df = self.df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Key features to lag: {column: [lag_periods_in_days]}
        lag_config = {
            # Core ticker signals
            'Daily_Return': [1, 2, 3, 5],
            'return_5d': [1, 2, 5],
            'return_20d': [1, 5, 10],
            'volatility_20d': [1, 5],
            'volume_ratio_20d': [1, 5],
            'bb_position_20d': [1, 5],
            'sharpe_20d': [5, 10],
            # Regime signals
            'reg_risk_on_off': [1, 5],
            'reg_vix_momentum_5d': [1, 5],
            'reg_spy_rvol_20d': [1, 5],
            # Flow signals
            'flow_bond_eq_rot_5d': [1, 5],
            'flow_safe_haven': [1, 5],
            'flow_risk_breadth_5d': [1, 5],
            # Momentum quality
            'mqual_risk_adj_20d': [5, 10],
            'mqual_quality_score': [5, 10],
            # Sentiment
            'sentiment_mean': [1, 5],
        }
        
        lag_count = 0
        for col, lags in lag_config.items():
            if col in self.df.columns:
                for lag in lags:
                    self.df[f'lag{lag}d_{col}'] = self.df.groupby('symbol')[col].shift(lag)
                    lag_count += 1
        
        new_features = len(self.df.columns) - feature_count
        print(f"  ✓ Added {new_features} explicit lag features")
    
    def save_features(self):
        """Save master feature set"""
        print("\n[7/7] Saving feature set...")
        
        # Sort by date and symbol
        self.df = self.df.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # Save to CSV
        self.df.to_csv(self.output_csv, index=False)
        
        # Summary statistics
        file_size_mb = os.path.getsize(self.output_csv) / (1024 * 1024)
        
        print(f"  ✓ Saved: {os.path.basename(self.output_csv)}")
        print(f"  ✓ Rows: {len(self.df):,}")
        print(f"  ✓ Features: {len(self.df.columns)}")
        print(f"  ✓ File size: {file_size_mb:.1f} MB")
        print(f"  ✓ Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"  ✓ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Feature breakdown
        print(f"\n  Feature Breakdown:")
        print(f"    - Base market data: ~20 features")
        print(f"    - Ticker-level technical: ~30 features")
        print(f"    - Cross-sectional rankings: ~10 features")
        print(f"    - Currency features: ~{len([c for c in self.df.columns if '_fx' in c])} features")
        print(f"    - Treasury features: ~{len([c for c in self.df.columns if '_tsy' in c or 'yield_' in c])} features")
        print(f"    - Futures features: ~{len([c for c in self.df.columns if '_fut' in c])} features")
        print(f"    - Options features: ~{len([c for c in self.df.columns if '_opt' in c or 'IV_' in c or 'PC_' in c])} features")
        print(f"    - Events features: ~{len([c for c in self.df.columns if '_evt' in c or 'fomc' in c.lower() or 'cpi' in c.lower()])} features")
        print(f"    - Sentiment features: ~{len([c for c in self.df.columns if 'sentiment' in c])} features")
        print(f"    - Interaction features: ~{len([c for c in self.df.columns if 'divergence' in c or 'arb' in c or 'interaction' in c])} features")
    
    def run(self):
        """Execute complete feature engineering pipeline"""
        start_time = datetime.now()
        
        try:
            # Load data
            self.load_database_tables()
            
            # Create base
            self.create_base_dataframe()
            
            # Generate features
            self.add_ticker_level_features()
            self.add_cross_sectional_features()
            self.merge_market_level_features()
            self.create_interaction_features()
            
            # Professional features (adapted from quant_trading_final.db pipeline)
            self.add_regime_features()
            self.add_cross_asset_flow_features()
            self.add_momentum_quality_features()
            self.add_sentiment_divergence_features()
            self.add_lag_features()
            
            # Save
            self.save_features()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print("\n" + "=" * 80)
            print("FEATURE ENGINEERING COMPLETE!")
            print("=" * 80)
            print(f"Duration: {elapsed:.1f} seconds")
            print(f"Output: {self.output_csv}")
            print(f"Total Features: {len(self.df.columns)}")
            print(f"Total Rows: {len(self.df):,}")
            print("\nNext Step: Step 10 - Model Training (with normalization)")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Error in feature engineering: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main execution"""
    print("🎯 STEP 9: RENAISSANCE-LEVEL FEATURE ENGINEERING")
    print("   Strategy: Generate raw features → Normalize in Step 10")
    print("   Target: 400-600 features from 10 database tables")
    print()
    
    engineer = RenaissanceFeatureEngineer()
    engineer.run()

if __name__ == "__main__":
    main()
