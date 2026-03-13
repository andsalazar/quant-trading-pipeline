#!/usr/bin/env python3
"""
IBKR Trading Automation Script
===============================

Execute confidence-weighted trades via Interactive Brokers API

Requirements:
    pip install ib_insync pandas

Usage:
    python ibkr_trader.py --dry-run  # Test mode (no real trades)
    python ibkr_trader.py            # Live trading mode

Configuration:
    Edit settings in CONFIG section below
"""

import pandas as pd
import numpy as np
from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
from datetime import datetime
import time
import argparse
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # IBKR Connection
    'host': '127.0.0.1',
    'port': 7497,  # 7497 = paper trading, 7496 = live trading
    'client_id': 1,
    
    # Trading Parameters
    'predictions_file': 'predictions_latest.csv',
    'confidence_threshold': 0.10,  # Only trade if confidence >= 0.10
    'position_sizing': 'confidence_weighted',  # 'equal_weight', 'confidence_weighted', 'kelly'
    
    # Risk Management
    'max_position_pct': 0.10,  # Max 10% in any single stock
    'max_total_long': 1.00,    # Max 100% long exposure
    'max_total_short': 0.20,   # Max 20% short exposure
    'use_stop_loss': True,
    'stop_loss_pct': 0.02,     # 2% stop loss
    
    # Order Settings
    'order_type': 'LMT',  # 'MKT' (market) or 'LMT' (limit)
    'limit_offset_pct': 0.005,  # 0.5% above/below market for limit orders
    
    # Logging
    'save_trade_log': True,
    'log_directory': 'trade_logs'
}

# ============================================================================
# TRADER CLASS
# ============================================================================

class IBKRTrader:
    """Automated trading via IBKR API"""
    
    def __init__(self, config):
        self.config = config
        self.ib = IB()
        self.predictions = None
        self.portfolio_value = 0
        self.executed_trades = []
        
    def connect(self):
        """Connect to IBKR TWS/Gateway"""
        print(f"\n{'='*80}")
        print(f"CONNECTING TO IBKR")
        print(f"{'='*80}")
        print(f"Host: {self.config['host']}")
        print(f"Port: {self.config['port']} ({'PAPER' if self.config['port'] == 7497 else 'LIVE'})")
        
        try:
            self.ib.connect(
                self.config['host'],
                self.config['port'],
                clientId=self.config['client_id']
            )
            print(f"✓ Connected successfully")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Ensure TWS or IB Gateway is running")
            print("  2. Enable API connections in TWS settings:")
            print("     File → Global Configuration → API → Settings")
            print("     - Enable ActiveX and Socket Clients")
            print("     - Socket port 7497 (paper) or 7496 (live)")
            print("  3. Check firewall settings")
            sys.exit(1)
        
        # Get account value
        self._get_portfolio_value()
        
    def _get_portfolio_value(self):
        """Retrieve current portfolio value"""
        account_values = self.ib.accountValues()
        
        for av in account_values:
            if av.tag == 'NetLiquidation' and av.currency == 'USD':
                self.portfolio_value = float(av.value)
                print(f"✓ Portfolio Value: ${self.portfolio_value:,.2f}")
                return
        
        print(f"⚠ Warning: Could not retrieve portfolio value")
        self.portfolio_value = 100000  # Default fallback
        
    def load_predictions(self):
        """Load and filter predictions"""
        print(f"\n{'='*80}")
        print(f"LOADING PREDICTIONS")
        print(f"{'='*80}")
        
        filepath = self.config['predictions_file']
        threshold = self.config['confidence_threshold']
        
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded {len(df)} predictions from {filepath}")
        except FileNotFoundError:
            print(f"✗ Error: File not found: {filepath}")
            print(f"  Run: python 01_05_generate_predictions.py")
            sys.exit(1)
        
        # Filter by confidence
        df_filtered = df[df['confidence'] >= threshold].copy()
        
        print(f"✓ Filtered to {len(df_filtered)} high-confidence trades (>={threshold})")
        print(f"  Date: {df['date'].iloc[0] if len(df) > 0 else 'N/A'}")
        
        # Separate long/short
        longs = df_filtered[df_filtered['prediction'].isin(['Up', 'Outperform'])]
        shorts = df_filtered[df_filtered['prediction'].isin(['Down', 'Underperform'])]
        
        print(f"  Long positions:  {len(longs)} ({len(longs)/len(df_filtered):.1%})")
        print(f"  Short positions: {len(shorts)} ({len(shorts)/len(df_filtered):.1%})")
        
        self.predictions = df_filtered
        return df_filtered
        
    def calculate_positions(self):
        """Calculate position sizes"""
        print(f"\n{'='*80}")
        print(f"CALCULATING POSITION SIZES")
        print(f"{'='*80}")
        print(f"Method: {self.config['position_sizing']}")
        
        method = self.config['position_sizing']
        
        if method == 'equal_weight':
            # Equal weight
            self.predictions['weight'] = 1 / len(self.predictions)
            
        elif method == 'confidence_weighted':
            # Confidence-weighted
            total_conf = self.predictions['confidence'].sum()
            self.predictions['weight'] = self.predictions['confidence'] / total_conf
            
        elif method == 'kelly':
            # Half-Kelly criterion
            def half_kelly(prob):
                kelly = prob - (1 - prob)
                return max(0, kelly * 0.5)  # Half-Kelly for safety
            
            self.predictions['kelly'] = self.predictions['ensemble_pred_proba'].apply(half_kelly)
            total_kelly = self.predictions['kelly'].sum()
            self.predictions['weight'] = self.predictions['kelly'] / total_kelly if total_kelly > 0 else 0
        
        # Calculate dollar allocations
        self.predictions['allocation_dollars'] = self.predictions['weight'] * self.portfolio_value
        
        # Apply position limits
        max_position = self.portfolio_value * self.config['max_position_pct']
        self.predictions['allocation_dollars'] = self.predictions['allocation_dollars'].clip(upper=max_position)
        
        # Recalculate weights after clipping
        total_allocated = self.predictions['allocation_dollars'].sum()
        self.predictions['weight'] = self.predictions['allocation_dollars'] / total_allocated
        
        print(f"✓ Positions calculated:")
        print(f"  Total allocated: ${self.predictions['allocation_dollars'].sum():,.2f}")
        print(f"  Portfolio utilization: {self.predictions['allocation_dollars'].sum() / self.portfolio_value:.1%}")
        print(f"  Avg per trade: ${self.predictions['allocation_dollars'].mean():,.2f}")
        print(f"  Max position: ${self.predictions['allocation_dollars'].max():,.2f} ({self.predictions['allocation_dollars'].max() / self.portfolio_value:.1%})")
        print(f"  Min position: ${self.predictions['allocation_dollars'].min():,.2f}")
        
        # Show top allocations
        print(f"\n  Top 10 Allocations:")
        top10 = self.predictions.nlargest(10, 'allocation_dollars')[
            ['symbol', 'prediction', 'confidence', 'allocation_dollars']
        ]
        for idx, row in top10.iterrows():
            print(f"    {row['symbol']:<6} {row['prediction']:<5} conf={row['confidence']:.3f}  ${row['allocation_dollars']:>8,.0f}")
        
        return self.predictions
        
    def get_stock_price(self, symbol):
        """Get current market price"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)  # Wait for data
            
            # Try different price fields
            if ticker.last and ticker.last > 0:
                price = ticker.last
            elif ticker.close and ticker.close > 0:
                price = ticker.close
            elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                price = (ticker.bid + ticker.ask) / 2
            else:
                print(f"    ⚠ No valid price for {symbol}")
                return None
            
            self.ib.cancelMktData(contract)
            return price
            
        except Exception as e:
            print(f"    ⚠ Error getting price for {symbol}: {e}")
            return None
            
    def place_order(self, symbol, shares, action, dry_run=True):
        """Place order (market or limit)"""
        
        if shares <= 0:
            print(f"    ⚠ Skipping {symbol}: {shares} shares")
            return None
        
        try:
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Get current price
            price = self.get_stock_price(symbol)
            if price is None:
                return None
            
            # Create order
            if self.config['order_type'] == 'MKT':
                order = MarketOrder(action, shares)
                print(f"    → Market order: {action} {shares} shares @ market")
                
            elif self.config['order_type'] == 'LMT':
                # Calculate limit price
                offset = self.config['limit_offset_pct']
                if action == 'BUY':
                    limit_price = price * (1 + offset)  # Buy at slight premium
                else:
                    limit_price = price * (1 - offset)  # Sell at slight discount
                
                limit_price = round(limit_price, 2)
                order = LimitOrder(action, shares, limit_price)
                print(f"    → Limit order: {action} {shares} shares @ ${limit_price:.2f}")
            
            # Execute or simulate
            if not dry_run:
                trade = self.ib.placeOrder(contract, order)
                print(f"    ✓ Order submitted (order ID: {trade.order.orderId})")
                
                # Place stop-loss if configured
                if self.config['use_stop_loss'] and action == 'BUY':
                    self._place_stop_loss(symbol, shares, price)
                
                return trade
            else:
                print(f"    [DRY RUN] Would place order")
                return None
                
        except Exception as e:
            print(f"    ✗ Error placing order for {symbol}: {e}")
            return None
            
    def _place_stop_loss(self, symbol, shares, entry_price):
        """Place stop-loss order"""
        try:
            stop_price = entry_price * (1 - self.config['stop_loss_pct'])
            stop_price = round(stop_price, 2)
            
            contract = Stock(symbol, 'SMART', 'USD')
            order = StopOrder('SELL', shares, stop_price)
            
            trade = self.ib.placeOrder(contract, order)
            print(f"    ✓ Stop-loss placed @ ${stop_price:.2f} ({-self.config['stop_loss_pct']:.1%})")
            
        except Exception as e:
            print(f"    ⚠ Stop-loss failed: {e}")
            
    def execute_all(self, dry_run=True):
        """Execute all calculated trades"""
        print(f"\n{'='*80}")
        print(f"EXECUTING TRADES ({'DRY RUN' if dry_run else 'LIVE TRADING'})")
        print(f"{'='*80}")
        
        if not dry_run:
            confirm = input(f"\n⚠ WARNING: You are about to place {len(self.predictions)} LIVE trades.\n"
                          f"   Portfolio: ${self.portfolio_value:,.2f}\n"
                          f"   Total allocation: ${self.predictions['allocation_dollars'].sum():,.2f}\n"
                          f"\n   Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("✗ Cancelled")
                return []
        
        trades = []
        
        for idx, row in self.predictions.iterrows():
            symbol = row['symbol']
            prediction = row['prediction']
            confidence = row['confidence']
            allocation = row['allocation_dollars']
            
            print(f"\n[{idx+1}/{len(self.predictions)}] {symbol} ({prediction}, conf={confidence:.3f})")
            print(f"  Allocation: ${allocation:,.2f}")
            
            # Get price and calculate shares
            price = self.get_stock_price(symbol)
            if price is None:
                continue
                
            shares = int(allocation / price)
            actual_value = shares * price
            
            print(f"  Price: ${price:.2f}")
            print(f"  Shares: {shares} (${actual_value:,.2f})")
            
            # Determine action
            action = 'BUY' if prediction in ('Up', 'Outperform') else 'SELL'
            
            # Place order
            trade = self.place_order(symbol, shares, action, dry_run=dry_run)
            
            # Log trade
            trades.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'shares': shares,
                'price': price,
                'total_value': actual_value,
                'confidence': confidence,
                'prediction': prediction,
                'order_status': 'Submitted' if trade else 'Simulated',
                'trade_object': trade
            })
            
            time.sleep(0.5)  # Rate limiting
        
        self.executed_trades = trades
        
        # Summary
        print(f"\n{'='*80}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total trades: {len(trades)}")
        
        if trades:
            total_value = sum(t['total_value'] for t in trades)
            longs = [t for t in trades if t['action'] == 'BUY']
            shorts = [t for t in trades if t['action'] == 'SELL']
            
            print(f"Total value: ${total_value:,.2f}")
            print(f"Portfolio utilization: {total_value / self.portfolio_value:.1%}")
            print(f"Long positions: {len(longs)} (${sum(t['total_value'] for t in longs):,.2f})")
            print(f"Short positions: {len(shorts)} (${sum(t['total_value'] for t in shorts):,.2f})")
        
        return trades
        
    def save_trade_log(self):
        """Save trade log to CSV"""
        if not self.config['save_trade_log'] or not self.executed_trades:
            return
            
        import os
        os.makedirs(self.config['log_directory'], exist_ok=True)
        
        # Convert to DataFrame
        log_df = pd.DataFrame([{
            'timestamp': t['timestamp'],
            'symbol': t['symbol'],
            'action': t['action'],
            'shares': t['shares'],
            'price': t['price'],
            'total_value': t['total_value'],
            'confidence': t['confidence'],
            'prediction': t['prediction'],
            'status': t['order_status']
        } for t in self.executed_trades])
        
        # Save
        filename = f"{self.config['log_directory']}/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_df.to_csv(filename, index=False)
        print(f"\n✓ Trade log saved: {filename}")
        
    def disconnect(self):
        """Disconnect from IBKR"""
        print(f"\n{'='*80}")
        print("DISCONNECTING")
        print(f"{'='*80}")
        self.ib.disconnect()
        print("✓ Disconnected from IBKR")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Execute trades via IBKR API')
    parser.add_argument('--dry-run', action='store_true', help='Simulation mode (no real trades)')
    parser.add_argument('--confidence', type=float, default=None, help='Confidence threshold (default: 0.10)')
    args = parser.parse_args()
    
    # Override config if specified
    if args.confidence is not None:
        CONFIG['confidence_threshold'] = args.confidence
    
    # Initialize trader
    trader = IBKRTrader(CONFIG)
    
    try:
        # Connect
        trader.connect()
        
        # Load predictions
        trader.load_predictions()
        
        if len(trader.predictions) == 0:
            print("\n⚠ No high-confidence predictions found. Exiting.")
            return
        
        # Calculate positions
        trader.calculate_positions()
        
        # Execute trades
        trades = trader.execute_all(dry_run=args.dry_run)
        
        # Save log
        trader.save_trade_log()
        
        print(f"\n{'='*80}")
        print("COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        trader.disconnect()

if __name__ == "__main__":
    main()
