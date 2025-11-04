#!/usr/bin/env python3
"""Analyze backtest results and logs.

Usage:
    python scripts/analyze_backtest.py --date 2025-10-29
    python scripts/analyze_backtest.py --date 2025-10-29 --symbol SPY
    python scripts/analyze_backtest.py --date 2025-10-29 --show-trades
    python scripts/analyze_backtest.py --date 2025-10-29 --show-timeline
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def load_backtest_results(date_str: str) -> Dict[str, Any]:
    """Load backtest results JSON."""
    results_path = PROJECT_ROOT / "data" / "daily_news" / date_str / "processed" / "backtest_results.json"
    if not results_path.exists():
        print(f"❌ Backtest results not found at {results_path}")
        sys.exit(1)
    
    with open(results_path, 'r') as f:
        return json.load(f)


def load_log_data(date_str: str) -> List[Dict[str, Any]]:
    """Load minute-by-minute log data."""
    log_path = PROJECT_ROOT / "data" / "daily_news" / date_str / "processed" / "live_loop_log.jsonl"
    if not log_path.exists():
        print(f"❌ Log file not found at {log_path}")
        return []
    
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return logs


def print_summary(results: Dict[str, Any]):
    """Print backtest summary."""
    print("\n" + "="*60)
    print(f"BACKTEST SUMMARY - {results['date']}")
    print("="*60)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']} ({results['win_rate']*100:.1f}%)")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"\nP/L: ${results['total_pnl']:.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    if results['winning_trades'] > 0:
        print(f"Average Win: ${results['average_win']:.2f}")
    if results['losing_trades'] > 0:
        print(f"Average Loss: ${results['average_loss']:.2f}")


def print_trades(results: Dict[str, Any]):
    """Print detailed trade list."""
    print("\n" + "="*60)
    print("TRADE DETAILS")
    print("="*60)
    
    for i, trade in enumerate(results['trades'], 1):
        print(f"\nTrade #{i}: {trade['symbol']} {trade['side'].upper()}")
        print(f"  Entry: ${trade['entry_price']:.2f} @ {trade['entry_time']}")
        print(f"  Exit: ${trade['exit_price']:.2f} @ {trade['exit_time']} ({trade['exit_reason']})")
        print(f"  Quantity: {trade['qty']} shares")
        print(f"  Stop Loss: ${trade['stop_loss']:.2f}")
        print(f"  Take Profit: ${trade['take_profit']:.2f}")
        pnl_sign = "+" if trade['pnl'] >= 0 else ""
        print(f"  P/L: {pnl_sign}${trade['pnl']:.2f}")


def print_symbol_timeline(logs: List[Dict[str, Any]], symbol: str):
    """Print timeline for a specific symbol."""
    print(f"\n" + "="*60)
    print(f"TIMELINE: {symbol}")
    print("="*60)
    
    for log in logs:
        if symbol not in log.get('symbols', {}):
            continue
        
        symbol_data = log['symbols'][symbol]
        timestamp = log.get('label', 'Unknown')
        status = symbol_data.get('status', 'unknown')
        z = symbol_data.get('z', 0)
        mu = symbol_data.get('mu', 0)
        sigma = symbol_data.get('sigma', 0)
        
        trade_info = ""
        if symbol_data.get('trade'):
            trade = symbol_data['trade']
            trade_info = f" [TRADE: {trade['side']} @ ${trade['entry']:.2f}]"
        
        print(f"{timestamp:12} | Status: {status:15} | z={z:7.3f} | μ={mu:8.2f} | σ={sigma:6.3f}{trade_info}")


def print_all_timeline(logs: List[Dict[str, Any]], max_lines: int = 50):
    """Print overall timeline."""
    print("\n" + "="*60)
    print("TIMELINE (First 50 ticks)")
    print("="*60)
    
    for i, log in enumerate(logs[:max_lines]):
        timestamp = log.get('label', 'Unknown')
        signals = log.get('signals', [])
        
        symbol_statuses = []
        for symbol, data in log.get('symbols', {}).items():
            status = data.get('status', 'unknown')
            z = data.get('z', 0)
            symbol_statuses.append(f"{symbol}:{status} (z={z:.2f})")
        
        print(f"{timestamp:12} | {', '.join(symbol_statuses)}")
        if signals:
            for sig in signals:
                print(f"  ⚡ SIGNAL: {sig['symbol']} {sig['setup_type']} {sig['side']} @ ${sig['entry_price']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument("--date", required=True, help="Backtest date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default=None, help="Show timeline for specific symbol")
    parser.add_argument("--show-trades", action="store_true", help="Show detailed trade list")
    parser.add_argument("--show-timeline", action="store_true", help="Show overall timeline")
    parser.add_argument("--all", action="store_true", help="Show everything")
    
    args = parser.parse_args()
    
    # Load data
    results = load_backtest_results(args.date)
    logs = load_log_data(args.date)
    
    # Print summary
    print_summary(results)
    
    # Show trades if requested
    if args.show_trades or args.all:
        print_trades(results)
    
    # Show timeline
    if args.symbol:
        print_symbol_timeline(logs, args.symbol)
    elif args.show_timeline or args.all:
        print_all_timeline(logs)
    
    print("\n" + "="*60)
    print(f"Full logs: data/daily_news/{args.date}/processed/")
    print(f"   - backtest_results.json (trade summary)")
    print(f"   - live_loop_log.jsonl (minute-by-minute ticks)")
    print("="*60)


if __name__ == "__main__":
    main()

