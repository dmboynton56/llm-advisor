#!/usr/bin/env python3
"""Backtest runner script.

Runs the trading system on historical data to simulate trades and calculate P/L.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.live.loop import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STDEV Backtest Runner")
    parser.add_argument("--date", required=True, help="Date to backtest (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to test")
    parser.add_argument("--fast", type=int, default=1, help="Loop speed in seconds (lower=faster)")
    parser.add_argument("--output", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Force test mode
    args.test = True
    
    print(f"Running backtest for {args.date}")
    print(f"Symbols: {args.symbols or 'default watchlist'}")
    print(f"Loop speed: {args.fast} seconds per iteration")
    print("-" * 60)
    
    main()

