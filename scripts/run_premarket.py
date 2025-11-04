#!/usr/bin/env python3
"""Premarket workflow orchestrator.

Replaces main.py with cleaner orchestrator that:
1. Runs bias_gatherer.py → get premarket context
2. Runs snapshot_builder.py → get STDEV snapshots
3. Combines outputs into single JSON file
4. Saves to data/daily_news/{date}/processed/premarket_context.json
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, date
import json
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.premarket.bias_gatherer import gather_premarket_bias, save_premarket_context
from src.premarket.snapshot_builder import build_premarket_snapshots
from src.core.config import Settings


def main():
    """Run premarket pipeline."""
    parser = argparse.ArgumentParser(description="Premarket pipeline orchestrator")
    parser.add_argument("--date", default=None, help="Trading date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to process")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()
    
    # Determine trading date
    if args.date:
        trading_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        et = pytz.timezone("US/Eastern")
        trading_date = datetime.now(et).date()
    
    date_str = trading_date.strftime("%Y-%m-%d")
    
    # Get symbols
    settings = Settings.load()
    symbols = args.symbols or settings.trading.watchlist
    
    print(f"=" * 60)
    print(f"Premarket Pipeline for {date_str}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"=" * 60)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "data" / "daily_news" / date_str / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Gather premarket bias and news
        print("\n[1/2] Gathering premarket bias and news...")
        premarket_context = gather_premarket_bias(
            trading_date=trading_date,
            symbols=symbols,
            output_dir=output_dir
        )
        print(f"[OK] Gathered bias for {len(premarket_context.symbols)} symbols")
        
        # Step 2: Build STDEV snapshots
        print("\n[2/2] Building STDEV snapshots...")
        snapshots = build_premarket_snapshots(
            symbols=symbols,
            trading_date=trading_date
        )
        print(f"[OK] Built snapshots for {len(snapshots.get('symbols', []))} symbols")
        
        # Step 3: Combine and save
        print("\n[3/3] Combining outputs...")
        combined_output = {
            "date": date_str,
            "premarket_context": premarket_context.__dict__,
            "snapshots": snapshots,
        }
        
        # Convert to JSON-serializable format
        output_dict = {
            "date": date_str,
            "premarket_context": {
                "date": premarket_context.date,
                "symbols": {
                    sym: {
                        "symbol": bias.symbol,
                        "daily_bias": bias.daily_bias,
                        "confidence": bias.confidence,
                        "model_output": bias.model_output,
                        "news_summary": bias.news_summary,
                        "premarket_price": bias.premarket_price,
                        "premarket_context": bias.premarket_context,
                    }
                    for sym, bias in premarket_context.symbols.items()
                },
                "market_context": premarket_context.market_context,
            },
            "snapshots": snapshots,
        }
        
        output_path = output_dir / "premarket_context.json"
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2, default=str)
        
        print(f"[OK] Saved combined output to {output_path}")
        print("\n" + "=" * 60)
        print("Premarket pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

