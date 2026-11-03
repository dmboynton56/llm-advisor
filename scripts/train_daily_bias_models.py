#!/usr/bin/env python3
"""Train actual daily priors without executing any trading workflow or orders."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from src.premarket.bias_training_data import fetch_history
from src.premarket.bias_training import build_dataset, fit_and_evaluate, export_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOG", "TSLA"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", required=True, help="Exclusive UTC date, never include an unfinished current session")
    parser.add_argument("--feed", choices=["iex", "sip"], default="sip")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    if end > datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0):
        parser.error("end must exclude today's unfinished session")
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rejected = []
    for symbol in args.symbols:
        frames = fetch_history(client, symbol, start, end, args.cache_dir, args.feed)
        dataset, quality = build_dataset(symbol, frames)
        dataset_path = args.report_dir / f"{symbol}_dataset.csv"
        dataset.to_csv(dataset_path, index=False)
        try:
            model, encoder, report, predictions = fit_and_evaluate(dataset)
        except ValueError as exc:
            report = {"status": "rejected", "reason": str(exc), "data_quality": quality}
            (args.report_dir / f"{symbol}_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(symbol, json.dumps(report), flush=True)
            rejected.append(symbol)
            continue
        (args.report_dir / f"{symbol}_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        predictions.to_csv(args.report_dir / f"{symbol}_holdout.csv", index=False)
        print(symbol, json.dumps({"quality": quality, "validation": report}), flush=True)
        if report["status"] == "accepted":
            export_model(symbol, model, encoder, report, args.output_dir, dataset_path, quality, args.feed)
        else:
            rejected.append(symbol)
    if rejected:
        raise SystemExit(f"Models failed validation and were NOT promoted: {', '.join(rejected)}")


if __name__ == "__main__":
    main()
