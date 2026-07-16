#!/usr/bin/env python3
"""Export a Robinhood-reviewed trade plan from a signal/state JSON payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.trade_plan_exporter import (  # noqa: E402
    DEFAULT_DAILY_LOSS_LIMIT_PCT,
    DEFAULT_EXPIRY_MINUTES,
    build_trade_plan_from_payload,
    write_trade_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing signal, trade, and optional state objects")
    parser.add_argument("--output-dir", type=Path, help="Destination trade_plans directory")
    parser.add_argument("--account-equity", type=float, help="Fresh Robinhood Agentic account equity")
    parser.add_argument("--realized-pnl", type=float, help="Fresh Robinhood realized P&L for the trading day")
    parser.add_argument(
        "--daily-loss-limit-pct",
        type=float,
        default=DEFAULT_DAILY_LOSS_LIMIT_PCT,
        help="Positive decimal loss magnitude (default: 0.01 = 1%%)",
    )
    parser.add_argument("--expires-in-minutes", type=int, default=DEFAULT_EXPIRY_MINUTES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Input JSON must be an object")
    plan = build_trade_plan_from_payload(
        payload,
        account_equity=args.account_equity,
        realized_pnl=args.realized_pnl,
        daily_loss_limit_pct=args.daily_loss_limit_pct,
        expires_in_minutes=args.expires_in_minutes,
    )
    output_dir = args.output_dir
    if output_dir is None:
        run_date = str(plan["generated_at"])[:10]
        output_dir = PROJECT_ROOT / "data" / "daily_news" / run_date / "processed" / "trade_plans"
    destination = write_trade_plan(plan, output_dir, overwrite=args.overwrite)
    print(destination)


if __name__ == "__main__":
    main()
