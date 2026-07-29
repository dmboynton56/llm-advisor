#!/usr/bin/env python3
"""Replay recorded option marks under premium and underlying-price exit policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.option_replay import (  # noqa: E402
    ReplayPolicy,
    load_marks_from_order_events,
    replay_exit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--entry-order-id")
    parser.add_argument("--entry-price", type=float, required=True)
    parser.add_argument("--qty", type=float, required=True)
    parser.add_argument("--target-pct", type=float, default=0.25)
    parser.add_argument("--stop-pct", type=float, default=0.35)
    parser.add_argument("--underlying-stop", type=float)
    parser.add_argument("--underlying-target", type=float)
    parser.add_argument("--underlying-side", choices=("long", "short"), default="long")
    args = parser.parse_args()

    marks = load_marks_from_order_events(
        args.events,
        option_symbol=args.symbol,
        entry_order_id=args.entry_order_id,
    )
    policies = [
        ReplayPolicy(
            name="premium_thresholds",
            profit_target_pct=args.target_pct,
            stop_loss_pct=args.stop_pct,
        )
    ]
    if args.underlying_stop is not None or args.underlying_target is not None:
        policies.append(
            ReplayPolicy(
                name="underlying_trade_plan",
                underlying_stop=args.underlying_stop,
                underlying_target=args.underlying_target,
                underlying_side=args.underlying_side,
            )
        )
    results = [
        asdict(
            replay_exit(
                marks,
                entry_price=args.entry_price,
                qty=args.qty,
                policy=policy,
            )
        )
        for policy in policies
    ]
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
