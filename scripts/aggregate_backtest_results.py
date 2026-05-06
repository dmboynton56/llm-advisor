#!/usr/bin/env python3
"""Roll up ``backtest_results.json`` files under ``data/daily_news/*/processed``.

Emits JSON suitable for portfolio seed copy / notebooks (multi-day simulated performance).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate backtest_results.json rollups.")
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data" / "daily_news"),
        help="Root containing YYYY-MM-DD/processed/backtest_results.json",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write JSON summary here (default: stdout)",
    )
    args = parser.parse_args()
    root = Path(args.data_root).resolve()

    rows: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or len(child.name) != 10:
            continue
        try:
            datetime.strptime(child.name, "%Y-%m-%d")
        except ValueError:
            continue
        bp = child / "processed" / "backtest_results.json"
        if not bp.exists():
            continue
        payload = load_json(bp)
        payload["_run_date"] = child.name
        rows.append(payload)

    if not rows:
        print(f"No backtest_results.json found under {root}", file=sys.stderr)
        raise SystemExit(1)

    pnls = [float(r.get("total_pnl") or 0) for r in rows]
    trades = [int(r.get("closed_trades") or r.get("total_trades") or 0) for r in rows]
    win_rates = [float(r["win_rate"]) for r in rows if r.get("win_rate") is not None]

    summary = {
        "schema": "llm_advisor_backtest_rollup_v1",
        "data_root": str(root),
        "n_days": len(rows),
        "first_date": rows[0]["_run_date"],
        "last_date": rows[-1]["_run_date"],
        "total_pnl_sum": round(sum(pnls), 4),
        "avg_daily_pnl": round(statistics.mean(pnls), 4) if pnls else None,
        "median_daily_pnl": round(statistics.median(pnls), 4) if pnls else None,
        "total_closed_trades": sum(trades),
        "avg_win_rate_daily": round(statistics.mean(win_rates), 6) if win_rates else None,
        "assumption_note": (
            "Simulated backtest per-day runs; not compounded equity unless notebook "
            "computes path-dependent sizing."
        ),
        "days": [
            {
                "date": r["_run_date"],
                "total_pnl": r.get("total_pnl"),
                "closed_trades": r.get("closed_trades"),
                "win_rate": r.get("win_rate"),
            }
            for r in rows
        ],
    }

    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
