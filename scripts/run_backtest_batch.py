#!/usr/bin/env python3
"""Run ``run_backtest.py`` for each weekday between --start and --end (inclusive).

Writes ``backtest_results.json`` under ``data/daily_news/<date>/processed/`` per day.
Skip dates where results already exist unless ``--force``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_iso(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def weekdays_between(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch historical backtests (weekdays).")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["SPY", "QQQ", "IWM"],
        help="Symbols passed through to run_backtest (default SPY QQQ IWM)",
    )
    parser.add_argument("--fast", type=int, default=1, help="Loop delay seconds")
    parser.add_argument("--force", action="store_true", help="Re-run even if backtest_results.json exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    start_d = parse_iso(args.start)
    end_d = parse_iso(args.end)
    if end_d < start_d:
        raise SystemExit("--end must be >= --start")

    days = weekdays_between(start_d, end_d)
    script = PROJECT_ROOT / "scripts" / "run_backtest.py"
    ok = 0
    skipped = 0
    failed = 0

    for d in days:
        ds = d.isoformat()
        out_json = PROJECT_ROOT / "data" / "daily_news" / ds / "processed" / "backtest_results.json"
        if out_json.exists() and not args.force:
            print(f"skip {ds} (exists {out_json})")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(script),
            "--date",
            ds,
            "--fast",
            str(args.fast),
        ]
        if args.symbols:
            cmd.append("--symbols")
            cmd.extend(args.symbols)

        print("RUN", " ".join(cmd))
        if args.dry_run:
            ok += 1
            continue

        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if proc.returncode != 0:
            print(f"FAIL {ds} exit={proc.returncode}", file=sys.stderr)
            failed += 1
        else:
            ok += 1

    print(f"done | ok={ok} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
