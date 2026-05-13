#!/usr/bin/env python3
"""Market calendar check script.

Exits with code 0 if market is open today, code 1 if closed (holiday/weekend).
Used by GitHub Actions to skip runs on non-trading days.

Optional ``--scheduled-et-window`` (with ``GITHUB_OUTPUT`` set): scheduled runs
outside the America/New_York clock window set ``skip_remaining_steps=true`` and
exit 0 so workflow YAML can decide whether to fail or noop the rest of the job.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Tuple

import pandas_market_calendars as mcal
import pytz


def market_open_today() -> Tuple[bool, str]:
    """Return (True, msg) if NYSE session exists today, else (False, msg)."""
    nyse = mcal.get_calendar("NYSE")
    tz = pytz.timezone("America/New_York")
    now = datetime.now(tz)

    if now.weekday() >= 5:
        day = "Saturday" if now.weekday() == 5 else "Sunday"
        return False, f"Today is {day}. Market is closed."

    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        return False, f"Market is closed today ({now.date()}) per NYSE calendar (Holiday)."

    return True, f"Market is open today ({now.date()}). Proceeding..."


def _write_skip_remaining(skip: bool) -> None:
    gh = os.environ.get("GITHUB_OUTPUT")
    if not gh:
        return
    with open(gh, "a", encoding="utf-8") as f:
        f.write(f"skip_remaining_steps={'true' if skip else 'false'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduled-et-window",
        nargs=2,
        metavar=("START", "END"),
        help=(
            "HH:MM HH:MM in America/New_York (inclusive). "
            "Requires GITHUB_OUTPUT; writes skip_remaining_steps when outside window."
        ),
    )
    args = parser.parse_args()

    ok, msg = market_open_today()
    print(msg)
    if not ok:
        sys.exit(1)

    if args.scheduled_et_window:
        tz = pytz.timezone("America/New_York")
        now = datetime.now(tz)
        start = datetime.strptime(args.scheduled_et_window[0], "%H:%M").time()
        end = datetime.strptime(args.scheduled_et_window[1], "%H:%M").time()
        now_t = now.time()
        if not (start <= now_t <= end):
            print(
                f"Outside scheduled ET window ({args.scheduled_et_window[0]}–{args.scheduled_et_window[1]}). "
                f"Now {now.strftime('%H:%M:%S')} ET — setting skip_remaining_steps=true."
            )
            _write_skip_remaining(True)
            sys.exit(0)
        _write_skip_remaining(False)

    sys.exit(0)


if __name__ == "__main__":
    main()
