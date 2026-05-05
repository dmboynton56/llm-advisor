#!/usr/bin/env python3
"""Rolling sync from BigQuery `trades` + `live_loop_logs` into Supabase serving tables."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_eod_aggregate as eod

LOGGER = logging.getLogger("sync_bq_to_supabase")


def rolling_run_dates(days: int) -> list[str]:
    today = datetime.now(timezone.utc).date()
    return [(today - timedelta(days=i)).isoformat() for i in range(max(days, 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description="BQ → Supabase sync for LLM Advisor telemetry.")
    parser.add_argument("--days", type=int, default=14, help="Calendar days to include (default 14).")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    dataset_id = os.getenv("GCP_DATASET_ID", "trading_signals").strip()
    if not project_id:
        LOGGER.warning("GCP_PROJECT_ID unset; nothing to sync.")
        sys.exit(0)

    dates = rolling_run_dates(args.days)
    runs, trades, heartbeats = eod.fetch_bq_ingest_for_dates(project_id, dataset_id, dates)
    runs = eod.dedupe_runs(runs)
    trades = eod.dedupe_trades(trades)
    heartbeats = eod.dedupe_heartbeats(heartbeats)

    LOGGER.info(
        "BQ snapshot | runs=%d trades=%d heartbeats=%d", len(runs), len(trades), len(heartbeats)
    )

    if not (runs or trades or heartbeats):
        LOGGER.warning("No rows returned from BigQuery for window; exiting without writes.")
        sys.exit(0)

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = eod.connect_supabase()
    try:
        with conn, conn.cursor() as cur:
            eod.upsert_runs(cur, runs, now_iso)
            eod.upsert_trades(cur, trades, now_iso)
            eod.upsert_heartbeats(cur, heartbeats, now_iso)
            if args.validate:
                checks = eod.validate(cur)
                LOGGER.info("Validation checks: %s", json.dumps(checks, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
