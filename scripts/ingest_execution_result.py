#!/usr/bin/env python3
"""Validate and ingest a Robinhood MCP execution_result.json artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.execution_result_ingestor import ingest_execution_result  # noqa: E402
from src.utils.notifications import send_discord_alert  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="llm_advisor_execution_result_v1 JSON")
    parser.add_argument("--trade-plan", required=True, type=Path, help="Linked trade plan JSON")
    parser.add_argument("--processed-dir", type=Path, help="Run's processed telemetry directory")
    parser.add_argument("--push-bigquery", action="store_true", help="Insert the event into a BigQuery order_events table")
    parser.add_argument("--push-supabase", action="store_true", help="Run the existing EOD Supabase upsert for this date")
    parser.add_argument("--no-discord", action="store_true", help="Do not send the configured Discord alert")
    return parser.parse_args()


def _push_bigquery(event: dict[str, Any], run_date: str) -> None:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise SystemExit("google-cloud-bigquery is required for --push-bigquery") from exc
    project = os.getenv("GCP_PROJECT_ID", "").strip()
    dataset = os.getenv("GCP_DATASET_ID", "trading_signals").strip()
    if not project:
        raise SystemExit("GCP_PROJECT_ID is required for --push-bigquery")
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.order_events"
    schema = [
        bigquery.SchemaField("event_uid", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("run_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("event_ts", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("order_id", "STRING"),
        bigquery.SchemaField("details", "JSON"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]
    client.create_table(bigquery.Table(table_id, schema=schema), exists_ok=True)
    details = event["details"]
    row = {
        "event_uid": details["event_uid"],
        "run_date": run_date,
        "event_ts": event["ts"],
        "event_type": event["event_type"],
        "symbol": event["symbol"],
        "order_id": details.get("order", {}).get("order_id"),
        "details": details,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    errors = client.insert_rows_json(table_id, [row], row_ids=[details["event_uid"]])
    if errors:
        raise RuntimeError(f"BigQuery insert failed: {errors}")


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    trade_plan = json.loads(args.trade_plan.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(trade_plan, dict):
        raise SystemExit("Result and trade plan files must contain JSON objects")
    run_date = str(trade_plan.get("generated_at") or "")[:10]
    if len(run_date) != 10:
        raise SystemExit("Trade plan generated_at does not contain a run date")
    processed = args.processed_dir or (
        PROJECT_ROOT / "data" / "daily_news" / run_date / "processed"
    )
    events_path, archived, appended, event = ingest_execution_result(result, trade_plan, processed)

    if args.push_bigquery:
        _push_bigquery(event, run_date)
    if args.push_supabase:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_eod_aggregate.py"),
                "--date",
                run_date,
                "--no-bigquery",
                "--validate",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    if not args.no_discord:
        send_discord_alert(
            "Robinhood execution result ingested\n"
            f"Plan: {result['trade_plan_id']}\n"
            f"Status: {result['status']}\n"
            f"Order: {result.get('robinhood_order_id') or 'none'}"
        )
    print(json.dumps({
        "order_events": str(events_path),
        "archived_result": str(archived),
        "appended": appended,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
