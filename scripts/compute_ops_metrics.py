#!/usr/bin/env python3
"""Compute ops-dashboard metrics over a date range and (optionally) upsert the
daily rollup row into llm_advisor_ops_metrics_daily.

Sources:
  --source supabase (default): reads the enriched llm_advisor_* serving tables.
  --source bq: reads BigQuery trades directly (order events / snapshots come
               from Supabase only, so funnel/equity sections are empty).

Usage:
  python scripts/compute_ops_metrics.py --days 30 --output ops_metrics.json
  python scripts/compute_ops_metrics.py --metric-date 2026-07-02 --upsert
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_eod_aggregate as eod
from src.analytics.ops_metrics import compute_ops_metrics

LOGGER = logging.getLogger("compute_ops_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LLM-Advisor ops metrics.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default 30).")
    parser.add_argument("--start", help="Range start YYYY-MM-DD (overrides --days).")
    parser.add_argument("--end", help="Range end YYYY-MM-DD (default today UTC).")
    parser.add_argument(
        "--metric-date",
        help="Date key for the ops_metrics_daily upsert (default: range end).",
    )
    parser.add_argument("--source", choices=("supabase", "bq"), default="supabase")
    parser.add_argument("--output", help="Write payload JSON to this path.")
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Upsert payload into llm_advisor_ops_metrics_daily.",
    )
    return parser.parse_args()


def _row_to_dict(cur, row) -> dict[str, Any]:
    return {desc[0]: value for desc, value in zip(cur.description, row)}


def fetch_supabase_rows(
    start_date: str, end_date: str
) -> tuple[list[dict], list[dict], list[dict]]:
    conn = eod.connect_supabase()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_uid, run_date::text AS run_date, order_id, symbol, side, qty,
                       entry_price, stop_loss, take_profit, entry_time, exit_time, exit_price,
                       exit_reason, pnl, status, underlying_symbol, asset_class, setup_type,
                       option_dte, option_metadata
                FROM llm_advisor_backtest_trades
                WHERE run_date BETWEEN %s AND %s
                """,
                (start_date, end_date),
            )
            trades = [_row_to_dict(cur, row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT event_uid, run_date::text AS run_date, event_ts, event_type, symbol,
                       setup_type, side, order_id, details
                FROM llm_advisor_order_events
                WHERE run_date BETWEEN %s AND %s
                """,
                (start_date, end_date),
            )
            order_events = [_row_to_dict(cur, row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT snapshot_date::text AS snapshot_date, captured_at, equity, last_equity,
                       buying_power, daily_pnl, daily_pnl_pct, source
                FROM llm_advisor_account_snapshots
                WHERE snapshot_date BETWEEN %s AND %s
                ORDER BY captured_at
                """,
                (start_date, end_date),
            )
            snapshots = [_row_to_dict(cur, row) for row in cur.fetchall()]
    finally:
        conn.close()
    return trades, order_events, snapshots


def fetch_bq_rows(start_date: str, end_date: str) -> list[dict]:
    project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    dataset_id = os.getenv("GCP_DATASET_ID", "trading_signals").strip() or "trading_signals"
    if not project_id:
        raise SystemExit("GCP_PROJECT_ID unset; cannot use --source bq.")

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    dates = [
        (start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)
    ]
    _, trade_rows, _ = eod.fetch_bq_ingest_for_dates(project_id, dataset_id, dates)
    trade_rows = eod.dedupe_trades(trade_rows)
    return [
        {
            "trade_uid": t.trade_uid,
            "run_date": t.run_date,
            "order_id": t.order_id,
            "symbol": t.symbol,
            "side": t.side,
            "qty": t.qty,
            "entry_price": t.entry_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl": t.pnl,
            "status": t.status,
            "underlying_symbol": t.underlying_symbol,
            "asset_class": t.asset_class,
            "setup_type": t.setup_type,
            "option_dte": t.option_dte,
            "option_metadata": t.option_metadata,
        }
        for t in trade_rows
    ]


def upsert_daily_rollup(metric_date: str, payload: dict[str, Any]) -> None:
    conn = eod.connect_supabase()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_advisor_ops_metrics_daily (metric_date, payload, updated_at)
                VALUES (%s, %s::jsonb, %s)
                ON CONFLICT (metric_date) DO UPDATE SET
                  payload = EXCLUDED.payload,
                  updated_at = EXCLUDED.updated_at
                """,
                (
                    metric_date,
                    json.dumps(payload, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    end_date = args.end or datetime.now(timezone.utc).date().isoformat()
    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        start_date = (end_dt - timedelta(days=max(args.days, 1))).isoformat()

    if args.source == "supabase":
        trades, order_events, snapshots = fetch_supabase_rows(start_date, end_date)
    else:
        trades = fetch_bq_rows(start_date, end_date)
        order_events, snapshots = [], []

    LOGGER.info(
        "Loaded rows | trades=%d order_events=%d account_snapshots=%d (%s..%s, source=%s)",
        len(trades),
        len(order_events),
        len(snapshots),
        start_date,
        end_date,
        args.source,
    )

    payload = compute_ops_metrics(
        trades=trades,
        order_events=order_events,
        account_snapshots=snapshots,
        start_date=start_date,
        end_date=end_date,
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["source"] = args.source

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        LOGGER.info("Wrote ops metrics to %s", out_path)
    else:
        print(json.dumps(payload, indent=2, default=str))

    if args.upsert:
        metric_date = args.metric_date or end_date
        upsert_daily_rollup(metric_date, payload)
        LOGGER.info("Upserted ops_metrics_daily row for %s", metric_date)


if __name__ == "__main__":
    main()
