#!/usr/bin/env python3
"""Warn when booked lifecycle PnL diverges from the flat broker account."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import psycopg2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = psycopg2.connect(
        host=os.environ["SUPABASE_DB_HOST"],
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.environ["SUPABASE_DB_PASSWORD"],
        port=int((os.getenv("SUPABASE_DB_PORT") or "5432").strip()),
        sslmode="require",
    )
    try:
        with conn, conn.cursor() as cur:
            if args.date:
                cur.execute(
                    """
                    SELECT reconciliation_date,booked_realized_pnl,broker_daily_pnl,
                           pnl_gap,tolerance,status
                    FROM llm_advisor_broker_reconciliation_daily
                    WHERE reconciliation_date = %s
                    """,
                    (args.date,),
                )
            else:
                cur.execute(
                    """
                    SELECT reconciliation_date,booked_realized_pnl,broker_daily_pnl,
                           pnl_gap,tolerance,status
                    FROM llm_advisor_broker_reconciliation_daily
                    ORDER BY reconciliation_date DESC LIMIT 1
                    """
                )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        print("No broker reconciliation row found.")
        return
    date, booked, broker, gap, tolerance, status = row
    print(
        f"Broker reconciliation {date}: booked={booked} broker={broker} "
        f"gap={gap} tolerance={tolerance} status={status}"
    )
    if status != "alert":
        return

    message = (
        f"⚠️ **LLM Advisor broker reconciliation gap**\n"
        f"Date: {date}\nBooked lifecycle PnL: ${float(booked):,.2f}\n"
        f"Broker daily PnL: ${float(broker):,.2f}\n"
        f"Gap: ${float(gap):,.2f} (tolerance ${float(tolerance):,.2f})"
    )
    print(f"::warning::{message.replace(chr(10), ' | ')}")
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    request = urllib.request.Request(
        webhook,
        data=json.dumps({"content": message}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status >= 300:
            raise SystemExit(f"Discord reconciliation alert failed: HTTP {response.status}")


if __name__ == "__main__":
    main()
