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
                           pnl_gap,tolerance,status,details
                    FROM llm_advisor_broker_reconciliation_daily
                    WHERE reconciliation_date = %s
                    """,
                    (args.date,),
                )
            else:
                cur.execute(
                    """
                    SELECT reconciliation_date,booked_realized_pnl,broker_daily_pnl,
                           pnl_gap,tolerance,status,details
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
    date, booked, broker, gap, tolerance, status, details = row
    
    # Parse details JSON for additional context
    details_dict = details if isinstance(details, dict) else {}
    broker_equity = details_dict.get("broker_equity")
    broker_last_equity = details_dict.get("broker_last_equity")
    gap_hints = details_dict.get("gap_analysis_hints", [])
    
    print(
        f"Broker reconciliation {date}: booked={booked} broker={broker} "
        f"gap={gap} tolerance={tolerance} status={status}"
    )
    if broker_equity and broker_last_equity:
        print(f"  Broker equity: {broker_equity}, last_equity: {broker_last_equity}")
    if gap_hints:
        print(f"  Gap analysis hints: {', '.join(gap_hints)}")
    
    if status != "alert":
        return

    message = (
        f"⚠️ **LLM Advisor broker reconciliation gap**\n"
        f"Date: {date}\nBooked lifecycle PnL: ${float(booked):,.2f}\n"
        f"Broker daily PnL: ${float(broker):,.2f}\n"
        f"Gap: ${float(gap):,.2f} (tolerance ${float(tolerance):,.2f})\n\n"
        f"_Note: Broker daily P&L includes open position MTM, fees, and adjustments. "
        f"Booked P&L only tracks realized fills. Small gaps (<$500) are often expected._"
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
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status >= 300:
                print(f"WARNING: Discord reconciliation alert failed: HTTP {response.status}")
    except Exception as e:
        print(f"WARNING: Discord reconciliation alert failed: {e}")


if __name__ == "__main__":
    main()
