#!/usr/bin/env python3
"""
account_snapshot.py

Reads ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_TRADING from .env,
then saves account_details.json (in this directory) containing:
- current equity (balance)
- daily P/L (vs previous trading day's close)
- P/L over past 24h, 7d, 30d, YTD, and all-time
- other useful account fields from the /v2/account endpoint

Usage:
  python account_snapshot.py
"""

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
import pytz

ET = pytz.timezone("US/Eastern")


def parse_bool(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def safe_float(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def base_url_for_env(paper: bool) -> str:
    return "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"


def api_get(base_url: str, path: str, headers: dict, params: dict | None = None) -> dict:
    url = f"{base_url}{path}"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_account(base_url: str, headers: dict) -> dict:
    return api_get(base_url, "/v2/account", headers)


def get_portfolio_history(base_url: str, headers: dict, **params) -> dict:
    # Only include non-None params
    q = {k: v for k, v in params.items() if v is not None}
    return api_get(base_url, "/v2/account/portfolio/history", headers, q)


def nearest_index_leq(timestamps: list[int], target_ts: int) -> int | None:
    """Find index of last timestamp <= target_ts (timestamps assumed ascending)."""
    idx = None
    for i, ts in enumerate(timestamps):
        if ts <= target_ts:
            idx = i
        else:
            break
    return idx


def change_from_series(series: list[float]) -> dict | None:
    if not series:
        return None
    start, end = series[0], series[-1]
    if start is None or end is None:
        return None
    abs_change = end - start
    pct_change = (abs_change / start) if start else None
    return {"start": start, "end": end, "absolute": abs_change, "percent": pct_change}


def change_from_history(hist: dict, start_ts: int | None = None) -> dict | None:
    eq = hist.get("equity") or []
    ts = hist.get("timestamp") or []
    if not eq or not ts:
        return None
    if start_ts is None:
        return change_from_series(eq)
    i = nearest_index_leq(ts, start_ts)
    if i is None:
        return None
    start, end = eq[i], eq[-1]
    abs_change = end - start
    pct_change = (abs_change / start) if start else None
    return {"start": start, "end": end, "absolute": abs_change, "percent": pct_change}


def main():
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    paper = parse_bool(os.getenv("ALPACA_PAPER_TRADING", "true"))

    if not api_key or not api_secret:
        raise SystemExit("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment.")

    base_url = base_url_for_env(paper)
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)

    # ---- Account snapshot ----
    account = get_account(base_url, headers)

    current_equity = safe_float(account.get("equity")) or 0.0
    # last_equity is equity as of prior trading day 16:00 ET
    last_equity = safe_float(account.get("last_equity")) or 0.0
    daily_abs = current_equity - last_equity if last_equity is not None else None
    daily_pct = (daily_abs / last_equity) if last_equity else None

    # ---- Portfolio history pulls for various windows ----
    # 24h (intraday with extended hours to include pre/post)
    try:
        start_24h = now_utc - timedelta(hours=24)
        hist_24h = get_portfolio_history(
            base_url,
            headers,
            date_start=start_24h.isoformat(),
            date_end=now_utc.isoformat(),
            timeframe="15Min",
            intraday_reporting="extended_hours",
            pnl_reset="trading_day",
        )
        change_24h = change_from_history(hist_24h, int(start_24h.timestamp()))
    except Exception:
        change_24h = None

    # 7 days (use daily points)
    try:
        start_7d = now_utc - timedelta(days=7)
        hist_7d = get_portfolio_history(
            base_url,
            headers,
            date_start=start_7d.isoformat(),
            date_end=now_utc.isoformat(),
            timeframe="1D",
        )
        change_7d = change_from_history(hist_7d, int(start_7d.timestamp()))
    except Exception:
        change_7d = None

    # 30 days (use daily points)
    try:
        start_30d = now_utc - timedelta(days=30)
        hist_30d = get_portfolio_history(
            base_url,
            headers,
            date_start=start_30d.isoformat(),
            date_end=now_utc.isoformat(),
            timeframe="1D",
        )
        change_30d = change_from_history(hist_30d, int(start_30d.timestamp()))
    except Exception:
        change_30d = None

    # YTD (from Jan 1 ET)
    try:
        ytd_start_et = ET.localize(datetime(now_et.year, 1, 1))
        ytd_start_utc = ytd_start_et.astimezone(timezone.utc)
        hist_ytd = get_portfolio_history(
            base_url,
            headers,
            date_start=ytd_start_utc.isoformat(),
            date_end=now_utc.isoformat(),
            timeframe="1D",
        )
        change_ytd = change_from_history(hist_ytd, int(ytd_start_utc.timestamp()))
    except Exception:
        change_ytd = None

    # All-time (try period=all, fall back to created_at)
    try:
        hist_all = get_portfolio_history(
            base_url,
            headers,
            period="all",
            timeframe="1W",  # coarse is fine for span
        )
        change_all = change_from_series(hist_all.get("equity") or [])
    except Exception:
        change_all = None
        try:
            created_at = account.get("created_at")
            if created_at:
                hist_all2 = get_portfolio_history(
                    base_url,
                    headers,
                    date_start=created_at,
                    date_end=now_utc.isoformat(),
                    timeframe="1D",
                )
                change_all = change_from_series(hist_all2.get("equity") or [])
        except Exception:
            pass

    # ---- Build structured output ----
    payload = {
        "generated_at": now_utc.isoformat(),
        "environment": "paper" if paper else "live",
        "account_overview": {
            "account_number": account.get("account_number"),
            "status": account.get("status"),
            "currency": account.get("currency"),
            "created_at": account.get("created_at"),
            "pattern_day_trader": account.get("pattern_day_trader"),
            "trading_blocked": account.get("trading_blocked"),
            "transfers_blocked": account.get("transfers_blocked"),
            "account_blocked": account.get("account_blocked"),
            "shorting_enabled": account.get("shorting_enabled"),
            "multiplier": account.get("multiplier"),
            "buying_power": safe_float(account.get("buying_power")),
            "regt_buying_power": safe_float(account.get("regt_buying_power")),
            "daytrading_buying_power": safe_float(account.get("daytrading_buying_power")),
            "cash": safe_float(account.get("cash")),
            "portfolio_value": safe_float(account.get("portfolio_value")),
            "equity": current_equity,
            "last_equity": last_equity,
            "daily_pl": {"absolute": daily_abs, "percent": daily_pct},
        },
        "performance": {
            "change_24h": change_24h,
            "change_7d": change_7d,
            "change_30d": change_30d,
            "change_ytd": change_ytd,
            "change_all_time": change_all,
        },
        "raw": {
            "account": account,  # helpful for auditing extra fields
        },
        "notes": {
            "24h_intraday_includes_extended_hours": True,
            "time_zone": "US/Eastern for daily reference; storage in UTC timestamps",
        },
    }

    out_path = Path(__file__).parent / "account_details.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
