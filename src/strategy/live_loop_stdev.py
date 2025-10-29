#!/usr/bin/env python3
"""Stdev-driven live loop (time-compressed test + live).

This is a simplified pipeline that:
  * Seeds HTF bias + 5m bands via premarket snapshot
  * Maintains rolling mean/std (z-score) on 1m closes
  * Implements basic Mean-Reversion (MR) and Trend-Continuation (TC) state machines
  * Logs signals/trades without hitting a broker yet

NOTE: This is an initial scaffolding. Execution + broker integration will follow once
thresholds are calibrated.
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import WATCHLIST
from src.features.stdev_features import RollingStats
from src.data_processing.premarket_stdev import run_premarket

ET = pytz.timezone("US/Eastern")


@dataclass
class Thresholds:
    k1: float = 1.2
    k2: float = 1.8
    k3: float = 0.6
    atr_mult: float = 1.4
    rr_min: float = 1.5
    atr_percentile_cap: float = 85.0


@dataclass
class TradePlan:
    setup: str
    side: str
    entry_price: float
    sl_price: float
    tp_price: float
    triggered_at: datetime


@dataclass
class SymbolState:
    symbol: str
    rolling: RollingStats
    htf_bias: str
    ema_slope_hourly: float
    atr_percentile: float
    atr_5m: float
    thresholds: Thresholds
    status: str = "idle"  # idle | mr_armed | mr_triggered | tc_armed | tc_triggered
    side: Optional[str] = None
    last_mu: float = 0.0
    last_sigma: float = 0.0
    last_z: float = 0.0
    armed_z: Optional[float] = None
    trade: Optional[TradePlan] = None
    last_update_utc: Optional[str] = None

    def should_gate(self) -> bool:
        return self.atr_percentile <= self.thresholds.atr_percentile_cap


def _bars_to_records(df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
    if df is None or isinstance(df, list):
        return [] if not df else df  # already list
    if df.empty:
        return []
    try:
        sdf = df.xs(symbol, level="symbol").sort_index()
    except Exception:
        sdf = df.sort_index()
    sdf = sdf.reset_index().rename(columns={"timestamp": "t"})
    records = []
    for _, row in sdf.iterrows():
        records.append({
            "t": pd.to_datetime(row["t"]).isoformat(),
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row.get("volume", 0.0)),
        })
    return records


class Clients:
    def __init__(self):
        load_dotenv()
        k = os.getenv("ALPACA_API_KEY")
        s = os.getenv("ALPACA_SECRET_KEY")
        if not (k and s):
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")
        self.data = StockHistoricalDataClient(k, s)
        feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
        self.feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX


def fetch_window_bars(clients: Clients,
                      symbols: List[str],
                      start_utc: datetime,
                      end_utc: datetime) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    req_1m = StockBarsRequest(symbol_or_symbols=symbols,
                              timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                              start=start_utc,
                              end=end_utc,
                              feed=clients.feed,
                              adjustment=Adjustment.SPLIT)
    req_5m = StockBarsRequest(symbol_or_symbols=symbols,
                              timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                              start=start_utc,
                              end=end_utc,
                              feed=clients.feed,
                              adjustment=Adjustment.SPLIT)
    df_1m = clients.data.get_stock_bars(req_1m).df
    df_5m = clients.data.get_stock_bars(req_5m).df
    return {
        sym: {
            "bars_1m": _bars_to_records(df_1m, sym),
            "bars_5m": _bars_to_records(df_5m, sym),
        }
        for sym in symbols
    }


class TestBarCache:
    def __init__(self, clients: Clients, symbols: List[str], preopen_utc: datetime, end_utc: datetime):
        req_1m = StockBarsRequest(symbol_or_symbols=symbols,
                                  timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                  start=preopen_utc,
                                  end=end_utc,
                                  feed=clients.feed,
                                  adjustment=Adjustment.SPLIT)
        req_5m = StockBarsRequest(symbol_or_symbols=symbols,
                                  timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                  start=preopen_utc,
                                  end=end_utc,
                                  feed=clients.feed,
                                  adjustment=Adjustment.SPLIT)
        self.df_1m = clients.data.get_stock_bars(req_1m).df
        self.df_5m = clients.data.get_stock_bars(req_5m).df
        self.symbols = symbols

    def slice(self, start_utc: datetime, end_utc: datetime) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        df1 = self.df_1m
        df5 = self.df_5m
        if df1 is not None and not df1.empty:
            df1 = df1.reset_index()
            df1 = df1[(df1["timestamp"] >= pd.Timestamp(start_utc)) & (df1["timestamp"] <= pd.Timestamp(end_utc))]
            df1 = df1.set_index(["symbol", "timestamp"]).sort_index()
        if df5 is not None and not df5.empty:
            df5 = df5.reset_index()
            df5 = df5[(df5["timestamp"] >= pd.Timestamp(start_utc)) & (df5["timestamp"] <= pd.Timestamp(end_utc))]
            df5 = df5.set_index(["symbol", "timestamp"]).sort_index()
        return {
            sym: {
                "bars_1m": _bars_to_records(df1, sym) if df1 is not None else [],
                "bars_5m": _bars_to_records(df5, sym) if df5 is not None else [],
            }
            for sym in self.symbols
        }


def seed_state(symbol: str,
               pre_snapshot: Dict,
               bars_1m: List[Dict[str, Any]],
               thresholds: Thresholds,
               window: int) -> SymbolState:
    htf = next((s for s in pre_snapshot["symbols"] if s["symbol"] == symbol), None)
    if not htf:
        raise RuntimeError(f"No premarket snapshot for {symbol}")
    closes = [bar["c"] for bar in bars_1m[-window:]]
    stats = RollingStats.from_seed(closes, window=window)
    mu = stats.mean
    sigma = stats.std
    z = 0.0 if sigma == 0 else (closes[-1] - mu) / sigma if closes else 0.0
    return SymbolState(
        symbol=symbol,
        rolling=stats,
        htf_bias=htf["htf"]["hh_ll_tag"],
        ema_slope_hourly=htf["htf"]["ema_slope_hourly"],
        atr_percentile=htf["htf"]["atr_percentile_daily"],
        atr_5m=htf["bands_5m"]["atr_5m"],
        thresholds=thresholds,
        last_mu=mu,
        last_sigma=sigma,
        last_z=z,
    )


def evaluate_signals(state: SymbolState, price: float, now_utc: datetime) -> None:
    mu, sigma, z = state.rolling.update(price)
    state.last_mu = mu
    state.last_sigma = sigma
    state.last_z = z
    state.last_update_utc = now_utc.isoformat()

    if not state.should_gate():
        state.status = "idle"
        state.side = None
        state.trade = None
        return

    # Mean reversion arming
    if state.status == "idle":
        if abs(z) >= state.thresholds.k1:
            state.status = "mr_armed"
            state.side = "long" if z < 0 else "short"
            state.armed_z = z
            return
        # Trend continuation arming (requires slope alignment)
        if state.ema_slope_hourly > 0 and z >= state.thresholds.k2:
            state.status = "tc_armed"
            state.side = "long"
            state.armed_z = z
            return
        if state.ema_slope_hourly < 0 and z <= -state.thresholds.k2:
            state.status = "tc_armed"
            state.side = "short"
            state.armed_z = z
            return

    # MR trigger
    if state.status == "mr_armed":
        if state.side == "long" and z >= -state.thresholds.k3:
            plan_trade(state, "MR", price, now_utc)
            return
        if state.side == "short" and z <= state.thresholds.k3:
            plan_trade(state, "MR", price, now_utc)
            return
        if abs(z) < state.thresholds.k1 / 2:
            state.status = "idle"
            state.side = None
            state.armed_z = None

    # TC trigger
    if state.status == "tc_armed":
        if state.side == "long" and z >= state.thresholds.k3:
            plan_trade(state, "TC", price, now_utc)
            return
        if state.side == "short" and z <= -state.thresholds.k3:
            plan_trade(state, "TC", price, now_utc)
            return
        if abs(z) < state.thresholds.k1:
            state.status = "idle"
            state.side = None
            state.armed_z = None


def plan_trade(state: SymbolState, setup: str, price: float, now_utc: datetime) -> None:
    atr_offset = state.thresholds.atr_mult * state.atr_5m
    if state.side == "long":
        sl = price - atr_offset
        tp = price + state.thresholds.rr_min * atr_offset
    else:
        sl = price + atr_offset
        tp = price - state.thresholds.rr_min * atr_offset
    state.trade = TradePlan(
        setup=setup,
        side=state.side or "long",
        entry_price=price,
        sl_price=sl,
        tp_price=tp,
        triggered_at=now_utc,
    )
    state.status = f"{setup.lower()}_triggered"


def log_tick(path: Path, sym_states: Dict[str, SymbolState], extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbols": {
            sym: {
                "status": st.status,
                "side": st.side,
                "z": st.last_z,
                "mu": st.last_mu,
                "sigma": st.last_sigma,
                "trade": {
                    "setup": st.trade.setup,
                    "side": st.trade.side,
                    "entry": st.trade.entry_price,
                    "sl": st.trade.sl_price,
                    "tp": st.trade.tp_price,
                    "triggered_at": st.trade.triggered_at.isoformat(),
                } if st.trade else None,
            }
            for sym, st in sym_states.items()
        },
        **extra,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def parse_hhmm(s: str) -> Tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def et_dt(day: datetime.date, hhmm: str) -> datetime:
    h, m = parse_hhmm(hhmm)
    return ET.localize(datetime(day.year, day.month, day.day, h, m))


def to_utc(dt_et: datetime) -> datetime:
    return dt_et.astimezone(timezone.utc)


GROK_API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/responses")


def call_grok_json(model: str, prompt: str, schema: dict, timeout: Optional[float] = None) -> Dict[str, Any]:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object", "json_schema": schema},
    }
    resp = requests.post(GROK_API_URL, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Grok call failed: HTTP {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    raw = data.get("output_text")
    if not raw and isinstance(data.get("output"), list):
        blocks = data["output"]
        if blocks and isinstance(blocks[0].get("content"), list):
            raw = "".join(chunk.get("text", "") for chunk in blocks[0]["content"])
    if not raw:
        raise RuntimeError("Grok response missing text")
    return json.loads(raw)


def send_trend_check(symbol_states: Dict[str, SymbolState], trend_log: Path, model: str = "grok-4") -> None:
    schema = {
        "type": "object",
        "properties": {
            "trend_bias": {"type": "string"},
            "confidence": {"type": "integer"},
            "notes": {"type": "string"},
        },
        "required": ["trend_bias", "confidence", "notes"],
    }
    for sym, st in symbol_states.items():
        prompt = (
            "## Trend Review\n"
            f"Symbol: {sym}\n"
            f"HTF bias: {st.htf_bias}, ema_slope_hourly={st.ema_slope_hourly:.4f}, atr_percentile={st.atr_percentile:.1f}\n"
            f"Latest z={st.last_z:.2f}, mu={st.last_mu:.2f}, sigma={st.last_sigma:.2f}\n"
            f"State: {st.status}, side={st.side}\n"
            "Return JSON {trend_bias, confidence, notes}."
        )
        try:
            review = call_grok_json(model, prompt, schema, timeout=20.0)
        except Exception as exc:
            review = {"trend_bias": "unknown", "confidence": 0, "notes": f"error: {exc}"}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "review": review,
        }
        trend_log.parent.mkdir(parents=True, exist_ok=True)
        with trend_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD test session")
    ap.add_argument("--fast", type=int, default=int(os.getenv("FAST_SECS", "15")))
    ap.add_argument("--symbols", nargs="*", default=WATCHLIST[:3])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    test_mode = bool(args.date)
    day_str = args.date or datetime.now(ET).strftime("%Y-%m-%d")
    day_date = datetime.strptime(day_str, "%Y-%m-%d").date()

    run_start_et = et_dt(day_date, os.getenv("RUN_START_ET", "09:30"))
    run_end_et = et_dt(day_date, os.getenv("RUN_END_ET", "11:00"))
    preopen_et = et_dt(day_date, os.getenv("PREOPEN_ET", "06:30"))
    run_start_utc = to_utc(run_start_et)
    run_end_utc = to_utc(run_end_et)
    preopen_utc = to_utc(preopen_et)

    clients = Clients()
    symbols = [s.upper() for s in args.symbols]

    print(f"Premarket snapshot for {symbols}")
    snapshot = run_premarket(symbols, feed=clients.feed.name)

    paths = {
        "log": PROJECT_ROOT / "data" / "daily_news" / day_str / "processed" / "stdev_live_log.jsonl",
        "trend_log": PROJECT_ROOT / "data" / "daily_news" / day_str / "processed" / "stdev_trend_reviews.jsonl",
    }

    cache = None
    if test_mode:
        cache = TestBarCache(clients, symbols, preopen_utc, run_end_utc)

    state_by_symbol: Dict[str, SymbolState] = {}

    # seed states with last window of 1m bars from preopen->run_start
    seed_start = run_start_utc - timedelta(minutes=240)
    seed_bars = fetch_window_bars(clients, symbols, seed_start, run_start_utc)
    thresholds = Thresholds()
    window = 120
    for sym in symbols:
        bars_1m = seed_bars[sym]["bars_1m"]
        if len(bars_1m) < window:
            raise RuntimeError(f"Insufficient seed data for {sym}")
        state_by_symbol[sym] = seed_state(sym, snapshot, bars_1m, thresholds, window)

    sim_et = run_start_et
    fast_secs = args.fast
    trend_interval = timedelta(minutes=int(os.getenv("TREND_CHECK_MINUTES", "15")))
    last_trend_check_utc: Optional[datetime] = None
    while True:
        if test_mode:
            if sim_et > run_end_et:
                print("[TEST] Session complete")
                break
            cur_utc = to_utc(sim_et)
            bars = cache.slice(cur_utc - timedelta(minutes=30), cur_utc)
            label = sim_et.strftime("%H:%M:%S ET")
        else:
            wall_et = datetime.now(ET)
            if wall_et < run_start_et:
                wait = max(1, int((run_start_et - wall_et).total_seconds()))
                print(f"[LIVE] Sleeping {wait}s until open")
                time.sleep(min(wait, fast_secs))
                continue
            if wall_et > run_end_et:
                print("[LIVE] Session complete")
                break
            cur_utc = wall_et.astimezone(timezone.utc)
            bars = fetch_window_bars(clients, symbols, cur_utc - timedelta(minutes=30), cur_utc)
            label = wall_et.strftime("%H:%M:%S ET")

        now_utc = datetime.now(timezone.utc)
        for sym in symbols:
            latest_1m = bars[sym]["bars_1m"][-1]["c"] if bars[sym]["bars_1m"] else None
            if latest_1m is None:
                continue
            evaluate_signals(state_by_symbol[sym], latest_1m, now_utc)

        if last_trend_check_utc is None or (now_utc - last_trend_check_utc) >= trend_interval:
            send_trend_check(state_by_symbol, paths["trend_log"])
            last_trend_check_utc = now_utc

        log_tick(paths["log"], state_by_symbol, {"label": label})
        if not test_mode:
            time.sleep(fast_secs)
        else:
            sim_et += timedelta(minutes=1)
            time.sleep(fast_secs)


if __name__ == "__main__":
    main()

