#!/usr/bin/env python3
"""Premarket stdev/HTF analysis.

Computes:
  * Daily + hourly EMA slope, HH/LL regime, ATR percentile
  * 5m rolling mean/std (z-bands) and ATR
  * Optional Grok/GPT briefing (not wired yet)

Outputs JSON dict ready to write to S3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Data classes


@dataclass
class HTFStats:
    ema_slope_daily: float
    ema_slope_hourly: float
    hh_ll_tag: str
    atr_percentile_daily: float


@dataclass
class Bands5m:
    mu: float
    sigma: float
    atr_5m: float
    k: Dict[str, float]


@dataclass
class SymbolSnapshot:
    symbol: str
    htf: HTFStats
    bands_5m: Bands5m


# ---------------------------------------------------------------------------
# HTF helpers


def _ema_slope(series: pd.Series, span: int = 20) -> float:
    ema = series.ewm(span=span, adjust=False).mean()
    if len(ema) < 2:
        return 0.0
    return float(ema.iloc[-1] - ema.iloc[-2])


def _hh_ll_tag(highs: pd.Series, lows: pd.Series, lookback: int = 5) -> str:
    if len(highs) < lookback + 1:
        return "flat"
    recent_highs = highs.iloc[-(lookback + 1) :]
    recent_lows = lows.iloc[-(lookback + 1) :]
    hh = recent_highs.diff().dropna()
    ll = recent_lows.diff().dropna()
    higher = hh.gt(0).all()
    lower = ll.lt(0).all()
    if higher and not lower:
        return "HH"
    if lower and not higher:
        return "LL"
    return "mixed"


def _atr(series: pd.DataFrame, period: int = 14) -> pd.Series:
    high = series["high"]
    low = series["low"]
    close = series["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def _atr_percentile(series: pd.Series, value: float) -> float:
    if series.empty:
        return 50.0
    return float((series <= value).mean() * 100.0)


def compute_htf_stats(df_daily: pd.DataFrame, df_hourly: pd.DataFrame, atr_lookback: int = 60) -> HTFStats:
    ema_slope_d = _ema_slope(df_daily["close"], span=20)
    ema_slope_h = _ema_slope(df_hourly["close"], span=20)
    tag = _hh_ll_tag(df_daily["high"], df_daily["low"], lookback=5)
    atr_daily = _atr(df_daily)
    atr_pct = _atr_percentile(atr_daily.tail(atr_lookback), atr_daily.iloc[-1]) if len(atr_daily) else 50.0
    return HTFStats(
        ema_slope_daily=ema_slope_d,
        ema_slope_hourly=ema_slope_h,
        hh_ll_tag=tag,
        atr_percentile_daily=atr_pct,
    )


# ---------------------------------------------------------------------------
# 5m bands


def compute_5m_bands(df_5m: pd.DataFrame, window: int = 120,
                     k_mr: float = 1.2, k_tc: float = 1.8, k_filter: float = 0.6) -> Bands5m:
    closes = df_5m["close"]
    mu = closes.rolling(window=window, min_periods=window // 2).mean().iloc[-1]
    sigma = closes.rolling(window=window, min_periods=window // 2).std().iloc[-1]
    atr_5m_series = _atr(df_5m)
    atr_5m = atr_5m_series.iloc[-1]
    return Bands5m(
        mu=float(mu),
        sigma=float(sigma),
        atr_5m=float(atr_5m),
        k={"k1": k_mr, "k2": k_tc, "k3": k_filter},
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _to_symbol_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(float)
    try:
        sdf = df.xs(symbol, level="symbol").sort_index()
    except Exception:
        sdf = df.sort_index()
    sdf.index = pd.to_datetime(sdf.index)
    return sdf


def fetch_bars(client: StockHistoricalDataClient,
               symbols: List[str],
               start: datetime,
               end: datetime,
               timeframe: TimeFrame,
               feed: DataFeed,
               adjustment: Adjustment = Adjustment.SPLIT) -> Dict[str, pd.DataFrame]:
    req = StockBarsRequest(symbol_or_symbols=symbols,
                           timeframe=timeframe,
                           start=start,
                           end=end,
                           feed=feed,
                           adjustment=adjustment)
    data = client.get_stock_bars(req).df
    return {sym: _to_symbol_frame(data, sym) for sym in symbols}


def build_premarket_snapshot(symbol: str,
                             daily: pd.DataFrame,
                             hourly: pd.DataFrame,
                             m5: pd.DataFrame,
                             config: Optional[Dict] = None) -> SymbolSnapshot:
    cfg = config or {}
    htf = compute_htf_stats(daily, hourly,
                            atr_lookback=cfg.get("atr_lookback", 60))
    bands = compute_5m_bands(m5,
                             window=cfg.get("m5_window", 120),
                             k_mr=cfg.get("k1", 1.2),
                             k_tc=cfg.get("k2", 1.8),
                             k_filter=cfg.get("k3", 0.6))
    return SymbolSnapshot(symbol=symbol, htf=htf, bands_5m=bands)


def assemble_snapshot(symbol_snapshots: Iterable[SymbolSnapshot]) -> Dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": [
            {
                "symbol": snap.symbol,
                "htf": asdict(snap.htf),
                "bands_5m": asdict(snap.bands_5m),
            }
            for snap in symbol_snapshots
        ],
    }


def snapshot_to_json(snapshot: Dict) -> str:
    return json.dumps(snapshot, indent=2)


def run_premarket(symbols: List[str],
                  client: Optional[StockHistoricalDataClient] = None,
                  feed: Optional[str] = None,
                  config: Optional[Dict] = None) -> Dict:
    cfg = config or {}
    load_dotenv()
    feed_enum = DataFeed(feed.upper()) if feed else DataFeed.IEX
    cli = client or StockHistoricalDataClient()

    end_utc = datetime.now(timezone.utc)
    start_daily = end_utc - timedelta(days=cfg.get("daily_days", 180))
    start_hourly = end_utc - timedelta(days=cfg.get("hourly_days", 30))
    start_5m = end_utc - timedelta(days=cfg.get("m5_days", 7))

    daily_bars = fetch_bars(cli, symbols, start_daily, end_utc, TimeFrame.Day, feed_enum)
    hourly_bars = fetch_bars(cli, symbols, start_hourly, end_utc, TimeFrame.Hour, feed_enum)
    m5_bars = fetch_bars(cli, symbols, start_5m, end_utc,
                         TimeFrame(5, TimeFrameUnit.Minute), feed_enum)

    snapshots: List[SymbolSnapshot] = []
    for sym in symbols:
        if daily_bars[sym].empty or hourly_bars[sym].empty or m5_bars[sym].empty:
            continue
        snapshots.append(build_premarket_snapshot(sym,
                                                  daily=daily_bars[sym],
                                                  hourly=hourly_bars[sym],
                                                  m5=m5_bars[sym],
                                                  config=cfg))

    return assemble_snapshot(snapshots)


# Placeholder for optional LLM briefing (to be integrated later)
def build_briefing_payload(snapshot: Dict) -> str:
    return (
        "Daily bias summary:\n"
        + "\n".join(
            f"- {sym['symbol']}: ema_slope_daily={sym['htf']['ema_slope_daily']:.4f}, "
            f"regime={sym['htf']['hh_ll_tag']}, atr_pct={sym['htf']['atr_percentile_daily']:.1f}%"
            for sym in snapshot["symbols"]
        )
    )


__all__ = [
    "HTFStats",
    "Bands5m",
    "SymbolSnapshot",
    "fetch_bars",
    "build_premarket_snapshot",
    "assemble_snapshot",
    "run_premarket",
    "snapshot_to_json",
    "build_briefing_payload",
]


if __name__ == "__main__":
    import argparse
    from config.settings import WATCHLIST

    parser = argparse.ArgumentParser(description="Premarket stdev snapshot")
    parser.add_argument("--symbols", nargs="*", default=WATCHLIST[:3], help="Symbols to process")
    parser.add_argument("--feed", default=None, help="Alpaca feed (iex|sip)")
    parser.add_argument("--output", default=None, help="Optional file to write JSON")
    args = parser.parse_args()

    snapshot = run_premarket(args.symbols, feed=args.feed)
    text = snapshot_to_json(snapshot)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote snapshot for {len(snapshot['symbols'])} symbols → {args.output}")
    else:
        print(text)

