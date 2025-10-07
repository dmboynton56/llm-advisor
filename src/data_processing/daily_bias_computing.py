#!/usr/bin/env python3
"""
Compute SAME-DAY daily bias at/just after the 9:30 ET open, per symbol.

- Uses the exact same open-snapshot features as the training script:
  * Overnight gap vs prev close (prefers 09:30 minute OPEN; falls back to 09:29 premarket CLOSE if needed)
  * Premarket range/volume, sweep flags, premarket returns
  * Prior-day OHLC context + sweep flags
  * Daily ATR14% (computed on prior days), position of today's open within prior day's range
  * HTF context (1H/4H SMA20 + momentum) using only bars strictly before 9:30 ET
- Loads models and feature name lists from /models
- Writes JSON to data/daily_news/YYYY-MM-DD/raw/daily_bias.json  (ET date)
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from alpaca.data.enums import DataFeed, Adjustment

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# Project root (this file is under src/data_processing/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import WATCHLIST  # noqa: E402

ET_TZ = "US/Eastern"

# ---------------------- Version-safe time window helper ---------------------- #
def _bt(df: pd.DataFrame, start: str, end: str, inclusive: str = "both") -> pd.DataFrame:
    try:
        return df.between_time(start, end, inclusive=inclusive)
    except TypeError:
        out = df.between_time(start, end)
        st = pd.Timestamp(start).time()
        en = pd.Timestamp(end).time()
        if inclusive == "both":
            return out
        if inclusive == "left":
            return out[out.index.time != en]
        if inclusive == "right":
            return out[out.index.time != st]
        if inclusive == "neither":
            return out[(out.index.time != st) & (out.index.time != en)]
        return out

# ---------------------- Env + clients ---------------------- #
def _load_env():
    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path if dotenv_path.exists() else None)

def _init_alpaca():
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca API credentials in .env")
    paper = (os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true")
    feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX
    trading = TradingClient(api_key, api_secret, paper=paper)
    data = StockHistoricalDataClient(api_key, api_secret)
    return trading, data, feed

def _as_single_symbol_df(bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if bars_df is None or bars_df.empty:
        return pd.DataFrame()
    if bars_df.index.nlevels == 2:
        try:
            df = bars_df.xs(symbol, level=0).copy()
        except KeyError:
            return pd.DataFrame()
    else:
        df = bars_df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    return df.sort_index()

# ---------------------- Data fetch helpers ---------------------- #
def _fetch_today_frames(symbol: str, data_client, feed: DataFeed):
    """
    Grab just enough history to compute today's open-snapshot features.
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=120)  # enough for ATR14, SMA20, etc.

    # Daily
    req_d = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_utc,
        end=now_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    ddf = _as_single_symbol_df(data_client.get_stock_bars(req_d).df, symbol).tz_convert(ET_TZ)

    # Minute
    req_m = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_utc,
        end=now_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    mdf = _as_single_symbol_df(data_client.get_stock_bars(req_m).df, symbol).tz_convert(ET_TZ)

    # 1H
    req_h1 = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,
        start=start_utc,
        end=now_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    h1df = _as_single_symbol_df(data_client.get_stock_bars(req_h1).df, symbol).tz_convert(ET_TZ)

    # 4H
    req_h4 = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(4, TimeFrameUnit.Hour),
        start=start_utc,
        end=now_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    h4df = _as_single_symbol_df(data_client.get_stock_bars(req_h4).df, symbol).tz_convert(ET_TZ)

    return ddf, mdf, h1df, h4df


# ---------------------- Feature engineering (mirror training) ---------------------- #
def _premarket_slice(mdf_et: pd.DataFrame, day_date):
    return _bt(mdf_et[mdf_et.index.date == day_date], "04:00", "09:29", inclusive="both")

def _first_rth_minute(mdf_et: pd.DataFrame, day_date):
    rth = _bt(mdf_et[mdf_et.index.date == day_date], "09:30", "16:00", inclusive="left")
    if rth.empty: return None
    return rth.iloc[0]

def _daily_atr14_pct(ddf_et: pd.DataFrame, up_to_date) -> float:
    hist = ddf_et[ddf_et.index.date < up_to_date].tail(60).copy()
    if hist.empty or len(hist) < 15:
        return np.nan
    prev_close = hist["close"].shift(1)
    tr = np.maximum(hist["high"] - hist["low"],
                    np.maximum((hist["high"] - prev_close).abs(),
                               (hist["low"] - prev_close).abs()))
    atr14 = tr.rolling(14).mean().iloc[-1]
    ref = hist["close"].iloc[-1]
    return float((atr14 / max(ref, 1e-10)) * 100.0)

def _compute_open_features_today(symbol: str, ddf_et, mdf_et, h1df, h4df, feature_names):
    today = pd.Timestamp.now(tz=ET_TZ).date()

    # Use actual prior sessions from the data (handles weekends/holidays)
    ddf_prior = ddf_et[ddf_et.index.date < today]
    if len(ddf_prior) < 2:
        return None, {"error": "insufficient_daily_history"}, None

    prev  = ddf_prior.tail(1)
    prev2 = ddf_prior.tail(2).head(1)


    prev_close = float(prev.iloc[-1]["close"])
    prev_high  = float(prev.iloc[-1]["high"])
    prev_low   = float(prev.iloc[-1]["low"])
    prev_open  = float(prev.iloc[-1]["open"])
    prev_range = max(prev_high - prev_low, 1e-10)
    prev_day_range_pct = (prev_range / max(prev_close, 1e-10)) * 100.0
    prev_day_body_pct = (abs(prev.iloc[-1]["close"] - prev_open) / prev_range) * 100.0
    prev_day_bull = int(prev.iloc[-1]["close"] > prev_open)

    prev2_high = float(prev2.iloc[-1]["high"])
    prev2_low  = float(prev2.iloc[-1]["low"])
    prev_day_swept_prior_high = int(prev_high > prev2_high)
    prev_day_swept_prior_low  = int(prev_low < prev2_low)

    pre = _premarket_slice(mdf_et, today)
    if pre.empty:
        pre_hi = np.nan; pre_lo = np.nan; pre_vol = 0.0
        pre_first = None; pre_last = None
    else:
        pre_hi = float(pre["high"].max())
        pre_lo = float(pre["low"].min())
        pre_vol = float(pre["volume"].sum())
        pre_first = pre.iloc[0]
        pre_last  = pre.iloc[-1]

    pre_rng_pct = ((pre_hi - pre_lo) / max(prev_close, 1e-10) * 100.0
                   if not (np.isnan(pre_hi) or np.isnan(pre_lo)) else 0.0)
    pre_close_vs_prev_close_pct = (((float(pre_last["close"]) - prev_close) / max(prev_close, 1e-10) * 100.0)
                                   if pre_last is not None else 0.0)
    pre_return_pct = (((float(pre_last["close"]) - float(pre_first["open"])) / max(float(pre_first["open"]), 1e-10) * 100.0)
                      if (pre_first is not None and pre_last is not None) else 0.0)
    pre_sweep_prev_high = int((not np.isnan(pre_hi)) and (pre_hi > prev_high))
    pre_sweep_prev_low  = int((not np.isnan(pre_lo)) and (pre_lo < prev_low))

    # Prefer 09:30 OPEN, else last premarket CLOSE as proxy if 09:30 isn't published yet
    open_source = "09:30_minute_open"
    m0930 = _first_rth_minute(mdf_et, today)
    if m0930 is not None and not np.isnan(m0930["open"]):
        open_0930 = float(m0930["open"])
    else:
        open_source = "09:29_close_proxy"
        if pre_last is None or np.isnan(pre_last["close"]):
            return None, {"error": "no_open_or_premarket"}, None
        open_0930 = float(pre_last["close"])

    open_pos_in_prev_range = float((open_0930 - prev_low) / max(prev_range, 1e-10))
    open_to_prev_high_pct_rng = float((prev_high - open_0930) / max(prev_range, 1e-10) * 100.0)
    open_to_prev_low_pct_rng  = float((open_0930 - prev_low) / max(prev_range, 1e-10) * 100.0)
    overnight_gap_pct = float((open_0930 - prev_close) / max(prev_close, 1e-10) * 100.0)
    atr14_pct = _daily_atr14_pct(ddf_et, today)

    open_ts = pd.Timestamp.combine(pd.Timestamp(today), pd.Timestamp("09:30").time()).tz_localize(ET_TZ)
    h1 = h1df[h1df.index < open_ts]
    h4 = h4df[h4df.index < open_ts]
    if h1.empty or h4.empty:
        return None, {"error": "insufficient_htf_bars"}, None

    h1_close = h1["close"]; h4_close = h4["close"]
    h1_sma20 = h1_close.rolling(20).mean()
    h4_sma20 = h4_close.rolling(20).mean()
    if np.isnan(h1_sma20.iloc[-1]) or np.isnan(h4_sma20.iloc[-1]):
        return None, {"error": "insufficient_sma_history"}, None

    h1_close_vs_sma20_pct = float(((h1_close.iloc[-1] - h1_sma20.iloc[-1]) / max(h1_sma20.iloc[-1], 1e-10)) * 100.0)
    h4_close_vs_sma20_pct = float(((h4_close.iloc[-1] - h4_sma20.iloc[-1]) / max(h4_sma20.iloc[-1], 1e-10)) * 100.0)
    h1_mom_5bars_pct = float(((h1_close.iloc[-1] - (h1_close.iloc[-6] if len(h1_close) > 5 else h1_close.iloc[0])) /
                               max((h1_close.iloc[-6] if len(h1_close) > 5 else h1_close.iloc[0]), 1e-10)) * 100.0)
    h4_mom_3bars_pct = float(((h4_close.iloc[-1] - (h4_close.iloc[-4] if len(h4_close) > 3 else h4_close.iloc[0])) /
                               max((h4_close.iloc[-4] if len(h4_close) > 3 else h4_close.iloc[0]), 1e-10)) * 100.0)

    feature_row = {
        "overnight_gap_pct": overnight_gap_pct,
        "premarket_range_pct": pre_rng_pct,
        "premarket_vol": pre_vol,
        "premarket_vol_vs_prev5d": float(pre_vol / max(ddf_et[ddf_et.index.date < today]["volume"].tail(5).mean() or 1.0, 1.0)),
        "premarket_sweep_prev_high": pre_sweep_prev_high,
        "premarket_sweep_prev_low": pre_sweep_prev_low,
        "premarket_close_vs_prev_close_pct": pre_close_vs_prev_close_pct,
        "premarket_return_pct": pre_return_pct,

        "prev_close": prev_close,
        "prev_day_range_pct": float((prev_high - prev_low) / max(prev_close, 1e-10) * 100.0),
        "prev_day_body_pct": prev_day_body_pct,
        "prev_day_bull": prev_day_bull,
        "prev_day_swept_prior_high": prev_day_swept_prior_high,
        "prev_day_swept_prior_low": prev_day_swept_prior_low,
        "open_pos_in_prev_range": open_pos_in_prev_range,
        "open_to_prev_high_pct_rng": open_to_prev_high_pct_rng,
        "open_to_prev_low_pct_rng": open_to_prev_low_pct_rng,
        "daily_atr14_pct": atr14_pct,

        "h1_close_vs_sma20_pct": h1_close_vs_sma20_pct,
        "h4_close_vs_sma20_pct": h4_close_vs_sma20_pct,
        "h1_mom_5bars_pct": h1_mom_5bars_pct,
        "h4_mom_3bars_pct": h4_mom_3bars_pct,
    }

    # Align to the saved feature order
    X = pd.DataFrame([[feature_row.get(f, 0.0) for f in feature_names]], columns=feature_names)
    meta = {
        "et_date": str(today),
        "open_source": open_source,
        "premarket_last_ts": (pre.index[-1].isoformat() if not pre.empty else None),
    }
    return X, None, meta

# ---------------------- Model I/O ---------------------- #
def _load_model_and_encoder(symbol: str):
    models_dir = PROJECT_ROOT / "models"
    with open(models_dir / f"{symbol}_daily_bias.pkl", "rb") as f:
        model = pickle.load(f)
    with open(models_dir / f"{symbol}_label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    # Feature list (symbol-specific; fallback to project-wide default if needed)
    feat_path = models_dir / f"{symbol}_feature_names.json"
    if feat_path.exists():
        feature_names = json.loads(feat_path.read_text(encoding="utf-8"))
    else:
        # Fallback (shouldn't happen if trained by the companion script)
        feature_names = [
            "overnight_gap_pct","premarket_range_pct","premarket_vol","premarket_vol_vs_prev5d",
            "premarket_sweep_prev_high","premarket_sweep_prev_low","premarket_close_vs_prev_close_pct",
            "premarket_return_pct","prev_close","prev_day_range_pct","prev_day_body_pct","prev_day_bull",
            "prev_day_swept_prior_high","prev_day_swept_prior_low","open_pos_in_prev_range",
            "open_to_prev_high_pct_rng","open_to_prev_low_pct_rng","daily_atr14_pct",
            "h1_close_vs_sma20_pct","h4_close_vs_sma20_pct","h1_mom_5bars_pct","h4_mom_3bars_pct",
        ]
    return model, le, feature_names

def _predict(model, le, X_row: pd.DataFrame):
    proba = model.predict_proba(X_row)[0]
    classes = list(le.classes_)
    idx = int(np.argmax(proba))
    return {
        "bias": str(classes[idx]),
        "confidence": float(proba[idx]),
        "probabilities": {c: float(proba[i]) for i, c in enumerate(classes)}
    }

# ---------------------- Output ---------------------- #
def _write_json(results_by_symbol: dict, feed: DataFeed, et_date: str):
    out_dir = PROJECT_ROOT / "data" / "daily_news" / et_date / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_bias.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_for_date_et": et_date,
        "data_feed": feed.value,
        "feature_set": "open_snapshot_premarket+HTF",
        "symbols": results_by_symbol,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Wrote: {out_path}")

# ---------------------- Main ---------------------- #
def main():
    _load_env()
    _trading, data_client, feed = _init_alpaca()
    et_date = str(pd.Timestamp.now(tz=ET_TZ).date())

    results = {}
    for symbol in WATCHLIST:
        print(f"--- {symbol} ---")
        try:
            model, le, feature_names = _load_model_and_encoder(symbol)
        except Exception as e:
            results[symbol] = {"error": f"model_load_failed: {e}"}
            continue

        ddf, mdf, h1df, h4df = _fetch_today_frames(symbol, data_client, feed)
        if any(x is None or x.empty for x in [ddf, mdf, h1df, h4df]):
            results[symbol] = {"error": "insufficient_market_data"}
            continue

        X_row, err, meta = _compute_open_features_today(symbol, ddf, mdf, h1df, h4df, feature_names)
        if err is not None:
            results[symbol] = {"error": err}
            continue

        pred = _predict(model, le, X_row)
        results[symbol] = {
            **pred,
            "asof_open_meta": meta,
            "features_snapshot": {k: float(X_row.iloc[0][k]) for k in feature_names}
        }

    _write_json(results, feed, et_date)

if __name__ == "__main__":
    main()
