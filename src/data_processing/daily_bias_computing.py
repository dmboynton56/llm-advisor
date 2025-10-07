#!/usr/bin/env python3
"""
Compute SAME-DAY daily bias at/just after the 9:30 ET open, per symbol,
then log independent price snapshots and cross-market context for LLM synthesis.

Outputs (ET date under data/daily_news/YYYY-MM-DD/raw/):
- daily_bias.json            (existing ML priors)
- snapshots.json             (per-symbol microstructure snapshot)
- cross_market.json          (VIX/DXY/UST proxies, breadth, SMT)
Also inlines snapshots + cross_market into daily_bias.json for convenience.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# ---------------------- Project paths / settings ---------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import WATCHLIST  # noqa: E402

ET_TZ = "US/Eastern"
SESSION_OPEN = "09:30"
SESSION_CLOSE = "16:00"
OR_MINUTES = 30
ATR_PERIOD = 14
EMA_PERIOD = 20

# ETF proxies for cross-market context (work on equity feeds)
VIX_PROXY = "VIXY"   # VIX futures proxy
DXY_PROXY = "UUP"    # DXY proxy
UST10Y_PROXY = "IEF" # 7-10y UST proxy
PROXY_SYMBOLS = [VIX_PROXY, DXY_PROXY, UST10Y_PROXY]

# ---------------------- Small helpers ---------------------- #
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_py(x):
    """Convert numpy/pandas scalars to plain Python for json.dump(default=_to_py)."""
    # All NumPy scalar types (float32/64, int*, bool_, etc.)
    if isinstance(x, np.generic):
        return x.item()
    # Pandas timestamps
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x

def _bt(df: pd.DataFrame, start: str, end: str, inclusive: str = "both") -> pd.DataFrame:
    """Time window slice that works across pandas versions."""
    try:
        return df.between_time(start, end, inclusive=inclusive)
    except TypeError:
        out = df.between_time(start, end)
        st = pd.Timestamp(start).time()
        en = pd.Timestamp(end).time()
        if inclusive == "left":
            return out[out.index.time != en]
        if inclusive == "right":
            return out[out.index.time != st]
        if inclusive == "neither":
            return out[(out.index.time != st) & (out.index.time != en)]
        return out

def _rth_slice(df: pd.DataFrame, day_date) -> pd.DataFrame:
    return _bt(df[df.index.date == day_date], SESSION_OPEN, SESSION_CLOSE, inclusive="left")

def _premarket_slice(df: pd.DataFrame, day_date) -> pd.DataFrame:
    return _bt(df[df.index.date == day_date], "04:00", "09:29", inclusive="both")

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

# ---------------------- Env + Alpaca client ---------------------- #
def _load_env():
    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path if dotenv_path.exists() else None)

def _init_alpaca():
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca API credentials in .env")
    feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX
    data = StockHistoricalDataClient(api_key, api_secret)
    return data, feed

# ---------------------- Data fetch ---------------------- #
def _fetch_today_frames(symbol: str, data_client, feed: DataFeed):
    """Fetch Day, 1m, 5m, 1h, 4h up to 'now' (UTC), returned in ET tz."""
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=120)

    def _req(tf):
        return StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start_utc,
            end=now_utc,
            feed=feed,
            adjustment=Adjustment.RAW,
        )

    ddf  = _as_single_symbol_df(data_client.get_stock_bars(_req(TimeFrame.Day)).df, symbol).tz_convert(ET_TZ)
    m1df = _as_single_symbol_df(data_client.get_stock_bars(_req(TimeFrame.Minute)).df, symbol).tz_convert(ET_TZ)
    m5df = _as_single_symbol_df(data_client.get_stock_bars(_req(TimeFrame(5, TimeFrameUnit.Minute))).df, symbol).tz_convert(ET_TZ)
    h1df = _as_single_symbol_df(data_client.get_stock_bars(_req(TimeFrame.Hour)).df, symbol).tz_convert(ET_TZ)
    h4df = _as_single_symbol_df(data_client.get_stock_bars(_req(TimeFrame(4, TimeFrameUnit.Hour))).df, symbol).tz_convert(ET_TZ)
    return ddf, m1df, m5df, h1df, h4df

# ---------------------- Training-aligned open snapshot (ML priors) ---------------------- #
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

def _first_rth_minute(mdf_et: pd.DataFrame, day_date):
    rth = _rth_slice(mdf_et, day_date)
    if rth.empty:
        return None
    return rth.iloc[0]

def _compute_open_features_today(symbol: str, ddf_et, m1df_et, h1df, h4df, feature_names):
    today = pd.Timestamp.now(tz=ET_TZ).date()
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
    prev_day_body_pct = (abs(prev.iloc[-1]["close"] - prev_open) / prev_range) * 100.0
    prev_day_bull = int(prev.iloc[-1]["close"] > prev_open)

    prev2_high = float(prev2.iloc[-1]["high"])
    prev2_low  = float(prev2.iloc[-1]["low"])
    prev_day_swept_prior_high = int(prev_high > prev2_high)
    prev_day_swept_prior_low  = int(prev_low < prev2_low)

    pre = _premarket_slice(m1df_et, today)
    if pre.empty:
        pre_hi = np.nan; pre_lo = np.nan; pre_vol = 0.0
        pre_first = None; pre_last = None
    else:
        pre_hi = float(pre["high"].max())
        pre_lo = float(pre["low"].min())
        pre_vol = float(pre["volume"].sum())
        pre_first = pre.iloc[0]; pre_last = pre.iloc[-1]

    pre_rng_pct = ((pre_hi - pre_lo) / max(prev_close, 1e-10) * 100.0
                   if not (np.isnan(pre_hi) or np.isnan(pre_lo)) else 0.0)
    pre_close_vs_prev_close_pct = (((float(pre_last["close"]) - prev_close) / max(prev_close, 1e-10) * 100.0)
                                   if pre_last is not None else 0.0)
    pre_return_pct = (((float(pre_last["close"]) - float(pre_first["open"])) / max(float(pre_first["open"]), 1e-10) * 100.0)
                      if (pre_first is not None and pre_last is not None) else 0.0)
    pre_sweep_prev_high = int((not np.isnan(pre_hi)) and (pre_hi > prev_high))
    pre_sweep_prev_low  = int((not np.isnan(pre_lo)) and (pre_lo < prev_low))

    # Prefer 09:30 OPEN, else last premarket CLOSE if 09:30 not available yet
    open_source = "09:30_minute_open"
    m0930 = _first_rth_minute(m1df_et, today)
    if m0930 is not None and not np.isnan(m0930["open"]):
        open_0930 = float(m0930["open"])
    else:
        open_source = "09:29_close_proxy"
        if pre_last is None or np.isnan(pre_last["close"]):
            return None, {"error": "no_open_or_premarket"}, None
        open_0930 = float(pre_last["close"])

    open_pos_in_prev_range = float((open_0930 - prev_low) / max(prev_high - prev_low, 1e-10))
    open_to_prev_high_pct_rng = float((prev_high - open_0930) / max(prev_high - prev_low, 1e-10) * 100.0)
    open_to_prev_low_pct_rng  = float((open_0930 - prev_low) / max(prev_high - prev_low, 1e-10) * 100.0)
    overnight_gap_pct = float((open_0930 - prev_close) / max(prev_close, 1e-10) * 100.0)
    atr14_pct = _daily_atr14_pct(ddf_et, today)

    open_ts = pd.Timestamp.combine(pd.Timestamp(today), pd.Timestamp(SESSION_OPEN).time()).tz_localize(ET_TZ)
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

    X = pd.DataFrame([[feature_row.get(f, 0.0) for f in feature_names]], columns=feature_names)
    meta = {
        "et_date": str(today),
        "open_source": "09:30_minute_open" if _first_rth_minute(m1df_et, today) is not None else "09:29_close_proxy",
        "premarket_last_ts": (pre.index[-1].isoformat() if not pre.empty else None),
    }
    return X, None, meta

# ---------------------- ML model I/O ---------------------- #
def _load_model_and_encoder(symbol: str):
    models_dir = PROJECT_ROOT / "models"
    with open(models_dir / f"{symbol}_daily_bias.pkl", "rb") as f:
        model = pickle.load(f)
    with open(models_dir / f"{symbol}_label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    feat_path = models_dir / f"{symbol}_feature_names.json"
    if feat_path.exists():
        feature_names = json.loads(feat_path.read_text(encoding="utf-8"))
    else:
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

# ---------------------- Independent price snapshot metrics ---------------------- #
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _vwap(df_1m: pd.DataFrame) -> pd.Series:
    typical = (df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3.0
    cum_vol = df_1m["volume"].cumsum().replace(0, np.nan)
    return (typical * df_1m["volume"]).cumsum() / cum_vol

def _atr_last(daily: pd.DataFrame, period: int = ATR_PERIOD) -> float | None:
    if daily is None or len(daily) < period + 1:
        return None
    h, l, c = daily["high"], daily["low"], daily["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    val = atr.iloc[-1]
    return _to_py(val) if pd.notna(val) else None

def _opening_range_pct_of_atr(df_1m: pd.DataFrame, atr_val: float | None) -> float | None:
    if df_1m.empty or not atr_val:
        return None
    first = df_1m.iloc[:OR_MINUTES]
    return float((first["high"].max() - first["low"].min()) / atr_val)

def _trend_5m_vs_ema20(df_5m: pd.DataFrame) -> float | None:
    if len(df_5m) < EMA_PERIOD + 1:
        return None
    ema20 = _ema(df_5m["close"], EMA_PERIOD)
    last = df_5m["close"].iloc[-1]
    last_ema = ema20.iloc[-1]
    if pd.isna(last_ema):
        return None
    return float((last - last_ema) / last_ema)

def _count_liquidity_sweeps(df_1m: pd.DataFrame, prior_high: float, prior_low: float) -> dict:
    if df_1m.empty:
        return {"high": 0, "low": 0}
    highs = (df_1m["high"] > prior_high).sum()
    lows  = (df_1m["low"]  < prior_low).sum()
    return {"high": int(highs), "low": int(lows)}

def _detect_fvgs_5m(df_5m: pd.DataFrame) -> dict:
    created, filled = 0, 0
    highs, lows = df_5m["high"].values, df_5m["low"].values
    n = len(df_5m); gaps = []
    for t in range(2, n):
        if lows[t] > highs[t-2]:      # bullish FVG
            created += 1; gaps.append(("bull", highs[t-2], lows[t], t))
        if highs[t] < lows[t-2]:      # bearish FVG
            created += 1; gaps.append(("bear", highs[t], lows[t-2], t))
    for (typ, top, bot, t0) in gaps:
        for k in range(t0 + 1, n):
            if not (df_5m["low"].iloc[k] > top or df_5m["high"].iloc[k] < bot):
                filled += 1; break
    return {"created": int(created), "filled": int(filled)}

def _gap_fill_stats(today_1m: pd.DataFrame, prior_close: float) -> dict:
    if today_1m.empty or prior_close is None:
        return {"dir": None, "magnitude_pct": None, "filled_pct": None, "filled_in_minutes": None}
    open_px = float(today_1m.iloc[0]["open"])
    dirn = "up" if open_px > prior_close else ("down" if open_px < prior_close else "flat")
    gap_mag = (open_px - prior_close) / max(prior_close, 1e-10) * 100.0
    lows = today_1m["low"].cummin(); highs = today_1m["high"].cummax()
    filled_pct, mins_to_fill = None, None
    if dirn == "up":
        gap_abs = abs(open_px - prior_close)
        closed = max(0.0, prior_close - lows.iloc[-1])
        filled_pct = float(min(1.0, (closed / max(gap_abs, 1e-10)))) if gap_abs else 0.0
        if (today_1m["low"] <= prior_close).any():
            mins_to_fill = int(np.argmax((today_1m["low"] <= prior_close).values))
    elif dirn == "down":
        gap_abs = abs(open_px - prior_close)
        closed = max(0.0, highs.iloc[-1] - prior_close)
        filled_pct = float(min(1.0, (closed / max(gap_abs, 1e-10)))) if gap_abs else 0.0
        if (today_1m["high"] >= prior_close).any():
            mins_to_fill = int(np.argmax((today_1m["high"] >= prior_close).values))
    return {"dir": dirn, "magnitude_pct": float(gap_mag), "filled_pct": filled_pct, "filled_in_minutes": mins_to_fill}

def _volume_vs_5d_avg(today_1m: pd.DataFrame, daily: pd.DataFrame) -> float | None:
    if today_1m.empty or daily.empty:
        return None
    cum_vol = float(today_1m["volume"].sum())
    avg_daily = float(daily["volume"].tail(5).mean())
    elapsed = len(today_1m)  # minutes since open
    norm = max(elapsed / 390.0, 1e-6)
    if avg_daily <= 0:
        return None
    return float(cum_vol / (avg_daily * norm))

def _corr_to_benchmark_5m(df_5m: pd.DataFrame, bench_5m: pd.DataFrame) -> float | None:
    if len(df_5m) < 5 or len(bench_5m) < 5:
        return None
    j = df_5m[["close"]].join(bench_5m[["close"]], lsuffix="_sym", rsuffix="_bench", how="inner")
    if len(j) < 5:
        return None
    r_sym = j["close_sym"].pct_change().dropna()
    r_ben = j["close_bench"].pct_change().dropna()
    if len(r_sym) < 3 or len(r_ben) < 3:
        return None
    return float(np.corrcoef(r_sym.values, r_ben.values)[0, 1])

def _smt_divergence_spy_qqq(spy_5m: pd.DataFrame | None,
                            qqq_5m: pd.DataFrame | None,
                            lookback_bars: int = 24) -> bool:
    if spy_5m is None or qqq_5m is None:
        return False
    if spy_5m.empty or qqq_5m.empty:
        return False
    s, q = spy_5m.tail(lookback_bars), qqq_5m.tail(lookback_bars)
    s_hh = s["high"].cummax().iloc[-1] > s["high"].iloc[0]
    q_hh = q["high"].cummax().iloc[-1] > q["high"].iloc[0]
    s_ll = s["low"].cummin().iloc[-1] < s["low"].iloc[0]
    q_ll = q["low"].cummin().iloc[-1] < q["low"].iloc[0]
    return bool((s_hh != q_hh) or (s_ll != q_ll))

def _build_symbol_snapshot(symbol: str,
                           today_1m: pd.DataFrame,
                           today_5m: pd.DataFrame,
                           daily: pd.DataFrame,
                           bench_5m: pd.DataFrame,
                           prior_high: float, prior_low: float, prior_close: float) -> dict:
    atr_val = _atr_last(daily, ATR_PERIOD)
    vwap_series = _vwap(today_1m) if not today_1m.empty else pd.Series(dtype=float)
    vwap_dist = None
    if not today_1m.empty and len(vwap_series):
        vwap_dist = float((today_1m["close"].iloc[-1] - vwap_series.iloc[-1]) / max(vwap_series.iloc[-1], 1e-10) * 100.0)
    snap = {
        "symbol": symbol,
        "opening_range_pct_of_ATR": _opening_range_pct_of_atr(today_1m, atr_val),
        "vwap_distance_pct": vwap_dist,
        "trend_5m_vs_20ema": _trend_5m_vs_ema20(today_5m),
        "sweeps_since_open": _count_liquidity_sweeps(today_1m, prior_high, prior_low),
        "fvg_5m": _detect_fvgs_5m(today_5m),
        "volume_vs_5d_avg": _volume_vs_5d_avg(today_1m, daily),
        "gap": _gap_fill_stats(today_1m, prior_close),
        "corr_to_SPY_5m": _corr_to_benchmark_5m(today_5m, bench_5m) if bench_5m is not None else None,
    }
    return {k: (_to_py(v) if isinstance(v, (np.generic, pd.Timestamp)) else v) for k, v in snap.items()}

def _build_cross_market(spy_5m, qqq_5m, vix_5m, dxy_5m, ust_5m,
                        daily_map: dict[str, pd.DataFrame], wl: list[str]) -> dict:
    def _last_and_delta(df: pd.DataFrame):
        if df is None or df.empty:
            return None, None
        return float(df["close"].iloc[-1]), float(df["close"].iloc[-1] - df["close"].iloc[0])

    VIX_level, VIX_delta = _last_and_delta(vix_5m)
    DXY_level, DXY_delta = _last_and_delta(dxy_5m)
    UST_level, UST_delta = _last_and_delta(ust_5m)

    above, total, adv, dec = 0, 0, 0, 0
    for s in wl:
        d = daily_map.get(s)
        if d is None or len(d) < 21:
            continue
        sma20 = d["close"].rolling(20).mean().iloc[-1]
        last = d["close"].iloc[-1]
        prev_close = d["close"].iloc[-2]
        total += 1
        if last > sma20: above += 1
        if last > prev_close: adv += 1
        elif last < prev_close: dec += 1
    breadth_pct = int(round(100.0 * above / total)) if total else None
    ad_ratio = float(adv / max(dec, 1)) if total else None

    return {
        "VIX": {"symbol": VIX_PROXY, "level_proxy": VIX_level, "delta": VIX_delta},
        "DXY": {"symbol": DXY_PROXY, "level_proxy": DXY_level, "delta": DXY_delta},
        "UST10Y_proxy": {"symbol": UST10Y_PROXY, "level_proxy": UST_level, "delta": UST_delta},
        "WL_breadth_pct_above_SMA20": breadth_pct,
        "adv_decl_ratio": ad_ratio,
        "ES_NQ_SMT_divergence": bool(_smt_divergence_spy_qqq(spy_5m, qqq_5m)),
    }

# ---------------------- JSON output ---------------------- #
def _paths_for_date(et_date: str):
    raw_dir = PROJECT_ROOT / "data" / "daily_news" / et_date / "raw"
    _ensure_dir(raw_dir)
    return raw_dir, raw_dir / "daily_bias.json", raw_dir / "snapshots.json", raw_dir / "cross_market.json"

def _write_daily_bias(results_by_symbol: dict, feed: DataFeed, et_date: str) -> Path:
    raw_dir, bias_path, _, _ = _paths_for_date(et_date)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_for_date_et": et_date,
        "data_feed": feed.value,
        "feature_set": "open_snapshot_premarket+HTF",
        "symbols": results_by_symbol,
    }
    with open(bias_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_to_py)
    print(f"Wrote: {bias_path}")
    return bias_path

def _write_snapshots_and_cross(et_date: str, snapshots: list[dict], cross_market: dict, inline_into_bias: bool = True):
    raw_dir, bias_path, snaps_path, cross_path = _paths_for_date(et_date)
    with open(snaps_path, "w", encoding="utf-8") as f:
        json.dump({"date": et_date, "snapshots": snapshots}, f, indent=2, default=_to_py)
    with open(cross_path, "w", encoding="utf-8") as f:
        json.dump({"date": et_date, "cross_market": cross_market}, f, indent=2, default=_to_py)
    print(f"Wrote: {snaps_path}")
    print(f"Wrote: {cross_path}")

    if inline_into_bias and bias_path.exists():
        try:
            with open(bias_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = {}
        doc["price_snapshots"] = snapshots
        doc["cross_market"] = cross_market
        with open(bias_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, default=_to_py)
        print(" Inlined snapshots & cross_market into daily_bias.json")

# ---------------------- Main ---------------------- #
def main():
    _load_env()
    data_client, feed = _init_alpaca()
    et_date = str(pd.Timestamp.now(tz=ET_TZ).date())

    # 1) Compute ML priors (as before, but trimmed & tidy)
    results = {}
    per_symbol_frames = {}  # cache frames to reuse in snapshot step
    for symbol in WATCHLIST:
        print(f"--- {symbol} ---")
        try:
            model, le, feature_names = _load_model_and_encoder(symbol)
        except Exception as e:
            results[symbol] = {"error": f"model_load_failed: {e}"}
            continue

        ddf, m1df, m5df, h1df, h4df = _fetch_today_frames(symbol, data_client, feed)
        per_symbol_frames[symbol] = (ddf, m1df, m5df)  # keep for snapshots
        if any(x is None or x.empty for x in [ddf, m1df, h1df, h4df]):
            results[symbol] = {"error": "insufficient_market_data"}
            continue

        X_row, err, meta = _compute_open_features_today(symbol, ddf, m1df, h1df, h4df, feature_names)
        if err is not None:
            results[symbol] = {"error": err}
            continue

        pred = _predict(model, le, X_row)
        results[symbol] = {
            **pred,
            "asof_open_meta": meta,
            "features_snapshot": {k: float(X_row.iloc[0][k]) for k in feature_names}
        }

    bias_path = _write_daily_bias(results, feed, et_date)

    # 2) Build price snapshots (independent evidence) + cross-market block
    #    Ensure we have benchmark & proxies 5m frames available.
    for proxy in PROXY_SYMBOLS + ["SPY", "QQQ"]:
        if proxy not in per_symbol_frames:
            ddf, m1df, m5df, _, _ = _fetch_today_frames(proxy, data_client, feed)
            per_symbol_frames[proxy] = (ddf, m1df, m5df)

    today = pd.Timestamp.now(tz=ET_TZ).date()
    spy_5m = _rth_slice(per_symbol_frames["SPY"][2], today)
    qqq_5m = _rth_slice(per_symbol_frames["QQQ"][2], today)
    vix_5m = _rth_slice(per_symbol_frames.get(VIX_PROXY, (None, None, pd.DataFrame()))[2], today)
    dxy_5m = _rth_slice(per_symbol_frames.get(DXY_PROXY, (None, None, pd.DataFrame()))[2], today)
    ust_5m = _rth_slice(per_symbol_frames.get(UST10Y_PROXY, (None, None, pd.DataFrame()))[2], today)

    snapshots = []
    daily_map = {}

    for symbol in WATCHLIST:
        ddf, m1df, m5df = per_symbol_frames.get(symbol, (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
        if ddf is None or ddf.empty or len(ddf) < 2:
            snapshots.append({"symbol": symbol, "needs_more_data": True, "missing": ["daily_bars"]})
            continue
        daily_map[symbol] = ddf

        prior = ddf[ddf.index.date < today].tail(1)
        prior_high = float(prior["high"].iloc[-1])
        prior_low  = float(prior["low"].iloc[-1])
        prior_close = float(prior["close"].iloc[-1])

        today_1m = _rth_slice(m1df, today)
        today_5m = _rth_slice(m5df, today)
        if today_1m.empty or today_5m.empty:
            snapshots.append({"symbol": symbol, "needs_more_data": True, "missing": ["intraday_1m_or_5m"]})
            continue

        snap = _build_symbol_snapshot(
            symbol=symbol,
            today_1m=today_1m,
            today_5m=today_5m,
            daily=ddf,
            bench_5m=spy_5m,
            prior_high=prior_high,
            prior_low=prior_low,
            prior_close=prior_close,
        )
        snapshots.append(snap)

    cross_market = _build_cross_market(spy_5m, qqq_5m, vix_5m, dxy_5m, ust_5m, daily_map, WATCHLIST)

    _write_snapshots_and_cross(et_date, snapshots, cross_market, inline_into_bias=True)

if __name__ == "__main__":
    main()
