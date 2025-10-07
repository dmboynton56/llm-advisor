#!/usr/bin/env python3
"""
Open-Snapshot Daily Bias Training (premarket + HTF context strictly before 9:30 ET)

What it does
------------
Per symbol:
- Downloads Daily + Minute + 1H + 4H bars.
- For each trading day, builds features that are available up to the 9:30 ET open.
- Labels the SAME DAY by end-of-day return from the 09:30 ET open to that day's close.
- Trains a classifier (RandomForest by default).
- Saves:
    models/<SYMBOL>_daily_bias.pkl
    models/<SYMBOL>_label_encoder.pkl
    models/<SYMBOL>_feature_names.json
    models/feature_importances/<SYMBOL>_importances.csv
    models/feature_importances/<SYMBOL>_importances.png
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from alpaca.data.enums import DataFeed, Adjustment

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---- Alpaca (alpaca-py) ----
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# Project root (this file sits under training/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Expect config/settings.py with WATCHLIST
from config.settings import WATCHLIST  # noqa: E402

warnings.filterwarnings("ignore")
ET_TZ = "US/Eastern"

# ---------------------- Version-safe time window helper ---------------------- #
def _bt(df: pd.DataFrame, start: str, end: str, inclusive: str = "both") -> pd.DataFrame:
    """
    Pandas between_time compatibility across versions.
    inclusive: 'both' | 'left' | 'right' | 'neither'
    """
    try:
        return df.between_time(start, end, inclusive=inclusive)
    except TypeError:
        out = df.between_time(start, end)  # old pandas defaults ~inclusive of both
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

# ---------------------- Data fetch ---------------------- #
def _fetch_all_frames(symbol: str, data_client, feed: DataFeed,
                      lookback_days: int = 1200):
    """
    Get Daily, Minute, 1H, 4H bars for lookback_days.
    """
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=lookback_days)

    # Daily
    req_d = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_utc,
        end=end_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    ddf = _as_single_symbol_df(data_client.get_stock_bars(req_d).df, symbol).tz_convert(ET_TZ)

    # Minute
    req_m = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_utc,
        end=end_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    mdf = _as_single_symbol_df(data_client.get_stock_bars(req_m).df, symbol).tz_convert(ET_TZ)

    # 1H
    req_h1 = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,
        start=start_utc,
        end=end_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    h1df = _as_single_symbol_df(data_client.get_stock_bars(req_h1).df, symbol).tz_convert(ET_TZ)

    # 4H
    req_h4 = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(4, TimeFrameUnit.Hour),
        start=start_utc,
        end=end_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    h4df = _as_single_symbol_df(data_client.get_stock_bars(req_h4).df, symbol).tz_convert(ET_TZ)

    return ddf, mdf, h1df, h4df


# ---------------------- Feature engineering (strictly <= 9:30 ET) ---------------------- #
FEATURES = [
    # Overnight & premarket
    "overnight_gap_pct",
    "premarket_range_pct",
    "premarket_vol",
    "premarket_vol_vs_prev5d",
    "premarket_sweep_prev_high",
    "premarket_sweep_prev_low",
    "premarket_close_vs_prev_close_pct",
    "premarket_return_pct",

    # Previous-day context
    "prev_close",
    "prev_day_range_pct",
    "prev_day_body_pct",
    "prev_day_bull",
    "prev_day_swept_prior_high",
    "prev_day_swept_prior_low",
    "open_pos_in_prev_range",
    "open_to_prev_high_pct_rng",
    "open_to_prev_low_pct_rng",
    "daily_atr14_pct",

    # HTF context (bars strictly before 9:30 ET)
    "h1_close_vs_sma20_pct",
    "h4_close_vs_sma20_pct",
    "h1_mom_5bars_pct",
    "h4_mom_3bars_pct",
]

def _first_rth_minute(mdf_et: pd.DataFrame, day_date):
    rth = _bt(mdf_et[mdf_et.index.date == day_date], "09:30", "16:00", inclusive="left")
    if rth.empty: return None
    return rth.iloc[0]

def _premarket_slice(mdf_et: pd.DataFrame, day_date):
    return _bt(mdf_et[mdf_et.index.date == day_date], "04:00", "09:29", inclusive="both")

def _last_before(ts_series: pd.Series, cutoff_ts: pd.Timestamp):
    return ts_series[ts_series.index < cutoff_ts].iloc[-1] if not ts_series[ts_series.index < cutoff_ts].empty else None

def _sma(series: pd.Series, n: int):
    return series.rolling(n).mean()

def _daily_atr14_pct(ddf_et: pd.DataFrame, up_to_date) -> float:
    # Use prior days only (<= prev day)
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

def _build_row_for_day(day_ts, ddf_et, mdf_et, h1df, h4df):
    day_date = day_ts.date()
    prev_day = (pd.Timestamp(day_ts).tz_convert(ET_TZ) - pd.Timedelta(days=1)).date()
    prev2_day = (pd.Timestamp(day_ts).tz_convert(ET_TZ) - pd.Timedelta(days=2)).date()

    prev = ddf_et[ddf_et.index.date == prev_day]
    prev2 = ddf_et[ddf_et.index.date == prev2_day]
    if prev.empty or prev2.empty:
        return None  # need at least prior two days for sweep flags

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

    # Premarket
    pre = _premarket_slice(mdf_et, day_date)
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

    # Open (09:30) and label
    m0930 = _first_rth_minute(mdf_et, day_date)
    if m0930 is None or np.isnan(m0930["open"]):
        return None  # cannot label day without a proper 09:30 open
    open_0930 = float(m0930["open"])

    # HTF context strictly before 09:30
    open_ts = pd.Timestamp.combine(pd.Timestamp(day_date), pd.Timestamp("09:30").time()).tz_localize(ET_TZ)
    h1 = h1df[h1df.index < open_ts]
    h4 = h4df[h4df.index < open_ts]
    if h1.empty or h4.empty:
        return None

    h1_close = h1["close"]
    h4_close = h4["close"]
    h1_sma20 = _sma(h1_close, 20)
    h4_sma20 = _sma(h4_close, 20)

    h1_close_vs_sma20_pct = float(((h1_close.iloc[-1] - h1_sma20.iloc[-1]) / max(h1_sma20.iloc[-1], 1e-10)) * 100.0) if not np.isnan(h1_sma20.iloc[-1]) else np.nan
    h4_close_vs_sma20_pct = float(((h4_close.iloc[-1] - h4_sma20.iloc[-1]) / max(h4_sma20.iloc[-1], 1e-10)) * 100.0) if not np.isnan(h4_sma20.iloc[-1]) else np.nan

    h1_mom_5bars_pct = float(((h1_close.iloc[-1] - (h1_close.iloc[-6] if len(h1_close) > 5 else h1_close.iloc[0])) /
                               max((h1_close.iloc[-6] if len(h1_close) > 5 else h1_close.iloc[0]), 1e-10)) * 100.0)
    h4_mom_3bars_pct = float(((h4_close.iloc[-1] - (h4_close.iloc[-4] if len(h4_close) > 3 else h4_close.iloc[0])) /
                               max((h4_close.iloc[-4] if len(h4_close) > 3 else h4_close.iloc[0]), 1e-10)) * 100.0)

    # Open position relative to prev day's range
    open_pos_in_prev_range = float((open_0930 - prev_low) / max(prev_range, 1e-10))
    open_to_prev_high_pct_rng = float((prev_high - open_0930) / max(prev_range, 1e-10) * 100.0)
    open_to_prev_low_pct_rng  = float((open_0930 - prev_low) / max(prev_range, 1e-10) * 100.0)

    # ATR14% based on prior days only
    atr14_pct = _daily_atr14_pct(ddf_et, day_date)

    # Overnight gap
    overnight_gap_pct = float((open_0930 - prev_close) / max(prev_close, 1e-10) * 100.0)

    # LABEL (same-day EOD movement from 09:30 open)
    today_row = ddf_et[ddf_et.index.date == day_date]
    if today_row.empty:
        return None
    close_px = float(today_row.iloc[-1]["close"])
    ret_pct = float((close_px - open_0930) / max(open_0930, 1e-10) * 100.0)
    if ret_pct > 0.25:
        label = "bullish"
    elif ret_pct < -0.25:
        label = "bearish"
    else:
        label = "choppy"

    row = {
        "date_et": pd.Timestamp(day_ts).tz_convert(ET_TZ).normalize(),
        "overnight_gap_pct": overnight_gap_pct,
        "premarket_range_pct": pre_rng_pct,
        "premarket_vol": pre_vol,
        "premarket_vol_vs_prev5d": float(pre_vol / max(ddf_et[ddf_et.index.date < day_date]["volume"].tail(5).mean() or 1.0, 1.0)),
        "premarket_sweep_prev_high": pre_sweep_prev_high,
        "premarket_sweep_prev_low": pre_sweep_prev_low,
        "premarket_close_vs_prev_close_pct": pre_close_vs_prev_close_pct,
        "premarket_return_pct": pre_return_pct,

        "prev_close": prev_close,
        "prev_day_range_pct": prev_day_range_pct,
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

        "label": label,
    }
    # clean missing HTF SMA if insufficient history
    if any(np.isnan([row["h1_close_vs_sma20_pct"], row["h4_close_vs_sma20_pct"], atr14_pct])):
        return None
    return row

def _build_training_frame(symbol: str, ddf_et: pd.DataFrame, mdf_et: pd.DataFrame, h1df: pd.DataFrame, h4df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ts, _ in ddf_et.iterrows():
        r = _build_row_for_day(ts, ddf_et, mdf_et, h1df, h4df)
        if r is not None:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date_et").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# ---------------------- Train + save ---------------------- #
def _train_and_save(symbol: str, df: pd.DataFrame):
    X = df[FEATURES]
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{symbol}] Test Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / f"{symbol}_daily_bias.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(models_dir / f"{symbol}_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(models_dir / f"{symbol}_feature_names.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    # Feature importances
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        fi_dir = models_dir / "feature_importances"
        fi_dir.mkdir(parents=True, exist_ok=True)
        imp_map = {feat: float(val) for feat, val in zip(FEATURES, importances)}
        (fi_dir / f"{symbol}_importances.csv").write_text(
            "feature,importance\n" + "\n".join([f"{k},{v}" for k, v in imp_map.items()]),
            encoding="utf-8"
        )
        # Plot
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(FEATURES)), importances[order])
        plt.xticks(range(len(FEATURES)), [FEATURES[i] for i in order], rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(fi_dir / f"{symbol}_importances.png")
        plt.close()

def run_training_pipeline():
    print("=" * 68)
    print("Open-Snapshot Daily Bias Training (premarket + HTF context)")
    print("Symbols:", ", ".join(WATCHLIST))
    print("=" * 68)

    _load_env()
    _trading, data_client, feed = _init_alpaca()

    for symbol in WATCHLIST:
        print(f"\n--- {symbol} ---")
        ddf, mdf, h1df, h4df = _fetch_all_frames(symbol, data_client, feed)
        if any(x is None or x.empty for x in [ddf, mdf, h1df, h4df]):
            print("  Skipping: missing data.")
            continue
        frame = _build_training_frame(symbol, ddf, mdf, h1df, h4df)
        if frame.empty:
            print("  Skipping: no rows after feature engineering.")
            continue
        _train_and_save(symbol, frame)

    print("\n✅ Training complete — models saved under /models")

if __name__ == "__main__":
    run_training_pipeline()
