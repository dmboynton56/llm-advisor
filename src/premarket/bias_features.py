"""Shared premarket features: completed bars only, fixed at 09:00 ET."""
import numpy as np
import pandas as pd

ET_TZ = "America/New_York"
FEATURE_VERSION = "premarket_0900_completed_bars_v1"
FEATURE_NAMES = (
    "overnight_gap_pct", "premarket_range_pct", "premarket_vol_vs_prev5d",
    "premarket_sweep_prev_high", "premarket_sweep_prev_low",
    "premarket_close_vs_prev_close_pct", "premarket_return_pct",
    "prev_day_range_pct", "prev_day_body_pct", "prev_day_bull",
    "prev_day_swept_prior_high", "prev_day_swept_prior_low",
    "open_pos_in_prev_range", "open_to_prev_high_pct_rng", "open_to_prev_low_pct_rng",
    "daily_atr14_pct", "h1_close_vs_sma20_pct", "h4_close_vs_sma20_pct",
    "h1_mom_5bars_pct", "h4_mom_3bars_pct",
)


def _premarket_slice(frame, day_date):
    return frame[frame.index.date == day_date].between_time("04:00", "09:00", inclusive="left")


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

def compute_bias_features(symbol: str, ddf_et, m1df_et, h1df, h4df, feature_names,
                                 as_of=None, bar_minutes=1):
    """Same 09:00 ET information boundary for training and live inference.

    Bars are timestamped at their start. Only COMPLETED bars may contribute;
    in particular the historical 09:00 hourly bar contains future prices.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz=ET_TZ)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    today = as_of.tz_convert(ET_TZ).date()
    cutoff = pd.Timestamp(f"{today} 09:00", tz=ET_TZ)
    if as_of < cutoff:
        return None, {"error": "before_bias_feature_cutoff"}, None
    m1df_et = m1df_et[m1df_et.index + pd.Timedelta(minutes=bar_minutes) <= cutoff]
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
    if not pre.empty and pre.index[-1] + pd.Timedelta(minutes=bar_minutes) < cutoff - pd.Timedelta(minutes=20):
        return None, {"error": "stale_premarket_bars"}, None
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

    if pre_last is None or np.isnan(pre_last["close"]):
        return None, {"error": "no_open_or_premarket"}, None
    reference_price = float(pre_last["close"])

    open_pos_in_prev_range = float((reference_price - prev_low) / max(prev_high - prev_low, 1e-10))
    open_to_prev_high_pct_rng = float((prev_high - reference_price) / max(prev_high - prev_low, 1e-10) * 100.0)
    open_to_prev_low_pct_rng  = float((reference_price - prev_low) / max(prev_high - prev_low, 1e-10) * 100.0)
    overnight_gap_pct = float((reference_price - prev_close) / max(prev_close, 1e-10) * 100.0)
    atr14_pct = _daily_atr14_pct(ddf_et, today)

    h1 = h1df[h1df.index + pd.Timedelta(hours=1) <= cutoff]
    h4 = h4df[h4df.index + pd.Timedelta(hours=4) <= cutoff]
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

    X = pd.DataFrame([[feature_row[f] for f in feature_names]], columns=feature_names)
    if not np.isfinite(X.to_numpy()).all():
        return None, {"error": "nonfinite_bias_features"}, None
    meta = {
        "et_date": str(today),
        "open_source": "09:00_premarket_close_proxy",
        "feature_cutoff": cutoff.isoformat(),
        "reference_price": reference_price,
        "premarket_last_ts": (pre.index[-1].isoformat() if not pre.empty else None),
    }
    return X, None, meta

