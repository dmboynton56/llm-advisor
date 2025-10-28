#!/usr/bin/env python3
"""
Deterministic ICT feature detectors (v0.2).

Inputs
------
- Bars as list[dict] with keys: t (ISO str, UTC), o,h,l,c,v (floats)

Outputs
-------
- Feature records grouped by timeframe, suitable for serialization.

Detectors
---------
- ATR(14)
- Swings, Market Structure & BOS
- EQ (50%) & OTE zone (62%–79%) for most recent leg
- FVG with fill / invalidation tracking and IFVG promotion
- Order Blocks with retest/invalid status, Breaker blocks when invalidated OB flips
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


K_PIVOT = 2
ATR_PERIOD = 14
FVG_FILL_POLICY = "midline"  # "midline" or "full"
FVG_TOL = 1e-6
OB_DISPLACEMENT_ATR_MULT = 1.2
OB_BODY_RATIO = 0.5
OB_TOL = 1e-6


# ------------------------- Utilities -------------------------
def _to_df(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame(bars)
    colmap = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "time"}
    for k, v in colmap.items():
        if k in df.columns:
            df[v] = df[k]
    if "time" in df.columns:
        idx = pd.to_datetime(df["time"], utc=True)
    else:
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="T", tz="UTC")
    out = pd.DataFrame({
        "open": df.get("open", df.get("o", pd.Series(dtype=float))),
        "high": df.get("high", df.get("h", pd.Series(dtype=float))),
        "low": df.get("low", df.get("l", pd.Series(dtype=float))),
        "close": df.get("close", df.get("c", pd.Series(dtype=float))),
        "volume": df.get("volume", df.get("v", pd.Series(dtype=float))),
    }, index=idx)
    out.index.name = "time"
    return out.sort_index()


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def _pivot_high(df: pd.DataFrame, k: int) -> pd.Series:
    return (df["high"] == df["high"].rolling(k * 2 + 1, center=True).max())


def _pivot_low(df: pd.DataFrame, k: int) -> pd.Series:
    return (df["low"] == df["low"].rolling(k * 2 + 1, center=True).min())


def _last_swings(df: pd.DataFrame, k: int) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[pd.Timestamp], Optional[float]]:
    ph = _pivot_high(df, k)
    pl = _pivot_low(df, k)
    sh_time = df.index[ph].tolist()[-1] if ph.any() else None
    sl_time = df.index[pl].tolist()[-1] if pl.any() else None
    sh_val = float(df.loc[sh_time, "high"]) if sh_time is not None else None
    sl_val = float(df.loc[sl_time, "low"]) if sl_time is not None else None
    return sh_time, sh_val, sl_time, sl_val


def _detect_bos(df: pd.DataFrame, last_sh: Optional[float], last_sl: Optional[float]) -> Dict[str, Any]:
    bos = {"trend": None, "last_bos": None, "last_bos_time": None, "last_bos_level": None}
    if len(df) < 3:
        return bos
    last_close = df["close"].iloc[-1]
    if last_sh is not None and last_close > last_sh:
        bos.update({"trend": "bullish", "last_bos": "BOS_up", "last_bos_time": df.index[-1].isoformat(), "last_bos_level": float(last_sh)})
    elif last_sl is not None and last_close < last_sl:
        bos.update({"trend": "bearish", "last_bos": "BOS_down", "last_bos_time": df.index[-1].isoformat(), "last_bos_level": float(last_sl)})
    return bos


def _compute_eq_ote(leg_high: Optional[float], leg_low: Optional[float]) -> Dict[str, Any]:
    if leg_high is None or leg_low is None:
        return {"eq": None, "ote": None, "leg": None}
    hi, lo = float(leg_high), float(leg_low)
    if hi <= lo:
        return {"eq": None, "ote": None, "leg": None}
    eq = lo + 0.5 * (hi - lo)
    ote_low = lo + 0.62 * (hi - lo)
    ote_high = lo + 0.79 * (hi - lo)
    return {
        "eq": float(eq),
        "ote": {"low": float(ote_low), "high": float(ote_high)},
        "leg": {"high": hi, "low": lo},
    }


def _build_fvg_record(df: pd.DataFrame, idx: int, direction: str, start: float, end: float) -> Dict[str, Any]:
    mid = float((start + end) / 2.0)
    future = df.iloc[idx + 1 :]
    filled = False
    fill_time: Optional[str] = None
    invalidated = False
    invalidated_at: Optional[str] = None
    retest_after_invalid: Optional[str] = None
    active = True

    fill_level = mid if FVG_FILL_POLICY == "midline" else (end if direction == "bullish" else start)

    for ts, bar in future.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])

        if direction == "bullish":
            if not filled and low <= fill_level + FVG_TOL:
                filled = True
                fill_time = ts.isoformat()
            if not invalidated and close < start - FVG_TOL:
                invalidated = True
                invalidated_at = ts.isoformat()
                active = False
        else:
            if not filled and high >= fill_level - FVG_TOL:
                filled = True
                fill_time = ts.isoformat()
            if not invalidated and close > end + FVG_TOL:
                invalidated = True
                invalidated_at = ts.isoformat()
                active = False

        if invalidated and retest_after_invalid is None:
            if direction == "bullish" and high >= start - FVG_TOL:
                retest_after_invalid = ts.isoformat()
                break
            if direction == "bearish" and low <= end + FVG_TOL:
                retest_after_invalid = ts.isoformat()
                break

        if filled and not invalidated:
            active = False
            break

    return {
        "time": df.index[idx].isoformat(),
        "direction": direction,
        "start": float(start),
        "end": float(end),
        "mid": mid,
        "filled": filled,
        "fill_time": fill_time,
        "invalidated": invalidated,
        "invalidated_at": invalidated_at,
        "active": active,
        "retested_after_invalid": retest_after_invalid,
    }


def _detect_fvg(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if len(df) < 3:
        return out
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]
        if c1["high"] + FVG_TOL < c3["low"]:
            out.append(_build_fvg_record(df, i, "bullish", float(c1["high"]), float(c3["low"])) )
        if c1["low"] - FVG_TOL > c3["high"]:
            out.append(_build_fvg_record(df, i, "bearish", float(c3["high"]), float(c1["low"])) )
    return out


def _promote_ifvg(fvg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for gap in fvg or []:
        if not (gap.get("invalidated") and gap.get("retested_after_invalid")):
            continue
        flipped = "bearish" if gap["direction"] == "bullish" else "bullish"
        out.append({
            "time": gap["retested_after_invalid"],
            "direction": flipped,
            "start": gap["start"],
            "end": gap["end"],
            "source_time": gap.get("time"),
            "source_direction": gap.get("direction"),
            "retested_at": gap["retested_after_invalid"],
            "active": True,
        })
    return out


def _find_ob_seed(df: pd.DataFrame, idx: int, direction: str) -> Tuple[Optional[int], Optional[pd.Series]]:
    if idx <= 0:
        return None, None
    for j in range(idx - 1, -1, -1):
        row = df.iloc[j]
        if direction == "bullish" and row["close"] < row["open"]:
            return j, row
        if direction == "bearish" and row["close"] > row["open"]:
            return j, row
    return None, None


def _evaluate_ob_future(record: Dict[str, Any], df: pd.DataFrame, direction: str, low_val: float, high_val: float):
    retested_at: Optional[str] = None
    invalidated = False
    invalidated_at: Optional[str] = None
    retested_after_invalid: Optional[str] = None
    active = True

    for ts, bar in df.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])

        if not invalidated:
            if direction == "bullish" and retested_at is None and low <= high_val + OB_TOL:
                retested_at = ts.isoformat()
            if direction == "bearish" and retested_at is None and high >= low_val - OB_TOL:
                retested_at = ts.isoformat()

            if direction == "bullish" and close < low_val - OB_TOL:
                invalidated = True
                invalidated_at = ts.isoformat()
                active = False
            elif direction == "bearish" and close > high_val + OB_TOL:
                invalidated = True
                invalidated_at = ts.isoformat()
                active = False
        else:
            if direction == "bullish" and retested_after_invalid is None and high >= low_val - OB_TOL:
                retested_after_invalid = ts.isoformat()
                break
            if direction == "bearish" and retested_after_invalid is None and low <= high_val + OB_TOL:
                retested_after_invalid = ts.isoformat()
                break

    record["retested_at"] = retested_at
    record["invalidated"] = invalidated
    record["invalidated_at"] = invalidated_at
    record["retested_after_invalid"] = retested_after_invalid
    record["active"] = active


def _detect_ob(df: pd.DataFrame, atr: pd.Series) -> List[Dict[str, Any]]:
    if len(df) < 5:
        return []
    atr = atr.ffill()
    records: Dict[Tuple[pd.Timestamp, str], Dict[str, Any]] = {}

    for i in range(1, len(df)):
        cur = df.iloc[i]
        atr_window = atr.iloc[: i + 1].dropna()
        if atr_window.empty:
            continue
        atr_val = float(atr_window.iloc[-1])
        rng = float(cur["high"] - cur["low"])
        body = float(abs(cur["close"] - cur["open"]))
        if rng < OB_DISPLACEMENT_ATR_MULT * atr_val or body < OB_BODY_RATIO * rng:
            continue
        direction = "bullish" if cur["close"] >= cur["open"] else "bearish"
        seed_idx, seed_row = _find_ob_seed(df, i, direction)
        if seed_idx is None:
            continue
        key = (df.index[seed_idx], direction)
        if key in records:
            continue
        low_val = float(min(seed_row["open"], seed_row["close"], seed_row["low"]))
        high_val = float(max(seed_row["open"], seed_row["close"], seed_row["high"]))
        record = {
            "time": df.index[seed_idx].isoformat(),
            "direction": direction,
            "low": low_val,
            "high": high_val,
            "active": True,
            "invalidated": False,
            "invalidated_at": None,
            "retested_at": None,
            "retested_after_invalid": None,
        }
        future = df.iloc[seed_idx + 1 :]
        _evaluate_ob_future(record, future, direction, low_val, high_val)
        records[key] = record

    return list(records.values())


def _detect_breaker_blocks(ob_list: List[Dict[str, Any]], bos: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    trend = bos.get("trend")
    if not trend:
        return out
    target_direction = "bearish" if trend == "bullish" else "bullish"
    for ob in reversed(ob_list):
        if ob.get("direction") != target_direction:
            continue
        if not (ob.get("invalidated") and ob.get("retested_after_invalid")):
            continue
        out.append({
            "time": ob["retested_after_invalid"],
            "direction": trend,
            "source_ob_time": ob.get("time"),
            "low": ob.get("low"),
            "high": ob.get("high"),
            "active": True,
        })
        break
    return out


# -------------------------- Public API --------------------------
def compute_features_for_tf(bars: List[Dict[str, Any]], tf_label: str) -> Dict[str, Any]:
    df = _to_df(bars)
    if df.empty:
        return {
            "tf": tf_label,
            "atr_last": None,
            "ms": {"trend": None, "last_bos": None, "last_bos_time": None, "last_bos_level": None},
            "swings": {"eq": None, "ote": None, "leg": None},
            "fvg": [],
            "ifvg": [],
            "ob": [],
            "bb": [],
        }

    atr = _atr(df, ATR_PERIOD)
    _, sh_v, _, sl_v = _last_swings(df, k=K_PIVOT)
    bos = _detect_bos(df, sh_v, sl_v)
    swings = _compute_eq_ote(sh_v, sl_v)

    fvg = _detect_fvg(df)
    for rec in fvg:
        rec["tf"] = tf_label
    ifvg = _promote_ifvg(fvg)
    for rec in ifvg:
        rec["tf"] = tf_label
    ob = _detect_ob(df, atr)
    for rec in ob:
        rec["tf"] = tf_label
    bb = _detect_breaker_blocks(ob, bos)
    for rec in bb:
        rec["tf"] = tf_label

    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else None

    return {
        "tf": tf_label,
        "atr_last": atr_last,
        "ms": bos,
        "swings": swings,
        "fvg": fvg,
        "ifvg": ifvg,
        "ob": ob,
        "bb": bb,
    }


def simple_smt(primary_df: pd.DataFrame, ref_df: pd.DataFrame, window: int = 20) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if primary_df.empty or ref_df.empty:
        return out
    p = primary_df.tail(window)
    r = ref_df.tail(window)
    try:
        p_hh = p["high"].idxmax(); p_ll = p["low"].idxmin()
        r_hh = r["high"].idxmax(); r_ll = r["low"].idxmin()
        if p_hh > p.index.min() and r_hh > r.index.min():
            if p["high"].loc[p_hh] > p["high"].iloc[-2] and r["high"].loc[r_hh] <= r["high"].iloc[-2]:
                out.append({"kind": "SMT_bearish", "time": p.index[-1].isoformat()})
        if p_ll > p.index.min() and r_ll > r.index.min():
            if p["low"].loc[p_ll] < p["low"].iloc[-2] and r["low"].loc[r_ll] >= r["low"].iloc[-2]:
                out.append({"kind": "SMT_bullish", "time": p.index[-1].isoformat()})
    except Exception:
        return out
    return out


def compute_smt_for_pair(primary_bars: List[Dict[str, Any]], ref_bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    p_df = _to_df(primary_bars)
    r_df = _to_df(ref_bars)
    return simple_smt(p_df, r_df)


