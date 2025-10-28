Below are **three new modules** and a **small patch** for `live_loop.py`.

- Drop these files under `src/features/`:
  - `ict_detectors.py`
  - `feature_packager.py`
  - `tracker.py`
- Then apply the small patch to `src/strategy/live_loop.py` (section at the end) to wire them in.

---

## `src/features/ict_detectors.py`

```python
#!/usr/bin/env python3
"""
Deterministic ICT feature detectors (v0.1).

Inputs
------
- Bars as a list[dict] with keys: t (ISO str, UTC), o,h,l,c,v (floats)
- We convert to a pandas DataFrame indexed by UTC timestamp for convenience.

Outputs
-------
- Feature records (lists of dicts) that are easy to serialize and feed into prompts.
- We compute on a per-timeframe basis (caller passes 1m or 5m arrays).

Detectors included (initial cut)
--------------------------------
- ATR(14)
- Swings (pivot-based), Market Structure & BOS
- EQ (50%), OTE zone (62%–79%) for the active HTF leg
- FVG & basic IFVG flip detection
- OB (last down/up before displacement) & Breaker Block (basic)
- SMT divergence (for a pair) — simple HH/LL mismatch within a small window

Notes
-----
- Thresholds are intentionally conservative and centralized in CONSTANTS for tuning.
- All times are treated as UTC. Caller should hand us SPLIT-adjusted prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ------------------------- Constants (tunable) -------------------------
K_PIVOT = 2                  # left/right pivot width
ATR_PERIOD = 14
DISPLACEMENT_ATR_MULT = 1.2  # impulse threshold for OB detection
FVG_FILL_POLICY = "midline"  # "midline" or "full"
IFVG_RETEST_BARS = 8         # bars to look for a retest after trade-through
BREAKER_LOOKAHEAD = 5        # bars after OB invalidation to find opposite BOS
SMT_WINDOW_BARS = 12         # search window for swing mismatch
PRICE_TOL = 1e-6

# ---------------------------- Utilities -------------------------------

def _to_df(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"]).astype({
            "o": float, "h": float, "l": float, "c": float, "v": float
        })
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.set_index("t").sort_index()
    return df[["o", "h", "l", "c", "v"]].astype(float)


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    h, l, c = df["h"], df["l"], df["c"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


# ------------------------- Swings & BOS -------------------------------

def find_swings(df: pd.DataFrame, k: int = K_PIVOT) -> Tuple[pd.Series, pd.Series]:
    """Return boolean Series for swing highs and swing lows (confirmed pivots)."""
    if len(df) < 2 * k + 1:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    highs = df["h"].values
    lows = df["l"].values
    sh = np.zeros(len(df), dtype=bool)
    sl = np.zeros(len(df), dtype=bool)
    for i in range(k, len(df) - k):
        if highs[i] > highs[i-k:i].max() and highs[i] > highs[i+1:i+k+1].max():
            sh[i] = True
        if lows[i] < lows[i-k:i].min() and lows[i] < lows[i+1:i+k+1].min():
            sl[i] = True
    return pd.Series(sh, index=df.index), pd.Series(sl, index=df.index)


def detect_bos(df: pd.DataFrame, sh: pd.Series, sl: pd.Series) -> Dict[str, Any]:
    """
    Detect the last Break of Structure. We say BOS up if a close exceeds the last
    confirmed swing high; BOS down if a close falls below the last confirmed swing low.
    Returns a dict {last_bos: 'bull'/'bear'/None, level, time}.
    """
    last_bos = None
    level = None
    ts = None

    swing_highs = df["h"][sh]
    swing_lows = df["l"][sl]
    c = df["c"]

    # Find latest BOS (scan forward)
    last_sh_idx = None
    last_sl_idx = None
    for t in df.index:
        if sh.loc[t]:
            last_sh_idx = t
        if sl.loc[t]:
            last_sl_idx = t
        if last_sh_idx is not None and c.loc[t] > df.loc[last_sh_idx, "h"] + PRICE_TOL:
            last_bos = "bull"
            level = float(df.loc[last_sh_idx, "h"])
            ts = t
        if last_sl_idx is not None and c.loc[t] < df.loc[last_sl_idx, "l"] - PRICE_TOL:
            last_bos = "bear"
            level = float(df.loc[last_sl_idx, "l"])
            ts = t

    trend = None
    if last_bos == "bull":
        trend = "up"
    elif last_bos == "bear":
        trend = "down"

    return {
        "trend": trend,
        "last_bos": last_bos,
        "last_bos_time": ts.isoformat() if ts is not None else None,
        "last_bos_level": level,
    }


def compute_eq_ote(df: pd.DataFrame, sh: pd.Series, sl: pd.Series, bos: Dict[str, Any]) -> Dict[str, Any]:
    """Compute EQ and OTE (62–79%) for the active leg around the last BOS.
    If no BOS, use the most recent swing-to-swing range.
    """
    if df.empty:
        return {"eq": None, "ote": None, "leg": None}

    swings = pd.concat([
        pd.DataFrame({"t": sh[sh].index, "price": df["h"][sh]}).assign(kind="H"),
        pd.DataFrame({"t": sl[sl].index, "price": df["l"][sl]}).assign(kind="L"),
    ]).sort_values("t")
    if swings.empty:
        # fallback:  lookback range
        high = float(df["h"].tail(50).max()) if len(df) else None
        low = float(df["l"].tail(50).min()) if len(df) else None
        if high is None or low is None:
            return {"eq": None, "ote": None, "leg": None}
        eq = (high + low) / 2.0
        ote = (low + 0.62 * (high - low), low + 0.79 * (high - low))
        return {"eq": eq, "ote": list(ote), "leg": {"high": high, "low": low}}

    # Try to pick last leg using BOS direction
    last = swings.tail(3)
    if bos.get("last_bos") == "bull":
        # Leg is last swing low -> last swing high
        lows = swings[swings["kind"] == "L"]["price"]
        highs = swings[swings["kind"] == "H"]["price"]
        if not lows.empty and not highs.empty:
            low = float(lows.tail(1).values[0])
            high = float(highs.tail(1).values[0])
        else:
            low = float(df["l"].tail(50).min())
            high = float(df["h"].tail(50).max())
    elif bos.get("last_bos") == "bear":
        highs = swings[swings["kind"] == "H"]["price"]
        lows = swings[swings["kind"] == "L"]["price"]
        if not lows.empty and not highs.empty:
            high = float(highs.tail(1).values[0])
            low = float(lows.tail(1).values[0])
        else:
            low = float(df["l"].tail(50).min())
            high = float(df["h"].tail(50).max())
    else:
        # no BOS; use recent range
        low = float(df["l"].tail(50).min())
        high = float(df["h"].tail(50).max())

    eq = (high + low) / 2.0
    ote = (low + 0.62 * (high - low), low + 0.79 * (high - low))
    return {"eq": eq, "ote": [float(ote[0]), float(ote[1])], "leg": {"high": high, "low": low}}


# --------------------------- FVG & IFVG -------------------------------

def detect_fvg(df: pd.DataFrame) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if len(df) < 3:
        return recs
    idx = df.index
    h = df["h"].values
    l = df["l"].values
    c = df["c"].values

    for i in range(1, len(df) - 1):
        # bar i-1, i, i+1
        # Bull FVG when high[i-1] < low[i+1]
        if h[i-1] + PRICE_TOL < l[i+1]:
            lower, upper = h[i-1], l[i+1]
            recs.append({
                "tf": None,  # set by caller
                "dir": "bull",
                "start": idx[i-1].isoformat(),
                "gap": [float(lower), float(upper)],
                "filled": False,
                "active": True,
                "kind": "FVG"
            })
        # Bear FVG when low[i-1] > high[i+1]
        if l[i-1] - PRICE_TOL > h[i+1]:
            lower, upper = h[i+1], l[i-1]
            recs.append({
                "tf": None,
                "dir": "bear",
                "start": idx[i-1].isoformat(),
                "gap": [float(lower), float(upper)],
                "filled": False,
                "active": True,
                "kind": "FVG"
            })

    # Post-process fill/trade-through against the full series end state
    last_close = float(df["c"].iloc[-1]) if len(df) else None
    for r in recs:
        lo, hi = r["gap"]
        mid = (lo + hi) / 2.0
        if FVG_FILL_POLICY == "full":
            filled = (r["dir"] == "bull" and last_close <= lo + PRICE_TOL) or \
                     (r["dir"] == "bear" and last_close >= hi - PRICE_TOL)
        else:  # midline
            filled = (r["dir"] == "bull" and last_close <= mid + PRICE_TOL) or \
                     (r["dir"] == "bear" and last_close >= mid - PRICE_TOL)
        r["filled"] = bool(filled)
        # trade-through: close beyond far bound
        traded_through = (r["dir"] == "bull" and last_close < lo - PRICE_TOL) or \
                          (r["dir"] == "bear" and last_close > hi + PRICE_TOL)
        r["traded_through"] = bool(traded_through)
        if filled:
            r["active"] = False

    return recs


def promote_ifvg(df: pd.DataFrame, fvgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert traded-through FVGs to IFVG if a retest occurs within window.
    Simplified rule: after trade-through, if within IFVG_RETEST_BARS we see a close
    back into the original gap range, mark IFVG with flip_role and active=True.
    """
    out: List[Dict[str, Any]] = []
    if df.empty:
        return out
    idx = df.index
    c = df["c"]; h = df["h"]; l = df["l"]

    for r in fvgs:
        if not r.get("traded_through"):
            continue
        start_ts = pd.Timestamp(r["start"])  # utc
        # find index of start
        try:
            start_pos = idx.get_indexer([start_ts], method="nearest")[0]
        except Exception:
            start_pos = max(0, len(idx) - IFVG_RETEST_BARS - 1)
        lo, hi = r["gap"]
        dir_ = r["dir"]
        look_end = min(len(idx), start_pos + IFVG_RETEST_BARS + 1)
        retested = False
        for j in range(start_pos, look_end):
            price = float(c.iloc[j])
            if lo - PRICE_TOL <= price <= hi + PRICE_TOL:
                retested = True
                ret_ts = idx[j]
                break
        if retested:
            out.append({
                "tf": r.get("tf"),
                "dir": dir_,
                "gap": r["gap"],
                "start": r["start"],
                "flip_role": "support" if dir_ == "bull" else "resistance",
                "active": True,
                "kind": "IFVG",
                "retested_at": ret_ts.isoformat(),
            })
    return out


# ---------------------------- Order Blocks ----------------------------

def detect_ob(df: pd.DataFrame, sh: pd.Series, sl: pd.Series, atr: pd.Series) -> List[Dict[str, Any]]:
    """
    Basic OB detector:
    - Bull OB: last down-close before an upward displacement (close rise > ATR*mult OR break above last swing high)
    - Bear OB: last up-close before a downward displacement (close fall > ATR*mult OR break below last swing low)
    """
    recs: List[Dict[str, Any]] = []
    if len(df) < 3:
        return recs

    c = df["c"]; o = df["o"]; h = df["h"]; l = df["l"]
    last_sh_price = None
    last_sl_price = None
    last_sh = sh[sh].index.tolist()
    last_sl = sl[sl].index.tolist()
    if last_sh:
        last_sh_price = float(h.loc[last_sh[-1]])
    if last_sl:
        last_sl_price = float(l.loc[last_sl[-1]])

    for i in range(1, len(df) - 1):
        body_dir = "down" if c.iloc[i] < o.iloc[i] else "up"
        # Upward displacement signal
        up_displacement = (c.iloc[i+1] - c.iloc[i]) > DISPLACEMENT_ATR_MULT * (atr.iloc[i] or 0)
        up_break = last_sh_price is not None and c.iloc[i+1] > last_sh_price + PRICE_TOL
        # Downward displacement signal
        dn_displacement = (c.iloc[i] - c.iloc[i+1]) > DISPLACEMENT_ATR_MULT * (atr.iloc[i] or 0)
        dn_break = last_sl_price is not None and c.iloc[i+1] < last_sl_price - PRICE_TOL

        if body_dir == "down" and (up_displacement or up_break):
            # Bull OB at candle i
            recs.append({
                "tf": None,
                "dir": "bull",
                "time": df.index[i].isoformat(),
                "body": [float(min(o.iloc[i], c.iloc[i])), float(max(o.iloc[i], c.iloc[i]))],
                "distal": float(l.iloc[i]),
                "active": True,
                "kind": "OB"
            })
        if body_dir == "up" and (dn_displacement or dn_break):
            # Bear OB at candle i
            recs.append({
                "tf": None,
                "dir": "bear",
                "time": df.index[i].isoformat(),
                "body": [float(min(o.iloc[i], c.iloc[i])), float(max(o.iloc[i], c.iloc[i]))],
                "distal": float(h.iloc[i]),
                "active": True,
                "kind": "OB"
            })

    # Invalidation: close beyond distal
    last_close = float(c.iloc[-1]) if len(df) else None
    for r in recs:
        if r["dir"] == "bull" and last_close is not None and last_close < r["distal"] - PRICE_TOL:
            r["active"] = False
            r["invalidated"] = True
        elif r["dir"] == "bear" and last_close is not None and last_close > r["distal"] + PRICE_TOL:
            r["active"] = False
            r["invalidated"] = True
        else:
            r["invalidated"] = False
    return recs


def detect_breaker_blocks(ob_list: List[Dict[str, Any]], bos: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Promote invalidated OBs to basic Breaker Blocks if a BOS in the opposite
    direction is present in the same window.
    """
    out: List[Dict[str, Any]] = []
    opp = "bear" if bos.get("last_bos") == "bull" else ("bull" if bos.get("last_bos") == "bear" else None)
    if opp is None:
        return out
    for ob in ob_list:
        if ob.get("invalidated") and ob.get("dir") != opp:
            out.append({
                "tf": ob.get("tf"),
                "dir": opp,
                "level": float(ob.get("body", [None, None])[1 if opp == "bear" else 0] or ob.get("distal")),
                "from_ob_time": ob.get("time"),
                "active": True,
                "kind": "BB"
            })
    return out


# ------------------------------- SMT ----------------------------------

def simple_smt(primary_df: pd.DataFrame, ref_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Simple SMT divergence: over last SMT_WINDOW_BARS, if primary prints HH
    while ref fails to do so (LH), flag *bear* SMT; analog for LL vs HL -> *bull* SMT.
    We use swing pivots with K_PIVOT for robustness.
    """
    out: List[Dict[str, Any]] = []
    if primary_df.empty or ref_df.empty:
        return out

    # Align time windows (inner join on timestamps)
    both = primary_df[["h", "l", "c"]].join(ref_df[["h", "l", "c"]], how="inner", lsuffix="_p", rsuffix="_r")
    if len(both) < 2 * K_PIVOT + 3:
        return out

    w = both.tail(SMT_WINDOW_BARS)
    # find last swings on both
    sh_p, sl_p = find_swings(w[["o_p" if "o_p" in w else "h_p", "h_p", "l_p", "c_p"]].rename(columns={"o_p":"o","h_p":"h","l_p":"l","c_p":"c"}), K_PIVOT)
    sh_r, sl_r = find_swings(w[["o_r" if "o_r" in w else "h_r", "h_r", "l_r", "c_r"]].rename(columns={"o_r":"o","h_r":"h","l_r":"l","c_r":"c"}), K_PIVOT)

    try:
        last_sh_p = float(w["h_p"][sh_p].tail(1).values[0])
        last_sh_r = float(w["h_r"][sh_r].tail(1).values[0])
    except Exception:
        last_sh_p = None; last_sh_r = None
    try:
        last_sl_p = float(w["l_p"][sl_p].tail(1).values[0])
        last_sl_r = float(w["l_r"][sl_r].tail(1).values[0])
    except Exception:
        last_sl_p = None; last_sl_r = None

    if last_sh_p is not None and last_sh_r is not None:
        # Bear SMT: primary HH while ref fails (LH or equal)
        if last_sh_p > last_sh_r + PRICE_TOL:
            out.append({"pair": "primary/ref", "type": "bear", "time": w.index[-1].isoformat()})
    if last_sl_p is not None and last_sl_r is not None:
        # Bull SMT: primary LL while ref fails (HL or equal)
        if last_sl_p < last_sl_r - PRICE_TOL:
            out.append({"pair": "primary/ref", "type": "bull", "time": w.index[-1].isoformat()})

    return out


# -------------------------- Public API --------------------------------

def compute_features_for_tf(
    bars: List[Dict[str, Any]],
    tf_label: str,
    ref_pair_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute all detectors for a single timeframe.

    Returns a dict with: atr_last, ms (trend/BOS), swings (eq/ote), fvg, ifvg, ob, bb.
    """
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

    atr = compute_atr(df, ATR_PERIOD)
    sh, sl = find_swings(df, K_PIVOT)
    bos = detect_bos(df, sh, sl)
    swings = compute_eq_ote(df, sh, sl, bos)

    fvg = detect_fvg(df)
    # fill tf labels
    for r in fvg:
        r["tf"] = tf_label
    ifvg = promote_ifvg(df, fvg)
    ob = detect_ob(df, sh, sl, atr)
    for r in ob:
        r["tf"] = tf_label
    bb = detect_breaker_blocks(ob, bos)
    for r in bb:
        r["tf"] = tf_label

    return {
        "tf": tf_label,
        "atr_last": float(atr.iloc[-1]) if len(atr) else None,
        "ms": bos,
        "swings": swings,
        "fvg": fvg,
        "ifvg": ifvg,
        "ob": ob,
        "bb": bb,
    }


def compute_smt_for_pair(primary_bars: List[Dict[str, Any]], ref_bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    p_df = _to_df(primary_bars)
    r_df = _to_df(ref_bars)
    return simple_smt(p_df, r_df)
```

---

## `src/features/feature_packager.py`

```python
#!/usr/bin/env python3
"""
Feature packager — runs ICT detectors per symbol & timeframe and writes capsules.

- Reads SPLIT-adjusted bar slices provided by the live/test loop.
- Computes features on 5m and 1m timeframes.
- Optionally computes SMT for index pairs (QQQ vs SPY, IWM vs SPY).
- Writes/updates `data/daily_news/YYYY-MM-DD/processed/context_capsules/<SYMBOL>.json`.

This module has no Alpaca/Gemini deps — pure feature computation + file I/O.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import json

from .ict_detectors import compute_features_for_tf, compute_smt_for_pair


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def capsule_path(project_root: Path, day_et: str, symbol: str) -> Path:
    base = project_root / "data" / "daily_news" / day_et / "processed" / "context_capsules"
    ensure_dir(base)
    return base / f"{symbol}.json"


def build_capsule(
    symbol: str,
    day_et: str,
    bars_1m: List[Dict[str, Any]],
    bars_5m: List[Dict[str, Any]],
    ref_pairs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    prior_levels: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute per-symbol features and return a capsule dict."""
    f5 = compute_features_for_tf(bars_5m, "5m")
    f1 = compute_features_for_tf(bars_1m, "1m")

    smt: List[Dict[str, Any]] = []
    if ref_pairs and symbol in ref_pairs:
        try:
            smt = compute_smt_for_pair(bars_5m, ref_pairs[symbol])
            for r in smt:
                r["tf"] = "5m"
        except Exception:
            smt = []

    return {
        "symbol": symbol,
        "day_et": day_et,
        "features": {
            "tf_5m": f5,
            "tf_1m": f1,
            "smt": smt,
            "levels": prior_levels or {},
        },
        "meta": {
            "generated_at_utc": _now_utc_iso(),
        }
    }


def write_capsule(project_root: Path, capsule: Dict[str, Any]) -> Path:
    sym = capsule.get("symbol")
    day_et = capsule.get("day_et")
    p = capsule_path(project_root, day_et, sym)
    with p.open("w", encoding="utf-8") as f:
        json.dump(capsule, f, indent=2)
    return p
```

---

## `src/features/tracker.py`

```python
#!/usr/bin/env python3
"""
ConfluenceTracker — keeps per-symbol state (active/inactive confluences, stages) and
merges new detector outputs into `current_context.json` consistently.

- Status: forming | active | filled | invalidated | expired
- Stages: htf_stage ∈ {none, reversal_seen, confirmed}, ltf_stage ∈ {none, scanning, entry_window, invalidated}
- Only confluences with status == active should be surfaced as `active_confluences` in prompts.
"""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime, timezone, timedelta

STATUS_TTL_MIN = 90  # expire if unseen for this many minutes


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mk_id(kind: str, tf: str, key_time: str) -> str:
    return f"{kind}:{tf}:{key_time}"


def _short(confl: Dict[str, Any]) -> str:
    k = confl.get("kind")
    tf = confl.get("tf")
    if k == "FVG":
        lo, hi = confl.get("gap", [None, None])
        return f"FVG({tf} {confl.get('dir')} @ {lo:.2f}–{hi:.2f})" if lo and hi else f"FVG({tf})"
    if k == "IFVG":
        lo, hi = confl.get("gap", [None, None])
        return f"IFVG({tf} {confl.get('dir')} @ {lo:.2f}–{hi:.2f})"
    if k == "OB":
        b = confl.get("body", [None, None])
        return f"OB({tf} {confl.get('dir')} @ {b[0]:.2f}–{b[1]:.2f})" if b[0] and b[1] else f"OB({tf})"
    if k == "BB":
        return f"BB({tf} {confl.get('dir')} @ {confl.get('level'):.2f})" if confl.get('level') else f"BB({tf})"
    if k == "BOS":
        return f"BOS({tf} {confl.get('dir')} @ {confl.get('level'):.2f})" if confl.get('level') else f"BOS({tf})"
    return f"{k}({tf})"


def _merge_list(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {e.get("id"): e for e in existing}
    for inc in incoming:
        inc_id = inc.get("id")
        if inc_id in by_id:
            # update fields & timestamps
            by_id[inc_id].update(inc)
            by_id[inc_id]["last_seen_utc"] = _now_utc_iso()
        else:
            inc["first_seen_utc"] = _now_utc_iso()
            inc["last_seen_utc"] = inc["first_seen_utc"]
            by_id[inc_id] = inc
    # expire old
    now = datetime.now(timezone.utc)
    pruned: List[Dict[str, Any]] = []
    for e in by_id.values():
        try:
            last_seen = datetime.fromisoformat(e.get("last_seen_utc"))
        except Exception:
            last_seen = now
        if e.get("status") in ("active", "forming"):
            pruned.append(e)  # keep active/forming regardless of age
        else:
            # filled/invalidated/expired -> TTL short retention
            if (now - last_seen) <= timedelta(minutes=STATUS_TTL_MIN):
                pruned.append(e)
    # sort by last_seen
    pruned.sort(key=lambda x: x.get("last_seen_utc", ""))
    return pruned


def _bos_record(tf_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    bos = tf_state.get("ms", {})
    if bos.get("last_bos"):
        return [{
            "id": _mk_id("BOS", tf_state.get("tf", "?"), bos.get("last_bos_time")),
            "kind": "BOS",
            "tf": tf_state.get("tf", "?"),
            "dir": "bull" if bos.get("last_bos") == "bull" else "bear",
            "level": bos.get("last_bos_level"),
            "status": "active",
        }]
    return []


def _statusize(kind: str, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in recs:
        rr = dict(r)
        rr["kind"] = kind
        # assign ID field based on a reasonable key
        key_time = r.get("time") or r.get("start") or r.get("retested_at") or _now_utc_iso()
        rr["id"] = _mk_id(kind, r.get("tf", "?"), key_time)
        # status rules (first pass)
        status = "active"
        if kind == "FVG":
            if r.get("filled"):
                status = "filled"
            elif not r.get("active", True):
                status = "invalidated"
        if kind == "OB":
            if r.get("invalidated"):
                status = "invalidated"
        rr["status"] = status
        out.append(rr)
    return out


def update_symbol_state(existing: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new feature snapshot into the persistent per-symbol state.
    - existing: current state slice (may be empty)
    - features: output of packager for this symbol (features.tf_5m, tf_1m, smt, levels)
    """
    state = dict(existing.get("state", {}))

    # Build incoming confluence set from features
    incoming: List[Dict[str, Any]] = []
    tf5 = features.get("tf_5m", {})
    tf1 = features.get("tf_1m", {})

    incoming += _statusize("FVG", tf5.get("fvg", []))
    incoming += _statusize("IFVG", tf5.get("ifvg", []))
    incoming += _statusize("OB", tf5.get("ob", []))
    incoming += _statusize("BB", tf5.get("bb", []))
    incoming += _bos_record(tf5)

    # LTF, too (useful for entry_window)
    incoming += _statusize("FVG", tf1.get("fvg", []))
    incoming += _statusize("IFVG", tf1.get("ifvg", []))
    incoming += _statusize("OB", tf1.get("ob", []))
    incoming += _statusize("BB", tf1.get("bb", []))

    # Merge
    merged = _merge_list(existing.get("confluences", []), incoming)

    # Stage logic (first pass):
    htf_stage = state.get("htf_stage", "none")
    ltf_stage = state.get("ltf_stage", "none")

    # If 5m BOS present, confirm HTF
    if tf5.get("ms", {}).get("last_bos"):
        htf_stage = "confirmed"
    else:
        # if we have IFVG/OB forming on 5m without BOS, call it reversal_seen
        if tf5.get("ifvg") or tf5.get("ob"):
            htf_stage = "reversal_seen"

    # LTF only valid after HTF confirmed
    if htf_stage == "confirmed":
        # if we have any active LTF confluence, we are in entry window
        has_ltf_active = any(c.get("tf") == "1m" and c.get("status") == "active" for c in merged)
        ltf_stage = "entry_window" if has_ltf_active else "scanning"
    else:
        ltf_stage = "none"

    # Filter active confluences for easy prompt consumption
    active = [c for c in merged if c.get("status") == "active"]
    active_short = [_short(c) for c in active]

    new_state = {
        "htf_stage": htf_stage,
        "ltf_stage": ltf_stage,
        "last_update_utc": _now_utc_iso(),
        "confluences": merged,
    }

    return {
        **existing,
        "state": new_state,
        "active_confluences": active_short,
    }
```

---

## Patch for `src/strategy/live_loop.py` (wire in features + tracker)

> Add these imports near the top (after existing imports):

```python
# New: feature computation & tracking
from pathlib import Path
from src.features.feature_packager import build_capsule, write_capsule
from src.features.tracker import update_symbol_state
```

> After you slice/fetch bars each fast tick (both test and live), **before** building the fast prompt, insert this per-symbol feature + state update block:

```python
# --- NEW: compute features & update state per symbol ---
# bars structure we already have: {"bars_1m": {sym: [...]}, "bars_5m": {sym: [...]}}
# Prepare optional SMT ref pairs for indices (QQQ/IWM vs SPY)
ref_pairs = {}
if "SPY" in WATCHLIST:
    # Use SPY 5m bars as reference for index peers
    ref_pairs = {
        "QQQ": bars["bars_5m"].get("SPY", []),
        "IWM": bars["bars_5m"].get("SPY", []),
    }

# Load current_context once
ctx_doc = load_json(paths["current_context"]) or {"symbols": []}
sym_map = {s.get("symbol"): s for s in ctx_doc.get("symbols", []) if isinstance(s, dict)}

for sym in WATCHLIST:
    b1 = bars["bars_1m"].get(sym, [])
    b5 = bars["bars_5m"].get(sym, [])
    # Optionally pass prior-day levels from key_levels.json if you have them
    prior_levels = load_json(paths["key_levels"]) or {}
    prior_levels_sym = prior_levels.get(sym) if isinstance(prior_levels, dict) else {}

    capsule = build_capsule(
        symbol=sym,
        day_et=paths["day"],
        bars_1m=b1,
        bars_5m=b5,
        ref_pairs={sym: ref_pairs.get(sym, [])} if sym in ref_pairs else None,
        prior_levels=prior_levels_sym,
    )
    # Save capsule for transparency / LLM prompt consumption
    write_capsule(PROJECT_ROOT, capsule)

    # Update persistent per-symbol state in current_context
    existing = sym_map.get(sym, {"symbol": sym})
    features = capsule.get("features", {})
    updated = update_symbol_state(existing, features)
    sym_map[sym] = updated

# Write back the merged context before LLM prompts
ctx_doc["symbols"] = list(sym_map.values())
save_json(paths["current_context"], ctx_doc)
```

> In your **per-symbol** prompt builder (where you assemble input for Gemini), make sure to pull only **active** confluences and stages from `current_context.json` for that symbol, e.g.:

```python
# Inside your per-symbol prompt construction
sym_state = next((s for s in (load_json(paths["current_context"]) or {}).get("symbols", []) if s.get("symbol") == symbol), {})
stage = {"htf": sym_state.get("state", {}).get("htf_stage"), "ltf": sym_state.get("state", {}).get("ltf_stage")}
active_cons = sym_state.get("active_confluences", [])

prompt += f"\n## CONTEXT_STAGE\n{json.dumps(stage)}\n"
prompt += f"\n## ACTIVE_CONFLUENCES\n{json.dumps(active_cons)}\n"
```

*(If your current prompt builder already pulls the whole `current_context` blob, just switch to using `active_confluences` and `state.htf_stage/ltf_stage` instead of any raw lists — that’s the key change.)*

---

### Notes
- This is a **foundational** implementation meant to be safe and incremental:
  - Detectors use conservative rules and come with tunable constants.
  - Tracker enforces stage gating and ensures only **active** confluences are surfaced.
- Next iterations can tighten definitions (e.g., OB criteria, IFVG nuance), add ATR sanity constraints to the post-validator, and expand SMT logic.
- Everything writes sidecar artifacts (capsules & context) so you can inspect easily during test replays.
```

