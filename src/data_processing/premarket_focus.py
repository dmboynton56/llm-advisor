#!/usr/bin/env python3
"""
Builds the premarket focus list and per-symbol context capsules.

- Reads:
  data/daily_news/YYYY-MM-DD/processed/processed_briefing.json
  data/daily_news/YYYY-MM-DD/raw/key_levels.json
- Always includes: SPY, QQQ, IWM (overrideable via --always)
- Selects 1–2 extra symbols by score (based on final_confidence if present)
- Writes:
  data/daily_news/YYYY-MM-DD/processed/focus_list.json
  data/daily_news/YYYY-MM-DD/processed/context_capsules/<SYMBOL>.json

Usage:
  python src/strategy/premarket_focus.py
  python src/strategy/premarket_focus.py --date 2025-10-14
  python src/strategy/premarket_focus.py --always SPY,QQQ,IWM --extras 2
  python src/strategy/premarket_focus.py --manual NVDA,AMZN   # force extras
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytz

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ET = pytz.timezone("US/Eastern")

# --- IO helpers ---
def load_json(p: Path) -> Any:
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def day_paths(day_et: str) -> Dict[str, Path]:
    base_proc = PROJECT_ROOT / "data" / "daily_news" / day_et / "processed"
    base_raw  = PROJECT_ROOT / "data" / "daily_news" / day_et / "raw"
    return {
        "processed_briefing": base_proc / "processed_briefing.json",
        "key_levels": base_raw / "key_levels.json",
        "focus_list": base_proc / "focus_list.json",
        "capsules_dir": base_proc / "context_capsules",
    }

# --- scoring helpers ---
def _parse_float_maybe(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"[-+]?\d*\.?\d+", x)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None

def score_symbol(sym: str, node: Dict[str, Any]) -> float:
    """
    Base score: abs(final_confidence) if available (assuming 0..1),
    else abs(ml_confidence). Optional tiny boosts if structured fields exist.
    """
    fc = _parse_float_maybe(node.get("final_confidence"))
    mc = _parse_float_maybe(node.get("ml_confidence"))
    base = abs(fc if fc is not None else (mc if mc is not None else 0.0))

    # Optional boosters if your processed_briefing includes these as structured fields.
    boost = 0.0
    gap = _parse_float_maybe(node.get("gap_pct"))  # e.g., -1.92
    if gap is not None and abs(gap) >= 1.0:
        boost += 0.05

    volrel = _parse_float_maybe(node.get("volume_rel_5d"))
    if volrel is not None and volrel >= 2.0:
        boost += 0.05

    dconf = _parse_float_maybe(node.get("delta_confidence"))
    if dconf is not None and abs(dconf) >= 0.25:
        boost += 0.05

    return float(base + boost)

def pick_focus_symbols(always: List[str],
                       symbols_blob: Dict[str, Any],
                       manual_extras: Optional[List[str]] = None,
                       extras: int = 2) -> Dict[str, Any]:
    """
    Returns dict with ranked candidates and final focus list:
    {
      "always": [...],
      "candidates_scored": [{"symbol": "NVDA", "score": 0.91, "reasons": [...]}, ...],
      "focus": ["SPY","QQQ","IWM","NVDA","AMZN"]
    }
    """
    # rank all non-always symbols by score
    scored = []
    for sym, node in symbols_blob.items():
        if sym in always:
            continue
        s = score_symbol(sym, node)
        reasons = []
        fc = _parse_float_maybe(node.get("final_confidence"))
        if fc is not None:
            reasons.append(f"final_conf {fc:.3f}")
        gap = _parse_float_maybe(node.get("gap_pct"))
        if gap is not None:
            reasons.append(f"gap {gap:.2f}%")
        volrel = _parse_float_maybe(node.get("volume_rel_5d"))
        if volrel is not None:
            reasons.append(f"vol_rel {volrel:.2f}x")
        scored.append({"symbol": sym, "score": s, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)

    if manual_extras:
        # use manual symbols if provided (and exist)
        selected = [s for s in manual_extras if s in symbols_blob and s not in always]
    else:
        selected = [x["symbol"] for x in scored[:max(0, extras)]]

    focus = list(dict.fromkeys(always + selected))  # unique preserve order
    return {
        "always": always,
        "candidates_scored": scored,
        "focus": focus,
    }

def build_capsule(sym: str,
                  brief_sym: Dict[str, Any],
                  key_levels_map: Dict[str, Any],
                  threshold_conf: int = 70,
                  rr_min: float = 1.0) -> Dict[str, Any]:
    """Create the compact context capsule for a symbol."""
    # Extract fields safely
    pre = {
        "final_bias": brief_sym.get("final_bias"),
        "final_confidence": _parse_float_maybe(brief_sym.get("final_confidence")),
        "delta_confidence": _parse_float_maybe(brief_sym.get("delta_confidence")),
        "ml_bias": brief_sym.get("ml_bias"),
        "ml_confidence": _parse_float_maybe(brief_sym.get("ml_confidence")),
    }

    # Optional structured extras if your briefing provides them
    for k in ("gap_pct", "volume_rel_5d"):
        if k in brief_sym:
            pre[k] = _parse_float_maybe(brief_sym.get(k))

    # Key levels
    levels = {}
    kl_sym = key_levels_map.get(sym) if isinstance(key_levels_map, dict) else None
    if isinstance(kl_sym, dict):
        # include only simple numeric levels
        for k, v in kl_sym.items():
            fv = _parse_float_maybe(v)
            if fv is not None:
                levels[k] = fv

    notes = []
    reasoning = brief_sym.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        # Trim the reasoning to keep capsules compact
        snippet = reasoning.strip()
        if len(snippet) > 360:
            snippet = snippet[:357] + "..."
        notes.append(snippet)

    return {
        "symbol": sym,
        "premarket": pre,
        "htf_levels": levels,
        "notes": notes,
        "strategy_meta": {
            "threshold_confidence": threshold_conf,
            "rr_min": rr_min
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat()
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (ET). Defaults to today.", default=None)
    ap.add_argument("--always", help="Comma list of always-on symbols",
                    default="SPY,QQQ,IWM")
    ap.add_argument("--extras", help="Number of extra symbols to add (default 2)",
                    type=int, default=2)
    ap.add_argument("--manual", help="Comma list of manual extra symbols (overrides scoring)",
                    default=None)
    args = ap.parse_args()

    # Resolve ET date string
    if args.date:
        day_str = args.date
    else:
        day_str = datetime.now(ET).strftime("%Y-%m-%d")

    paths = day_paths(day_str)
    briefing = load_json(paths["processed_briefing"]) or {}
    key_levels = load_json(paths["key_levels"]) or {}

    # drill into your briefing structure
    symbols_blob = {}
    try:
        symbols_blob = (
            briefing.get("model_responses", {})
                    .get("gemini", {})
                    .get("symbols", {})
        ) or {}
    except Exception:
        symbols_blob = {}

    if not symbols_blob:
        raise SystemExit(f"ERROR: No symbols found in {paths['processed_briefing']}")

    always = [s.strip().upper() for s in args.always.split(",") if s.strip()]
    manual_extras = None
    if args.manual:
        manual_extras = [s.strip().upper() for s in args.manual.split(",") if s.strip()]

    selection = pick_focus_symbols(always, symbols_blob,
                                   manual_extras=manual_extras,
                                   extras=args.extras)

    # Write focus_list.json
    focus_doc = {
        "date_et": day_str,
        **selection
    }
    save_json(paths["focus_list"], focus_doc)

    # Write capsules for each symbol in focus
    caps_dir = paths["capsules_dir"]
    caps_dir.mkdir(parents=True, exist_ok=True)

    for sym in selection["focus"]:
        if sym not in symbols_blob:
            # If an always-on symbol wasn't in briefing, create a minimal capsule
            capsule = {
                "symbol": sym,
                "premarket": {},
                "htf_levels": key_levels.get(sym, {}) if isinstance(key_levels, dict) else {},
                "notes": ["No premarket briefing entry found; minimal capsule."],
                "strategy_meta": {"threshold_confidence": 70, "rr_min": 1.0},
                "generated_at_utc": datetime.now(timezone.utc).isoformat()
            }
        else:
            capsule = build_capsule(sym, symbols_blob[sym], key_levels)

        save_json(caps_dir / f"{sym}.json", capsule)

    print(f"[PREMARKET] {day_str} | focus={selection['focus']}")
    print(f"  wrote: {paths['focus_list']}")
    print(f"  capsules -> {caps_dir}")

if __name__ == "__main__":
    main()
