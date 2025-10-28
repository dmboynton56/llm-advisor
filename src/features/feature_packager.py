#!/usr/bin/env python3
"""
Feature packager — runs ICT detectors per symbol & timeframe and returns a compact dict.

It is stateless; persistence is handled by the tracker and live loop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ict_detectors import compute_features_for_tf, compute_smt_for_pair


def pack_features_for_symbol(symbol: str, bars_for_sym: dict, ref_pairs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    bars_for_sym: {"bars_1m": {SYM: [...]}, "bars_5m": {SYM: [...]}}
    ref_pairs: optional mapping like {"QQQ": "SPY"} to compute SMT
    """
    one = (bars_for_sym.get("bars_1m", {}) or {}).get(symbol, [])
    five = (bars_for_sym.get("bars_5m", {}) or {}).get(symbol, [])
    tf_1m = compute_features_for_tf(one, "1m")
    tf_5m = compute_features_for_tf(five, "5m")

    smt: List[Dict[str, Any]] = []
    if ref_pairs and symbol in ref_pairs:
        ref = ref_pairs[symbol]
        ref_one = (bars_for_sym.get("bars_1m", {}) or {}).get(ref, [])
        if ref_one:
            smt = compute_smt_for_pair(one, ref_one)

    # No levels yet; placeholder for EQ/OTE, OR high/low etc. We can add later.
    return {
        "symbol": symbol,
        "tf_1m": tf_1m,
        "tf_5m": tf_5m,
        "smt": smt,
        "levels": {},
    }


