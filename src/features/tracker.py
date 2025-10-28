#!/usr/bin/env python3
"""
Confluence tracker: merge new feature snapshots into per-symbol state and derive active confluences.

Statuses: active | filled | invalidated | expired
Stages: htf_stage ∈ {none, reversal_seen, confirmed}, ltf_stage ∈ {none, scanning, entry_window}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

STATUS_TTL_MIN = 90  # keep filled/invalidated items around for audit for this many minutes


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mk_id(kind: str, tf: str, time_str: str) -> str:
    return f"{kind}:{tf}:{time_str}"


def _short(rec: Dict[str, Any]) -> str:
    kind = rec.get("kind") or rec.get("last_bos") or "?"
    tf = rec.get("tf") or "?"
    direction = rec.get("direction")
    level = rec.get("level")
    low = rec.get("low")
    high = rec.get("high")
    if kind.startswith("BOS"):
        dir_tag = "bull" if direction and direction.startswith("bull") else "bear"
        return f"{kind}@{tf}:{dir_tag}"
    if kind in {"FVG", "IFVG"}:
        dir_tag = "bull" if direction and direction.startswith("bull") else "bear"
        if low is not None and high is not None:
            return f"{kind}@{tf}:{dir_tag}[{low:.2f}-{high:.2f}]"
        return f"{kind}@{tf}:{dir_tag}"
    if kind == "OB" or kind == "BB":
        dir_tag = "bull" if direction and direction.startswith("bull") else "bear"
        if low is not None and high is not None:
            return f"{kind}@{tf}:{dir_tag}[{low:.2f}-{high:.2f}]"
        if level is not None:
            return f"{kind}@{tf}:{dir_tag}[{level:.2f}]"
        return f"{kind}@{tf}:{dir_tag}"
    return f"{kind}@{tf}"


def _statusize(kind: str, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in recs or []:
        rr = dict(r)
        rr["kind"] = kind
        key_time = r.get("time") or r.get("start") or r.get("retested_at") or _now_iso()
        rr["id"] = _mk_id(kind, r.get("tf", "?"), key_time)
        status = "active"
        if kind == "FVG":
            if r.get("invalidated"):
                status = "invalidated"
            elif r.get("filled"):
                status = "filled"
            elif r.get("active") is False:
                status = "invalidated"
        elif kind == "OB":
            if r.get("invalidated"):
                status = "invalidated"
            elif r.get("active") is False:
                status = "invalidated"
        elif kind in {"IFVG", "BB"}:
            if r.get("active") is False:
                status = "invalidated"
        rr["status"] = status
        # Normalize direction hints
        dir_hint = r.get("direction")
        if dir_hint:
            if dir_hint.lower().startswith("bull"):
                rr["direction"] = "bullish"
            elif dir_hint.lower().startswith("bear"):
                rr["direction"] = "bearish"
            else:
                rr["direction"] = dir_hint
        out.append(rr)
    return out


def _bos_record(tf_blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    ms = tf_blob.get("ms") or {}
    last_bos = ms.get("last_bos")
    if not last_bos:
        return []
    direction = "bullish" if last_bos == "BOS_up" else "bearish" if last_bos == "BOS_down" else None
    return [{
        "kind": last_bos,
        "tf": tf_blob.get("tf"),
        "time": ms.get("last_bos_time") or _now_iso(),
        "level": ms.get("last_bos_level"),
        "status": "active",
        "direction": direction,
        "id": _mk_id(last_bos, tf_blob.get("tf", "?"), ms.get("last_bos_time") or _now_iso()),
    }]


def _merge_list(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    by_id = {r.get("id"): dict(r) for r in (existing or []) if r.get("id")}
    for r in incoming or []:
        rid = r.get("id")
        if not rid:
            continue
        prev = by_id.get(rid, {})
        merged = {**prev, **r}
        merged.setdefault("first_seen_utc", prev.get("first_seen_utc", _now_iso()))
        merged["last_seen_utc"] = _now_iso()
        by_id[rid] = merged
    pruned: List[Dict[str, Any]] = []
    for rec in by_id.values():
        status = rec.get("status")
        if status in ("active", "forming"):
            pruned.append(rec)
            continue
        try:
            last_seen = datetime.fromisoformat(rec.get("last_seen_utc"))
        except Exception:
            last_seen = now
        if (now - last_seen) <= timedelta(minutes=STATUS_TTL_MIN):
            pruned.append(rec)
    pruned.sort(key=lambda x: x.get("last_seen_utc", ""))
    return pruned


def _active_structures(merged: List[Dict[str, Any]], trend: Optional[str], tf: str) -> List[Dict[str, Any]]:
    candidates = {"FVG", "IFVG", "OB", "BB"}
    filtered: List[Dict[str, Any]] = []
    for c in merged:
        if c.get("tf") != tf or c.get("status") != "active" or c.get("kind") not in candidates:
            continue
        dir_hint = c.get("direction")
        if trend is None or dir_hint is None or dir_hint == trend:
            filtered.append(c)
    return filtered


def update_symbol_state(existing: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(existing.get("state", {}))

    incoming: List[Dict[str, Any]] = []
    tf5 = features.get("tf_5m", {})
    tf1 = features.get("tf_1m", {})

    incoming += _statusize("FVG", tf5.get("fvg", []))
    incoming += _statusize("IFVG", tf5.get("ifvg", []))
    incoming += _statusize("OB", tf5.get("ob", []))
    incoming += _statusize("BB", tf5.get("bb", []))
    incoming += _bos_record(tf5)

    incoming += _statusize("FVG", tf1.get("fvg", []))
    incoming += _statusize("IFVG", tf1.get("ifvg", []))
    incoming += _statusize("OB", tf1.get("ob", []))
    incoming += _statusize("BB", tf1.get("bb", []))

    merged = _merge_list(existing.get("confluences", []), incoming)

    ms5 = tf5.get("ms", {})
    bos_kind = ms5.get("last_bos")
    trend = ms5.get("trend") or state.get("trend")

    # Stage logic using HTF structures + BOS
    htf_stage = "none"
    ltf_stage = "none"

    if bos_kind:
        htf_structs = _active_structures(merged, trend, "5m")
        if htf_structs:
            htf_stage = "confirmed"
        else:
            htf_stage = "reversal_seen"

    if htf_stage == "confirmed":
        ltf_structs = _active_structures(merged, trend, "1m")
        ltf_stage = "entry_window" if ltf_structs else "scanning"

    active = [c for c in merged if c.get("status") == "active"]
    active_short = [_short(c) for c in active]

    new_state = {
        **state,
        "htf_stage": htf_stage,
        "ltf_stage": ltf_stage,
        "trend": trend,
        "last_bos": bos_kind,
        "last_update_utc": _now_iso(),
        "confluences": merged,
    }

    return {
        **existing,
        "state": new_state,
        "active_confluences": active_short,
    }


