"""Ops-dashboard metrics computed from enriched trade / order-event / equity rows.

Pure functions over plain dicts so they can be unit-tested against fixtures and
fed from either Supabase or BigQuery (see scripts/compute_ops_metrics.py).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

DTE_BUCKETS = ("0", "1-3", "4-7", "8-14", "15+")

# Order-event types that make up the execution funnel, in order.
FUNNEL_STAGES = (
    ("signals", ("signal_detected",)),
    ("validation_approved", ("validation_approved",)),
    ("execution_attempted", ("execution_attempt",)),
    ("executed", ("execution_succeeded",)),
)

REJECTION_EVENT_TYPES = (
    "validation_rejected",
    "validation_error",
    "execution_failed",
    "execution_timeout",
    "max_concurrent_skipped",
)


def _f(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def dte_bucket(dte: Any) -> Optional[str]:
    try:
        d = int(dte)
    except (TypeError, ValueError):
        return None
    if d <= 0:
        return "0"
    if d <= 3:
        return "1-3"
    if d <= 7:
        return "4-7"
    if d <= 14:
        return "8-14"
    return "15+"


def is_closed(trade: Dict[str, Any]) -> bool:
    return str(trade.get("status") or "").lower() == "closed" and trade.get("pnl") is not None


def _per_trade_rr(trade: Dict[str, Any]) -> Optional[float]:
    """Planned risk/reward from entry vs SL/TP where both are known."""
    entry = _f(trade.get("entry_price"))
    sl = _f(trade.get("stop_loss"))
    tp = _f(trade.get("take_profit"))
    if entry is None or sl is None or tp is None:
        return None
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    return reward / risk


def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Win rate / PnL / RR stats for one cell (overall or a breakdown slice)."""
    closed = [t for t in trades if is_closed(t)]
    pnls = [float(t["pnl"]) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    rr_values = [rr for rr in (_per_trade_rr(t) for t in closed) if rr is not None]

    return {
        "trades": len(trades),
        "closed_trades": len(closed),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "total_pnl": round(total_pnl, 2),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "average_win": round(avg_win, 2) if avg_win is not None else None,
        "average_loss": round(avg_loss, 2) if avg_loss is not None else None,
        "avg_realized_rr": (
            round(avg_win / abs(avg_loss), 3) if avg_win is not None and avg_loss else None
        ),
        "avg_planned_rr": round(sum(rr_values) / len(rr_values), 3) if rr_values else None,
        "profit_factor": (
            round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
        ),
    }


def _breakdown(
    trades: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], Optional[str]]
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades:
        key = key_fn(trade)
        if not key:
            key = "unknown"
        groups.setdefault(key, []).append(trade)
    return {key: summarize_trades(rows) for key, rows in sorted(groups.items())}


def _side_key(trade: Dict[str, Any]) -> Optional[str]:
    side = str(trade.get("side") or "").lower()
    if side in ("buy", "long"):
        return "long"
    if side in ("sell", "short"):
        return "short"
    return side or None


def max_drawdown(equity_series: List[float]) -> Optional[float]:
    """Largest peak-to-trough decline (absolute dollars) over the series."""
    if len(equity_series) < 2:
        return None
    peak = equity_series[0]
    worst = 0.0
    for value in equity_series[1:]:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return round(worst, 2)


def trades_per_day(trades: List[Dict[str, Any]]) -> Optional[float]:
    days = {str(t.get("run_date")) for t in trades if t.get("run_date")}
    if not days:
        return None
    return round(len(trades) / len(days), 2)


def biggest_losers(
    trades: List[Dict[str, Any]],
    order_events: List[Dict[str, Any]],
    n: int = 5,
) -> List[Dict[str, Any]]:
    """Bottom-N closed trades by PnL, with validation reasoning joined from order events."""
    closed = sorted((t for t in trades if is_closed(t)), key=lambda t: float(t["pnl"]))
    losers = [t for t in closed if float(t["pnl"]) < 0][:n]

    events_by_order: Dict[str, List[Dict[str, Any]]] = {}
    reasoning_by_correlation: Dict[tuple[str, str, str], str] = {}
    for event in order_events:
        order_id = str(event.get("order_id") or "").strip()
        if order_id:
            events_by_order.setdefault(order_id, []).append(event)
        details = event.get("details") or {}
        reasoning = details.get("reasoning") if isinstance(details, dict) else None
        run_date = str(event.get("run_date") or "").strip()
        symbol = str(event.get("symbol") or "").strip().upper()
        loop_count = event.get("loop_count")
        if (
            event.get("event_type") == "validation_approved"
            and reasoning
            and run_date
            and symbol
            and loop_count is not None
        ):
            reasoning_by_correlation.setdefault(
                (run_date, symbol, str(loop_count)),
                str(reasoning),
            )

    out: List[Dict[str, Any]] = []
    for trade in losers:
        order_id = str(trade.get("order_id") or "").strip()
        reasoning = None
        for event in events_by_order.get(order_id, []):
            details = event.get("details") or {}
            if isinstance(details, dict) and details.get("reasoning"):
                reasoning = str(details["reasoning"])
                break
        if reasoning is None:
            for event in events_by_order.get(order_id, []):
                run_date = str(event.get("run_date") or "").strip()
                symbol = str(event.get("symbol") or "").strip().upper()
                loop_count = event.get("loop_count")
                if not run_date or not symbol or loop_count is None:
                    continue
                reasoning = reasoning_by_correlation.get(
                    (run_date, symbol, str(loop_count))
                )
                if reasoning:
                    break
        out.append(
            {
                "trade_uid": trade.get("trade_uid"),
                "run_date": trade.get("run_date"),
                "symbol": trade.get("symbol"),
                "underlying_symbol": trade.get("underlying_symbol"),
                "side": _side_key(trade),
                "setup_type": trade.get("setup_type"),
                "option_dte": trade.get("option_dte"),
                "pnl": round(float(trade["pnl"]), 2),
                "exit_reason": trade.get("exit_reason"),
                "validation_reasoning": reasoning,
            }
        )
    return out


def execution_funnel(order_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for event in order_events:
        event_type = str(event.get("event_type") or "")
        counts[event_type] = counts.get(event_type, 0) + 1

    stages = {
        name: sum(counts.get(t, 0) for t in types) for name, types in FUNNEL_STAGES
    }

    rejections: Dict[str, int] = {}
    for event in order_events:
        event_type = str(event.get("event_type") or "")
        if event_type not in REJECTION_EVENT_TYPES:
            continue
        details = event.get("details") or {}
        reason = None
        if isinstance(details, dict):
            reason = details.get("reason") or details.get("error")
        key = f"{event_type}:{reason}" if reason else event_type
        rejections[key] = rejections.get(key, 0) + 1

    approved = stages["validation_approved"]
    rejected = counts.get("validation_rejected", 0)
    validated_total = approved + rejected

    return {
        "stages": stages,
        "rejection_reasons": dict(sorted(rejections.items(), key=lambda kv: -kv[1])),
        "llm_approval_rate": (approved / validated_total) if validated_total else None,
    }


def compute_ops_metrics(
    trades: List[Dict[str, Any]],
    order_events: List[Dict[str, Any]],
    account_snapshots: List[Dict[str, Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Full ops-metrics payload; shape is what llm_advisor_ops_metrics_daily.payload stores."""
    overall = summarize_trades(trades)

    equity_series = [
        e
        for e in (
            _f(s.get("equity"))
            for s in sorted(account_snapshots, key=lambda s: str(s.get("captured_at") or ""))
        )
        if e is not None
    ]

    overall["max_drawdown"] = max_drawdown(equity_series)
    overall["trades_per_day"] = trades_per_day(trades)

    return {
        "range": {"start": start_date, "end": end_date},
        "overall": overall,
        "breakdowns": {
            "by_underlying": _breakdown(trades, lambda t: t.get("underlying_symbol")),
            "by_side": _breakdown(trades, _side_key),
            "by_setup_type": _breakdown(trades, lambda t: t.get("setup_type")),
            "by_dte_bucket": _breakdown(trades, lambda t: dte_bucket(t.get("option_dte"))),
        },
        "biggest_losers": biggest_losers(trades, order_events),
        "funnel": execution_funnel(order_events),
        "equity": {
            "points": len(equity_series),
            "latest": equity_series[-1] if equity_series else None,
        },
    }
