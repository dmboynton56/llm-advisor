"""Ops-dashboard metrics computed from enriched trade / order-event / equity rows.

Pure functions over plain dicts so they can be unit-tested against fixtures and
fed from either Supabase or BigQuery (see scripts/compute_ops_metrics.py).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

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

_ET = ZoneInfo("America/New_York")
_GUARD_FAILURE_REASONS = {
    "duplicate_option_contract",
    "underlying_direction_exposure",
    "max_concurrent_trades",
    "stopout_cooldown",
}


def _f(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _event_details(event: Dict[str, Any]) -> Dict[str, Any]:
    details = event.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (TypeError, json.JSONDecodeError):
            details = {}
    return details if isinstance(details, dict) else {}


def _signal_uid(event: Dict[str, Any]) -> Optional[str]:
    details = _event_details(event)
    signal = event.get("signal")
    signal_uid = (
        event.get("signal_uid")
        or details.get("signal_uid")
        or (signal.get("signal_uid") if isinstance(signal, dict) else None)
    )
    value = str(signal_uid or "").strip()
    return value or None


def _event_datetime(event: Dict[str, Any]) -> Optional[datetime]:
    value = event.get("event_ts") or event.get("ts") or event.get("timestamp")
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_ET)


def _empty_signal_counts() -> Dict[str, int]:
    return {
        "detected": 0,
        "approved": 0,
        "rejected": 0,
        "capacity_blocked": 0,
        "capacity_expired": 0,
        "attempted": 0,
        "execution_succeeded": 0,
        "execution_guard_failed": 0,
        "execution_failed": 0,
        "approved_no_attempt": 0,
    }


def signal_level_funnel(order_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deduplicate signal lifecycle outcomes while preserving retry metrics separately."""
    records: Dict[str, Dict[str, Any]] = {}
    for event in order_events:
        uid = _signal_uid(event)
        if not uid:
            continue
        event_type = str(event.get("event_type") or "")
        details = _event_details(event)
        record = records.setdefault(
            uid,
            {
                "detected": False,
                "approved": False,
                "rejected": False,
                "capacity_blocked": False,
                "capacity_expired": False,
                "attempted": False,
                "execution_succeeded": False,
                "execution_guard_failed": False,
                "execution_failed": False,
                "terminal_outcome": None,
                "detected_at": None,
                "guard_reasons": set(),
            },
        )

        if event_type == "signal_detected":
            record["detected"] = True
            record["detected_at"] = _event_datetime(event) or record["detected_at"]
        elif event_type == "validation_approved":
            record["approved"] = True
        elif event_type == "validation_rejected":
            record["rejected"] = True
        elif event_type == "execution_attempt":
            record["attempted"] = True
        elif event_type == "execution_succeeded":
            record["execution_succeeded"] = True
        elif event_type == "max_concurrent_skipped":
            record["capacity_blocked"] = True
        elif event_type == "execution_timeout":
            if str(details.get("phase") or "") == "capacity":
                record["capacity_expired"] = True
            else:
                record["execution_failed"] = True
        elif event_type == "execution_failed":
            reason = str(details.get("reason") or details.get("error") or "")
            if reason in _GUARD_FAILURE_REASONS:
                record["guard_reasons"].add(reason)
                record["execution_guard_failed"] = True
            else:
                record["execution_failed"] = True
        elif event_type == "signal_outcome":
            outcome = str(details.get("outcome") or "").strip()
            if outcome and not record["terminal_outcome"]:
                record["terminal_outcome"] = outcome
            if details.get("detected_at") and record["detected_at"] is None:
                record["detected_at"] = _event_datetime(
                    {"timestamp": details.get("detected_at")}
                )
            if details.get("approved_at"):
                record["approved"] = True
            execution_attempts = _f(details.get("execution_attempts")) or 0.0
            if execution_attempts > 0:
                record["attempted"] = True
            capacity_skip_count = _f(details.get("capacity_skip_count")) or 0.0
            if capacity_skip_count > 0:
                record["capacity_blocked"] = True
            guard_failure_reasons = details.get("guard_failure_reasons") or []
            if isinstance(guard_failure_reasons, str):
                guard_failure_reasons = [guard_failure_reasons]
            for reason in guard_failure_reasons:
                record["guard_reasons"].add(str(reason))
            if outcome == "capacity_expired":
                record["capacity_expired"] = True
            elif outcome == "execution_guard_failed":
                record["execution_guard_failed"] = True
            elif outcome == "execution_failed":
                record["execution_failed"] = True
            elif outcome == "execution_succeeded":
                record["execution_succeeded"] = True
            elif outcome == "validation_rejected":
                record["rejected"] = True

    counts = _empty_signal_counts()
    session_periods = {
        "before_11": _empty_signal_counts(),
        "11_to_13": _empty_signal_counts(),
        "13_plus": _empty_signal_counts(),
    }
    terminal_outcomes: Dict[str, int] = {}

    for record in records.values():
        if record["approved"] and not record["attempted"]:
            record["approved_no_attempt"] = True
        terminal_outcome = str(record.get("terminal_outcome") or "")
        if terminal_outcome:
            record["execution_succeeded"] = terminal_outcome == "execution_succeeded"
            record["execution_guard_failed"] = terminal_outcome == "execution_guard_failed"
            record["execution_failed"] = terminal_outcome in {
                "execution_failed",
                "execution_expired",
            }
        elif record["execution_succeeded"]:
            # A signal that eventually succeeded is not a terminal failure even
            # when an earlier candidate or broker attempt failed.
            record["execution_guard_failed"] = False
            record["execution_failed"] = False
        else:
            record["execution_guard_failed"] = bool(record["guard_reasons"])
        if record["terminal_outcome"]:
            outcome = str(record["terminal_outcome"])
            terminal_outcomes[outcome] = terminal_outcomes.get(outcome, 0) + 1

        for key in counts:
            counts[key] += int(bool(record.get(key)))

        detected_at = record.get("detected_at")
        if detected_at is None:
            continue
        if (detected_at.hour, detected_at.minute) < (11, 0):
            bucket = session_periods["before_11"]
        elif (detected_at.hour, detected_at.minute) < (13, 0):
            bucket = session_periods["11_to_13"]
        else:
            bucket = session_periods["13_plus"]
        for key in bucket:
            bucket[key] += int(bool(record.get(key)))

    return {
        **counts,
        "signals_with_uid": len(records),
        "terminal_outcomes": dict(sorted(terminal_outcomes.items())),
        "session_periods": session_periods,
    }


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
        details = _event_details(event)
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
        "signal_funnel": signal_level_funnel(order_events),
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
