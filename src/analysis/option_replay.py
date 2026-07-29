"""Deterministic replay of option exits from recorded one-minute position marks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class OptionMark:
    timestamp: datetime
    option_price: float
    underlying_price: Optional[float] = None


@dataclass(frozen=True)
class ReplayPolicy:
    name: str
    profit_target_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    underlying_stop: Optional[float] = None
    underlying_target: Optional[float] = None
    underlying_side: str = "long"
    max_hold_minutes: Optional[int] = None


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    exit_reason: str
    exit_time: datetime
    exit_option_price: float
    pnl: float
    return_pct: float
    hold_minutes: float


def replay_exit(
    marks: Iterable[OptionMark],
    *,
    entry_price: float,
    qty: float,
    policy: ReplayPolicy,
    multiplier: float = 100.0,
) -> ReplayResult:
    ordered = sorted(marks, key=lambda mark: mark.timestamp)
    if not ordered:
        raise ValueError("Replay requires at least one mark")
    if entry_price <= 0 or qty <= 0:
        raise ValueError("entry_price and qty must be positive")

    opened_at = ordered[0].timestamp
    selected = ordered[-1]
    reason = "end_of_data"
    for mark in ordered:
        option_return = mark.option_price / entry_price - 1.0
        if (
            policy.stop_loss_pct is not None
            and option_return <= -abs(policy.stop_loss_pct)
        ):
            selected, reason = mark, "premium_stop"
            break
        if (
            policy.profit_target_pct is not None
            and option_return >= abs(policy.profit_target_pct)
        ):
            selected, reason = mark, "premium_target"
            break
        if mark.underlying_price is not None:
            if policy.underlying_side == "short":
                stop_hit = (
                    policy.underlying_stop is not None
                    and mark.underlying_price >= policy.underlying_stop
                )
                target_hit = (
                    policy.underlying_target is not None
                    and mark.underlying_price <= policy.underlying_target
                )
            else:
                stop_hit = (
                    policy.underlying_stop is not None
                    and mark.underlying_price <= policy.underlying_stop
                )
                target_hit = (
                    policy.underlying_target is not None
                    and mark.underlying_price >= policy.underlying_target
                )
            if stop_hit:
                selected, reason = mark, "underlying_stop"
                break
            if target_hit:
                selected, reason = mark, "underlying_target"
                break
        if policy.max_hold_minutes is not None:
            held = (mark.timestamp - opened_at).total_seconds() / 60.0
            if held >= policy.max_hold_minutes:
                selected, reason = mark, "time_stop"
                break

    return_pct = selected.option_price / entry_price - 1.0
    return ReplayResult(
        policy=policy.name,
        exit_reason=reason,
        exit_time=selected.timestamp,
        exit_option_price=selected.option_price,
        pnl=(selected.option_price - entry_price) * qty * multiplier,
        return_pct=return_pct,
        hold_minutes=(selected.timestamp - opened_at).total_seconds() / 60.0,
    )


def load_marks_from_order_events(
    path: Path,
    *,
    option_symbol: str,
    entry_order_id: Optional[str] = None,
) -> list[OptionMark]:
    marks: list[OptionMark] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "option_position_mark":
                continue
            if str(event.get("symbol", "")).upper() != option_symbol.upper():
                continue
            details = event.get("details") or {}
            if entry_order_id and str(details.get("entry_order_id") or "") != entry_order_id:
                continue
            option_price = _float_or_none(details.get("current_price"))
            timestamp = _timestamp(event.get("ts"))
            if option_price is None or timestamp is None:
                continue
            marks.append(
                OptionMark(
                    timestamp=timestamp,
                    option_price=option_price,
                    underlying_price=_float_or_none(details.get("underlying_price")),
                )
            )
    return sorted(marks, key=lambda mark: mark.timestamp)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
