"""Deterministic replay of option exits from recorded one-minute position marks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from src.execution.tiered_exit import TieredExitState


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
    tiered: bool = False
    tp1_return_pct: float = 0.25
    tp2_return_pct: float = 0.50
    tp1_fraction: float = 0.50
    tp2_fraction: float = 0.25
    post_tp1_stop_return_pct: float = -0.05
    runner_floor_return_pct: float = 0.25
    runner_giveback_pct: float = 0.25
    shadow_levels: Optional[dict[str, float]] = None


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    exit_reason: str
    exit_time: datetime
    exit_option_price: float
    pnl: float
    return_pct: float
    hold_minutes: float
    fills: tuple[dict[str, Any], ...] = ()
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    runner_contribution: float = 0.0
    legacy_pnl: Optional[float] = None
    max_drawdown_pct: float = 0.0
    shadow_crossings: tuple[dict[str, Any], ...] = ()


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

    if policy.tiered and int(qty) >= 4:
        return _replay_tiered(
            ordered,
            entry_price=float(entry_price),
            qty=int(qty),
            policy=policy,
            multiplier=multiplier,
        )

    opened_at = ordered[0].timestamp
    selected = ordered[-1]
    reason = "end_of_data"
    running_peak = float("-inf")
    max_drawdown = 0.0
    mfe = float("-inf")
    mae = float("inf")
    for mark in ordered:
        option_return = mark.option_price / entry_price - 1.0
        mfe = max(mfe, option_return)
        mae = min(mae, option_return)
        running_peak = max(running_peak, option_return)
        max_drawdown = max(max_drawdown, running_peak - option_return)
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
        legacy_pnl=(selected.option_price - entry_price) * qty * multiplier,
        mfe_pct=0.0 if mfe == float("-inf") else mfe,
        mae_pct=0.0 if mae == float("inf") else mae,
        max_drawdown_pct=max_drawdown,
    )


def _replay_tiered(
    ordered: list[OptionMark],
    *,
    entry_price: float,
    qty: int,
    policy: ReplayPolicy,
    multiplier: float,
) -> ReplayResult:
    opened_at = ordered[0].timestamp
    state = TieredExitState.create(
        lifecycle_id="replay",
        underlying_symbol="",
        option_symbol="",
        initial_qty=qty,
        entry_price=entry_price,
        original_stop_price=entry_price * (1.0 - abs(float(policy.stop_loss_pct or 0.35))),
        tp1_return_pct=policy.tp1_return_pct,
        tp2_return_pct=policy.tp2_return_pct,
        tp1_fraction=policy.tp1_fraction,
        tp2_fraction=policy.tp2_fraction,
        post_tp1_stop_return_pct=policy.post_tp1_stop_return_pct,
        runner_floor_return_pct=policy.runner_floor_return_pct,
        runner_giveback_pct=policy.runner_giveback_pct,
        shadow_levels=policy.shadow_levels,
    )
    fills: list[dict[str, Any]] = []
    mfe = float("-inf")
    mae = float("inf")
    runner_pnl = 0.0
    max_drawdown = 0.0
    running_peak = float("-inf")
    shadow_crossings: list[dict[str, Any]] = []
    previous_underlying: Optional[float] = None
    selected = ordered[-1]
    reason = "end_of_data"

    for mark in ordered:
        selected = mark
        option_return = mark.option_price / entry_price - 1.0
        mfe = max(mfe, option_return)
        mae = min(mae, option_return)
        running_peak = max(running_peak, option_return)
        max_drawdown = max(max_drawdown, running_peak - option_return)
        if mark.underlying_price is not None:
            for name, raw_level in (policy.shadow_levels or {}).items():
                level = _float_or_none(raw_level)
                if level is None or previous_underlying is None or previous_underlying == mark.underlying_price:
                    continue
                if (previous_underlying - level) * (mark.underlying_price - level) <= 0:
                    if not any(item["level_name"] == name for item in shadow_crossings):
                        shadow_crossings.append(
                            {
                                "level_name": name,
                                "level": level,
                                "underlying_price": mark.underlying_price,
                                "option_return_pct": option_return,
                                "timestamp": mark.timestamp,
                            }
                        )
            previous_underlying = float(mark.underlying_price)
        action = state.next_profit_stage(option_return)
        if action is not None and action[0] == "runner_trail":
            stage, qty_to_exit = action
            state.apply_fill(
                stage=stage,
                qty=qty_to_exit,
                exit_price=mark.option_price,
                multiplier=multiplier,
                now=mark.timestamp,
            )
            fill_pnl = (mark.option_price - entry_price) * qty_to_exit * multiplier
            if stage == "runner_trail":
                runner_pnl += fill_pnl
            fills.append({"stage": stage, "qty": qty_to_exit, "price": mark.option_price, "pnl": fill_pnl})
            if state.stage == "closed":
                reason = "runner_trail"
                break
            continue
        stop_return = state.current_return_pct(state.active_stop_price)
        if option_return <= stop_return:
            stage = "tiered_stop"
            qty_to_exit = state.remaining_qty
            state.apply_fill(
                stage=stage,
                qty=qty_to_exit,
                exit_price=mark.option_price,
                multiplier=multiplier,
                now=mark.timestamp,
            )
            fills.append({"stage": stage, "qty": qty_to_exit, "price": mark.option_price})
            reason = "tiered_stop"
            break
        if (
            policy.max_hold_minutes is not None
            and (mark.timestamp - opened_at).total_seconds() / 60.0 >= policy.max_hold_minutes
        ):
            qty_to_exit = state.remaining_qty
            state.apply_fill(
                stage="time_stop",
                qty=qty_to_exit,
                exit_price=mark.option_price,
                multiplier=multiplier,
                now=mark.timestamp,
            )
            fills.append({"stage": "time_stop", "qty": qty_to_exit, "price": mark.option_price})
            reason = "time_stop"
            break
        if action is not None:
            stage, qty_to_exit = action
            state.apply_fill(
                stage=stage,
                qty=qty_to_exit,
                exit_price=mark.option_price,
                multiplier=multiplier,
                now=mark.timestamp,
            )
            fill_pnl = (mark.option_price - entry_price) * qty_to_exit * multiplier
            fills.append({"stage": stage, "qty": qty_to_exit, "price": mark.option_price, "pnl": fill_pnl})

    if state.remaining_qty > 0:
        qty_to_exit = state.remaining_qty
        state.apply_fill(
            stage="end_of_data",
            qty=qty_to_exit,
            exit_price=selected.option_price,
            multiplier=multiplier,
            now=selected.timestamp,
        )
        fills.append({"stage": "end_of_data", "qty": qty_to_exit, "price": selected.option_price})

    legacy_policy = ReplayPolicy(
        name=f"{policy.name}:legacy_counterfactual",
        profit_target_pct=policy.profit_target_pct if policy.profit_target_pct is not None else 0.25,
        stop_loss_pct=abs(float(policy.stop_loss_pct or 0.35)),
        underlying_stop=policy.underlying_stop,
        underlying_target=policy.underlying_target,
        underlying_side=policy.underlying_side,
        max_hold_minutes=policy.max_hold_minutes,
    )
    legacy = replay_exit(
        ordered,
        entry_price=entry_price,
        qty=qty,
        policy=legacy_policy,
        multiplier=multiplier,
    )
    exit_qty = max(1, state.realized_exit_qty)
    weighted_exit_price = state.weighted_exit_value / exit_qty
    return ReplayResult(
        policy=policy.name,
        exit_reason=reason,
        exit_time=selected.timestamp,
        exit_option_price=weighted_exit_price,
        pnl=state.realized_pnl,
        return_pct=weighted_exit_price / entry_price - 1.0,
        hold_minutes=(selected.timestamp - opened_at).total_seconds() / 60.0,
        fills=tuple(fills),
        mfe_pct=0.0 if mfe == float("-inf") else mfe,
        mae_pct=0.0 if mae == float("inf") else mae,
        runner_contribution=runner_pnl,
        legacy_pnl=legacy.pnl,
        max_drawdown_pct=max_drawdown,
        shadow_crossings=tuple(shadow_crossings),
    )


def evaluate_strategy_gate(
    tiered_results: Iterable[ReplayResult],
    legacy_results: Iterable[ReplayResult],
) -> dict[str, Any]:
    """Apply the paper-trial go/no-go gate without changing position sizing."""
    tiered = list(tiered_results)
    legacy = list(legacy_results)
    if len(tiered) != len(legacy) or not tiered:
        return {
            "passed": False,
            "reason": "mismatched_or_empty_lifecycles",
            "lifecycle_count": min(len(tiered), len(legacy)),
        }
    tiered_pnl = sum(float(row.pnl) for row in tiered)
    legacy_pnl = sum(float(row.pnl) for row in legacy)
    tiered_winners = [float(row.pnl) for row in tiered if row.pnl > 0]
    legacy_winners = [float(row.pnl) for row in legacy if row.pnl > 0]
    avg_tiered_winner = sum(tiered_winners) / len(tiered_winners) if tiered_winners else 0.0
    avg_legacy_winner = sum(legacy_winners) / len(legacy_winners) if legacy_winners else 0.0
    worst_tiered = min(float(row.return_pct) for row in tiered)
    worst_legacy = min(float(row.return_pct) for row in legacy)
    return {
        "passed": (
            tiered_pnl - legacy_pnl >= 0
            and avg_tiered_winner >= avg_legacy_winner
            and worst_tiered - worst_legacy <= 0.05
        ),
        "lifecycle_count": len(tiered),
        "cumulative_pnl_delta": tiered_pnl - legacy_pnl,
        "average_winner_delta": avg_tiered_winner - avg_legacy_winner,
        "worst_lifecycle_return_delta": worst_tiered - worst_legacy,
    }


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
