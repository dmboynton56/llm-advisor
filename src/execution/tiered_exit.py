"""Pure state and allocation helpers for paper-only tiered option exits.

The live tracker owns broker interaction; this module owns deterministic policy
state so replay and live execution can share the same transitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import floor
from typing import Any, Dict, Optional


TIERED_EXIT_POLICY_VERSION = "tiered_v1"


def _utc_iso(value: Optional[datetime] = None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def allocate_tier_quantities(
    initial_qty: int,
    tp1_fraction: float = 0.50,
    tp2_fraction: float = 0.25,
) -> tuple[int, int, int]:
    """Return TP1, TP2, runner quantities for an eligible position.

    TP1 uses ceil(50%), TP2 uses round-half-up(25%), and at least one contract
    remains for the runner. Positions below four contracts are intentionally
    rejected by the caller because the legacy policy handles them.
    """
    qty = int(initial_qty)
    if qty < 4:
        raise ValueError("tiered exits require at least four contracts")
    if not 0 < float(tp1_fraction) < 1 or not 0 < float(tp2_fraction) < 1:
        raise ValueError("tier fractions must be between zero and one")
    if float(tp1_fraction) + float(tp2_fraction) >= 1:
        raise ValueError("tier fractions must leave a runner")
    # ceil(qty * fraction), written without floating point surprises for the
    # default half allocation; the policy is intentionally contract based.
    tp1 = max(1, int(-floor(-qty * float(tp1_fraction))))
    remaining = qty - tp1
    # round-half-up for positive contract counts.
    tp2 = max(1, floor(qty * float(tp2_fraction) + 0.5))
    tp2 = min(tp2, max(1, remaining - 1))
    runner = qty - tp1 - tp2
    if runner < 1:
        raise ValueError("tier allocation must leave at least one runner contract")
    return tp1, tp2, runner


@dataclass
class TieredExitState:
    """Persisted lifecycle state for one net broker option position."""

    lifecycle_id: str
    underlying_symbol: str
    option_symbol: str
    initial_qty: int
    remaining_qty: int
    entry_price: float
    original_stop_price: float
    active_stop_price: float
    tp1_qty: int
    tp2_qty: int
    runner_qty: int
    stage: str = "pre_tp1"
    tp1_return_pct: float = 0.25
    tp2_return_pct: float = 0.50
    tp1_fraction: float = 0.50
    tp2_fraction: float = 0.25
    post_tp1_stop_return_pct: float = -0.05
    runner_floor_return_pct: float = 0.25
    runner_giveback_pct: float = 0.25
    peak_return_pct: float = 0.0
    mfe_return_pct: float = 0.0
    mae_return_pct: float = 0.0
    realized_pnl: float = 0.0
    realized_exit_qty: int = 0
    weighted_exit_value: float = 0.0
    exit_fills: list[Dict[str, Any]] = field(default_factory=list)
    pending_stage: Optional[str] = None
    pending_qty: int = 0
    pending_order_id: Optional[str] = None
    pending_client_order_id: Optional[str] = None
    pending_requested_at: Optional[str] = None
    pending_attempts: int = 0
    shadow_levels: Dict[str, Any] = field(default_factory=dict)
    shadow_crossings: Dict[str, Any] = field(default_factory=dict)
    last_event_at: Optional[str] = None
    fail_safe_reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        lifecycle_id: str,
        underlying_symbol: str,
        option_symbol: str,
        initial_qty: int,
        entry_price: float,
        original_stop_price: float,
        tp1_return_pct: float = 0.25,
        tp2_return_pct: float = 0.50,
        tp1_fraction: float = 0.50,
        tp2_fraction: float = 0.25,
        post_tp1_stop_return_pct: float = -0.05,
        runner_floor_return_pct: float = 0.25,
        runner_giveback_pct: float = 0.25,
        shadow_levels: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> "TieredExitState":
        tp1_qty, tp2_qty, runner_qty = allocate_tier_quantities(
            initial_qty, tp1_fraction=tp1_fraction, tp2_fraction=tp2_fraction
        )
        return cls(
            lifecycle_id=str(lifecycle_id),
            underlying_symbol=str(underlying_symbol or "").upper(),
            option_symbol=str(option_symbol or "").upper(),
            initial_qty=int(initial_qty),
            remaining_qty=int(initial_qty),
            entry_price=float(entry_price),
            original_stop_price=float(original_stop_price),
            active_stop_price=float(original_stop_price),
            tp1_qty=tp1_qty,
            tp2_qty=tp2_qty,
            runner_qty=runner_qty,
            tp1_return_pct=float(tp1_return_pct),
            tp2_return_pct=float(tp2_return_pct),
            tp1_fraction=float(tp1_fraction),
            tp2_fraction=float(tp2_fraction),
            post_tp1_stop_return_pct=float(post_tp1_stop_return_pct),
            runner_floor_return_pct=float(runner_floor_return_pct),
            runner_giveback_pct=float(runner_giveback_pct),
            shadow_levels=dict(shadow_levels or {}),
            last_event_at=_utc_iso(now),
        )

    @property
    def is_active(self) -> bool:
        return self.stage not in {"closed", "fail_safe"}

    @property
    def has_pending_exit(self) -> bool:
        return bool(self.pending_stage and self.pending_qty > 0)

    @property
    def quantity_integrity_ok(self) -> bool:
        return int(self.remaining_qty) + int(self.realized_exit_qty) == int(self.initial_qty)

    @property
    def partial_realized_pnl(self) -> float:
        """Realized P/L from confirmed partial fills in this lifecycle."""
        return float(self.realized_pnl)

    def current_return_pct(self, option_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return float(option_price) / self.entry_price - 1.0

    def next_profit_stage(self, return_pct: float) -> Optional[tuple[str, int]]:
        """Return at most one next profit action for the current mark."""
        if not self.is_active or self.has_pending_exit:
            return None
        value = float(return_pct)
        self.mfe_return_pct = max(self.mfe_return_pct, value)
        self.mae_return_pct = min(self.mae_return_pct, value)
        if self.stage == "pre_tp1" and value >= self.tp1_return_pct:
            return "tp1", self.tp1_qty
        if self.stage == "post_tp1" and value >= self.tp2_return_pct:
            return "tp2", self.tp2_qty
        if self.stage == "runner":
            self.peak_return_pct = max(self.peak_return_pct, value)
            trail = max(
                self.runner_floor_return_pct,
                self.peak_return_pct - self.runner_giveback_pct,
            )
            if value <= trail and self.remaining_qty > 0:
                return "runner_trail", self.remaining_qty
        return None

    def begin_pending(
        self,
        *,
        stage: str,
        qty: int,
        client_order_id: str,
        now: Optional[datetime] = None,
    ) -> None:
        if self.has_pending_exit:
            raise ValueError("a tiered exit is already pending")
        qty = int(qty)
        if qty <= 0 or qty > self.remaining_qty:
            raise ValueError("pending exit quantity exceeds remaining position")
        self.pending_stage = str(stage)
        self.pending_qty = qty
        self.pending_client_order_id = str(client_order_id)
        self.pending_order_id = None
        self.pending_requested_at = _utc_iso(now)
        self.pending_attempts += 1
        self.last_event_at = _utc_iso(now)

    def attach_pending_order(self, order_id: str, now: Optional[datetime] = None) -> None:
        if not self.has_pending_exit:
            raise ValueError("cannot attach an order without a pending exit")
        self.pending_order_id = str(order_id)
        self.last_event_at = _utc_iso(now)

    def clear_pending(self, now: Optional[datetime] = None) -> None:
        self.pending_stage = None
        self.pending_qty = 0
        self.pending_order_id = None
        self.pending_client_order_id = None
        self.pending_requested_at = None
        self.last_event_at = _utc_iso(now)

    def apply_fill(
        self,
        *,
        stage: str,
        qty: int,
        exit_price: float,
        multiplier: float = 100.0,
        pending_remaining_qty: int = 0,
        now: Optional[datetime] = None,
    ) -> None:
        qty = int(qty)
        if qty <= 0 or qty > self.remaining_qty:
            raise ValueError("filled quantity exceeds remaining position")
        pnl = (float(exit_price) - self.entry_price) * qty * float(multiplier)
        self.realized_pnl += pnl
        self.realized_exit_qty += qty
        self.weighted_exit_value += float(exit_price) * qty
        self.remaining_qty -= qty
        self.exit_fills.append(
            {
                "kind": "partial_exit",
                "stage": str(stage),
                "timestamp": _utc_iso(now),
                "qty": qty,
                "price": float(exit_price),
                "pnl": pnl,
                "remaining_qty": self.remaining_qty,
            }
        )
        pending_stage = self.pending_stage
        pending_order_id = self.pending_order_id
        pending_client_order_id = self.pending_client_order_id
        self.clear_pending(now)
        pending_remaining_qty = int(pending_remaining_qty)
        if pending_remaining_qty > 0:
            # Keep the lifecycle blocked while a broker order is partially
            # filled.  The tracker cancels the remainder at the timeout and
            # retries the exact missing quantity once.
            self.pending_stage = str(pending_stage or stage)
            self.pending_qty = pending_remaining_qty
            self.pending_order_id = pending_order_id
            self.pending_client_order_id = pending_client_order_id
            self.pending_requested_at = _utc_iso(now)
        if self.remaining_qty <= 0:
            self.remaining_qty = 0
            self.stage = "closed"
        elif pending_remaining_qty > 0:
            # Do not advance a tier until its requested quantity is confirmed.
            pass
        elif stage == "tp1":
            self.stage = "post_tp1"
            self.active_stop_price = max(
                self.original_stop_price,
                self.entry_price * (1.0 + self.post_tp1_stop_return_pct),
            )
        elif stage == "tp2":
            self.stage = "runner"
            self.peak_return_pct = max(
                self.peak_return_pct,
                self.tp2_return_pct,
                self.current_return_pct(float(exit_price)),
            )
            self.active_stop_price = max(
                self.active_stop_price,
                self.entry_price * (1.0 + self.runner_floor_return_pct),
            )
        elif stage == "runner_trail":
            self.stage = "closed"
        self.last_event_at = _utc_iso(now)

    def mark_fail_safe(self, reason: str, now: Optional[datetime] = None) -> None:
        self.fail_safe_reason = str(reason)
        self.stage = "fail_safe"
        self.clear_pending(now)
        self.last_event_at = _utc_iso(now)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "policy_version": TIERED_EXIT_POLICY_VERSION,
            "lifecycle_id": self.lifecycle_id,
            "trade_id": self.lifecycle_id,
            "underlying_symbol": self.underlying_symbol,
            "option_symbol": self.option_symbol,
            "initial_qty": self.initial_qty,
            "remaining_qty": self.remaining_qty,
            "entry_price": self.entry_price,
            "original_stop_price": self.original_stop_price,
            "active_stop_price": self.active_stop_price,
            "tp1_qty": self.tp1_qty,
            "tp2_qty": self.tp2_qty,
            "runner_qty": self.runner_qty,
            "stage": self.stage,
            "tp1_return_pct": self.tp1_return_pct,
            "tp2_return_pct": self.tp2_return_pct,
            "tp1_fraction": self.tp1_fraction,
            "tp2_fraction": self.tp2_fraction,
            "post_tp1_stop_return_pct": self.post_tp1_stop_return_pct,
            "runner_floor_return_pct": self.runner_floor_return_pct,
            "runner_giveback_pct": self.runner_giveback_pct,
            "peak_return_pct": self.peak_return_pct,
            "runner_high_water_return_pct": self.peak_return_pct,
            "mfe_return_pct": self.mfe_return_pct,
            "mae_return_pct": self.mae_return_pct,
            "realized_pnl": self.realized_pnl,
            "partial_realized_pnl": self.realized_pnl,
            "realized_exit_qty": self.realized_exit_qty,
            "weighted_exit_value": self.weighted_exit_value,
            "exit_fills": self.exit_fills,
            "pending_stage": self.pending_stage,
            "pending_qty": self.pending_qty,
            "pending_order_id": self.pending_order_id,
            "pending_client_order_id": self.pending_client_order_id,
            "pending_requested_at": self.pending_requested_at,
            "pending_attempts": self.pending_attempts,
            "shadow_levels": self.shadow_levels,
            "shadow_crossings": self.shadow_crossings,
            "last_event_at": self.last_event_at,
            "fail_safe_reason": self.fail_safe_reason,
        }
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> Optional["TieredExitState"]:
        if not isinstance(value, dict):
            return None
        if value.get("policy_version") != TIERED_EXIT_POLICY_VERSION:
            return None
        normalized = dict(value)
        if "lifecycle_id" not in normalized and "trade_id" in normalized:
            normalized["lifecycle_id"] = normalized["trade_id"]
        if "realized_pnl" not in normalized and "partial_realized_pnl" in normalized:
            normalized["realized_pnl"] = normalized["partial_realized_pnl"]
        if "peak_return_pct" not in normalized and "runner_high_water_return_pct" in normalized:
            normalized["peak_return_pct"] = normalized["runner_high_water_return_pct"]
        fields = {
            key: normalized[key]
            for key in cls.__dataclass_fields__
            if key in normalized
        }
        try:
            return cls(**fields)
        except (TypeError, ValueError):
            return None


def parse_tiered_state(value: Any) -> Optional[TieredExitState]:
    """Accept a state object, JSON string, or missing value."""
    if isinstance(value, TieredExitState):
        return value
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return TieredExitState.from_dict(value)
