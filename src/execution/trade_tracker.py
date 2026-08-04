"""Track open positions and monitor for exits."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import date, datetime, timezone

from src.core.config import OptionsSettings
from src.execution.order_manager import StockOrderManager
from src.execution.tiered_exit import TieredExitState, parse_tiered_state

if TYPE_CHECKING:
    from src.data.storage import StorageAdapter

logger = logging.getLogger(__name__)

_OCC_OPTION_RE = re.compile(r"^[A-Z0-9.]{1,10}(?P<expiry>\d{6})[CP]\d{8}$")


def option_expiration_date(symbol: str) -> Optional[date]:
    """Extract the YYMMDD expiration encoded in an OCC option symbol."""
    match = _OCC_OPTION_RE.fullmatch(str(symbol).upper().replace(" ", ""))
    if not match:
        return None
    try:
        return datetime.strptime(match.group("expiry"), "%y%m%d").date()
    except ValueError:
        return None


def option_dte(symbol: str, on_date: date) -> Optional[int]:
    expiry = option_expiration_date(symbol)
    return (expiry - on_date).days if expiry else None


class TradeTracker:
    """Tracks open positions and monitors for exits."""

    def __init__(
        self,
        order_manager: StockOrderManager,
        storage: Optional["StorageAdapter"] = None,
        options_settings: Optional[OptionsSettings] = None,
    ):
        self.order_manager = order_manager
        self.storage = storage
        self.options_settings = options_settings
        self.tracked_positions: Dict[str, Dict[str, Any]] = {}
        self._order_meta: Dict[str, Dict[str, Any]] = {}
        self._closing_symbols: set[str] = set()
        self._closing_context: Dict[str, Dict[str, Any]] = {}
        self._exit_events: List[Dict[str, Any]] = []
        self.session_closed: List[Dict[str, Any]] = []
        self._underlying_marks: Dict[str, float] = {}

    def register_open_trade(
        self,
        symbol: str,
        order_id: Optional[str],
        trade_pk: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Link Alpaca order id + internal trades PK for DB updates on exit."""
        normalized_symbol = str(symbol).upper()
        meta = {
            "order_id": str(order_id) if order_id else "",
            "trade_pk": int(trade_pk),
            "opened_at": datetime.now(timezone.utc),
            **(metadata or {}),
        }
        raw_state = meta.get("exit_state") or meta.get("tiered_exit_state")
        restored = parse_tiered_state(raw_state)
        if restored is not None:
            meta["tiered_exit_state"] = restored
        elif raw_state is not None:
            meta["tiered_exit_unreadable"] = True
        self._order_meta[normalized_symbol] = meta

    def set_underlying_mark(self, symbol: str, price: float) -> None:
        self._underlying_marks[str(symbol).upper()] = float(price)

    def update_positions(
        self,
        now: Optional[datetime] = None,
        force_close_options: bool = False,
        force_close_reason: str = "option_forced_exit",
    ) -> List[Dict[str, Any]]:
        """
        Update and return current open positions.

        Returns:
            List of position dicts
        """
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        positions = self.order_manager.get_open_positions()
        fetched_symbols = {str(pos.get("symbol", "")).upper() for pos in positions}
        self._closing_symbols.intersection_update(fetched_symbols)

        if self.options_settings and hasattr(self.order_manager, "ensure_protective_stops"):
            try:
                self.order_manager.ensure_protective_stops(
                    positions=positions,
                    order_meta=self._order_meta,
                )
                if hasattr(self.order_manager, "pop_risk_events"):
                    risk_events = self.order_manager.pop_risk_events()
                    self._exit_events.extend(risk_events)
                    self._persist_actual_entry_fills(risk_events)
                    self._close_unprotected_positions(risk_events, now)
            except Exception as exc:
                logger.error("Failed to ensure broker option protection: %s", exc)

        for pos in positions:
            symbol = str(pos["symbol"]).upper()
            previous_pos = self.tracked_positions.get(symbol)
            self.tracked_positions[symbol] = {
                **pos,
                "last_updated": datetime.now().isoformat(),
            }
            if self.options_settings and self._is_option_position(pos):
                self._reconcile_tiered_pending_fill(symbol, previous_pos, pos, now)
                meta = self._order_meta.get(symbol, {})
                underlying = str(meta.get("underlying_symbol") or "").upper()
                self._exit_events.append(
                    {
                        "event_type": "option_position_mark",
                        "symbol": symbol,
                        "details": {
                            "entry_order_id": meta.get("order_id"),
                            "current_price": self._float_or_none(pos.get("current_price")),
                            "entry_price": self._float_or_none(pos.get("entry_price")),
                            "qty": self._float_or_none(pos.get("qty")),
                            "unrealized_pl": self._float_or_none(pos.get("unrealized_pl")),
                            "unrealized_plpc": self._normalized_pct(
                                pos.get("unrealized_plpc")
                            ),
                            "underlying_symbol": meta.get("underlying_symbol"),
                            "underlying_price": self._underlying_marks.get(underlying),
                            "underlying_trade_plan": meta.get("option_plan"),
                            "tiered_exit_state": self._tier_state_dict(meta),
                        },
                    }
                )

        self._manage_tiered_positions(positions, now, force_close_options=force_close_options)

        positions = self._close_option_positions_if_needed(
            positions,
            now=now,
            force_close_options=force_close_options,
            force_close_reason=force_close_reason,
        )

        open_symbols = {str(pos["symbol"]).upper() for pos in positions}
        closed_symbols = set(self.tracked_positions.keys()) - open_symbols

        for symbol in closed_symbols:
            old_pos = self.tracked_positions.pop(symbol)
            meta = self._order_meta.pop(symbol, {})
            context = self._closing_context.pop(symbol, {})
            fill = self._latest_exit_fill(symbol)
            exit_px = self._float_or_none((fill or {}).get("filled_avg_price"))
            exit_qty = self._float_or_none((fill or {}).get("filled_qty"))
            pnl = self._realized_pnl_from_fill(old_pos, exit_px, exit_qty)
            if pnl is None:
                pnl = float(old_pos.get("unrealized_pl", 0) or 0)
            tier_state = parse_tiered_state(meta.get("tiered_exit_state"))
            if tier_state is not None and exit_qty is None:
                exit_qty = self._float_or_none(old_pos.get("qty"))
            total_pnl = float(pnl)
            total_exit_qty = exit_qty
            total_exit_price = exit_px
            partial_realized_pnl = 0.0
            if tier_state is not None:
                partial_realized_pnl = float(tier_state.realized_pnl)
                tier_state.realized_pnl += float(pnl)
                if exit_qty:
                    tier_state.realized_exit_qty += int(exit_qty)
                    tier_state.weighted_exit_value += float(exit_px or 0.0) * int(exit_qty)
                total_pnl = float(tier_state.realized_pnl)
                total_exit_qty = tier_state.realized_exit_qty
                if total_exit_qty:
                    total_exit_price = tier_state.weighted_exit_value / total_exit_qty
                tier_state.remaining_qty = 0
                if tier_state.stage != "fail_safe":
                    tier_state.stage = "closed"
                tier_state.clear_pending(now)
            reason = str(context.get("reason") or "position_closed")
            if not context and (fill or {}).get("is_protective_stop"):
                reason = "option_stop_loss"
            logger.info("Position closed: %s (realized/estimated P/L: $%.2f)", symbol, total_pnl)
            self._record_session_closed(
                symbol=symbol,
                pnl=total_pnl,
                exit_reason=reason,
            )
            if self._is_option_position(old_pos):
                self._exit_events.append(
                    {
                        "event_type": "option_exit_filled",
                        "symbol": symbol,
                        "details": {
                            "reason": reason,
                            "entry_order_id": meta.get("order_id"),
                            "exit_order": fill,
                            "actual_exit_price": total_exit_price,
                            "actual_filled_qty": total_exit_qty,
                            "realized_pnl": total_pnl,
                            "partial_realized_pnl": partial_realized_pnl,
                            "tiered_exit_state": self._tier_state_dict(meta),
                            "position": old_pos,
                        },
                    }
                )
                self._record_stopout_and_overshoot(symbol, old_pos, total_pnl, reason)

            if self.storage and meta:
                try:
                    if total_exit_price is None:
                        total_exit_price = old_pos.get("current_price")
                    if total_exit_price is not None:
                        try:
                            total_exit_price = float(total_exit_price)
                        except (TypeError, ValueError):
                            total_exit_price = None
                    if total_exit_price is None and total_pnl:
                        qty = abs(float(old_pos.get("qty", 0) or 0))
                        entry = old_pos.get("avg_entry_price") or old_pos.get("entry_price")
                        try:
                            entry_f = float(entry) if entry is not None else None
                        except (TypeError, ValueError):
                            entry_f = None
                        side = str(old_pos.get("side", "")).lower()
                        if qty > 0 and entry_f is not None:
                            multiplier = 100.0 if self._is_option_position(old_pos) else 1.0
                            if side in ("long", "buy"):
                                total_exit_price = entry_f + (total_pnl / (qty * multiplier))
                            elif side in ("short", "sell"):
                                total_exit_price = entry_f - (total_pnl / (qty * multiplier))
                    if meta.get("trade_pk") is not None:
                        trade_pk = int(meta["trade_pk"])
                        if hasattr(self.storage, "close_trade_by_pk"):
                            self.storage.close_trade_by_pk(
                                trade_pk,
                                datetime.now(timezone.utc),
                                exit_price=total_exit_price,
                                pnl=total_pnl,
                                exit_reason=reason,
                            )
                        elif meta.get("order_id"):
                            self.storage.save_trade(
                                {
                                    "trade_id": meta["order_id"],
                                    "status": "closed",
                                    "exit_time": datetime.now(timezone.utc),
                                    "exit_price": total_exit_price,
                                    "pnl": total_pnl,
                                    "exit_reason": reason,
                                }
                            )
                        self.storage.delete_position_by_trade_pk(trade_pk)
                except Exception as exc:
                    logger.error("Failed to persist position close for %s: %s", symbol, exc)

        return positions

    def pop_exit_events(self) -> List[Dict[str, Any]]:
        """Return and clear option exit events generated during position updates."""
        events = list(self._exit_events)
        self._exit_events.clear()
        return events

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        return self.tracked_positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked positions."""
        return self.tracked_positions.copy()

    def calculate_total_unrealized_pl(self) -> float:
        """Calculate total unrealized P/L across all positions."""
        return sum(
            float(pos.get("unrealized_pl", 0.0) or 0.0)
            for pos in self.tracked_positions.values()
        )

    def check_stop_loss_take_profit(self) -> List[Dict[str, Any]]:
        """
        Check if any positions hit stop loss or take profit.

        Note: With bracket orders, Alpaca handles SL/TP automatically.
        This method is for monitoring/logging purposes.

        Returns:
            List of positions that may have been exited
        """
        current_positions = self.update_positions()
        return current_positions

    def _close_option_positions_if_needed(
        self,
        positions: List[Dict[str, Any]],
        now: datetime,
        force_close_options: bool,
        force_close_reason: str,
    ) -> List[Dict[str, Any]]:
        if not self.options_settings:
            return positions

        remaining: List[Dict[str, Any]] = []
        for pos in positions:
            symbol = str(pos.get("symbol", ""))
            if not self._is_option_position(pos):
                remaining.append(pos)
                continue

            reason = self._option_exit_reason(
                symbol=symbol,
                pos=pos,
                now=now,
                force_close_options=force_close_options,
                force_close_reason=force_close_reason,
            )
            if not reason:
                remaining.append(pos)
                continue

            if symbol in self._closing_symbols:
                remaining.append(pos)
                continue

            self._closing_symbols.add(symbol)
            closed = False
            try:
                pending_state = parse_tiered_state(
                    self._order_meta.get(symbol, {}).get("tiered_exit_state")
                )
                if pending_state is not None and pending_state.has_pending_exit:
                    if hasattr(self.order_manager, "cancel_exit_order"):
                        cancel = self.order_manager.cancel_exit_order(
                            symbol, pending_state.pending_order_id
                        ) or {}
                        if cancel.get("status") == "filled":
                            self._closing_symbols.discard(symbol)
                            remaining.append(pos)
                            continue
                        if cancel.get("status") not in {
                            "absent",
                            "canceled",
                            "rejected",
                            "expired",
                            "replaced",
                        }:
                            self._closing_symbols.discard(symbol)
                            remaining.append(pos)
                            continue
                    pending_state.clear_pending(now)
                    self._persist_tier_state(symbol, pending_state, pos)
                closed = bool(self.order_manager.close_position(symbol))
            except Exception as exc:
                logger.error("Failed to request option exit for %s: %s", symbol, exc)

            details = {
                "reason": reason,
                "close_requested": closed,
                "position": pos,
                "unrealized_pl": self._float_or_zero(pos.get("unrealized_pl")),
                "unrealized_plpc": self._normalized_pct(pos.get("unrealized_plpc")),
                "hold_minutes": self._hold_minutes(symbol, now),
            }
            self._exit_events.append(
                {
                    "event_type": "option_exit_requested" if closed else "option_exit_failed",
                    "symbol": symbol,
                    "details": details,
                }
            )

            if closed:
                self._closing_context[symbol] = {
                    "reason": reason,
                    "requested_at": now.isoformat(),
                    "position": pos,
                }
                # Keep the position tracked until Alpaca confirms it is absent.
                remaining.append(pos)
            else:
                self._closing_symbols.discard(symbol)
                remaining.append(pos)

        return remaining

    def _manage_tiered_positions(
        self,
        positions: List[Dict[str, Any]],
        now: datetime,
        *,
        force_close_options: bool = False,
    ) -> None:
        if not self.options_settings:
            return
        if force_close_options:
            return
        for pos in positions:
            if not self._is_option_position(pos):
                continue
            symbol = str(pos.get("symbol", "")).upper()
            meta = self._order_meta.get(symbol, {})
            state = parse_tiered_state(meta.get("tiered_exit_state"))
            if state is None or not state.is_active:
                continue
            if not state.quantity_integrity_ok:
                self._exit_events.append(
                    {
                        "event_type": "option_tiered_exit_quantity_mismatch",
                        "symbol": symbol,
                        "details": state.to_dict(),
                    }
                )
                self._enter_tier_fail_safe(
                    symbol, pos, state, "quantity_integrity_mismatch", now
                )
                continue
            broker_qty = int(round(abs(float(pos.get("qty") or 0))))
            if not state.has_pending_exit and broker_qty != int(state.remaining_qty):
                self._exit_events.append(
                    {
                        "event_type": "option_tiered_exit_quantity_mismatch",
                        "symbol": symbol,
                        "details": {
                            "broker_qty": broker_qty,
                            "state_remaining_qty": state.remaining_qty,
                            "state": state.to_dict(),
                        },
                    }
                )
                self._enter_tier_fail_safe(
                    symbol, pos, state, "broker_quantity_mismatch", now
                )
                continue
            self._record_shadow_crossings(symbol, pos, state, now)
            if state.has_pending_exit:
                continue
            option_price = self._float_or_none(pos.get("current_price"))
            if option_price is None:
                continue
            return_pct = state.current_return_pct(option_price)
            previous_peak = state.peak_return_pct
            previous_mfe = state.mfe_return_pct
            previous_mae = state.mae_return_pct
            action = state.next_profit_stage(return_pct)
            if (
                state.peak_return_pct != previous_peak
                or state.mfe_return_pct != previous_mfe
                or state.mae_return_pct != previous_mae
            ):
                self._persist_tier_state(symbol, state, pos)
            if action is not None:
                stage, qty = action
                self._request_tiered_exit(symbol, pos, state, stage, qty, now)

    def _record_shadow_crossings(
        self,
        symbol: str,
        pos: Dict[str, Any],
        state: TieredExitState,
        now: datetime,
    ) -> None:
        """Record first crossings of entry-time underlying levels.

        These are observational only.  A crossing is never passed to the exit
        decision path during this paper trial.
        """
        meta = self._order_meta.get(symbol, {})
        underlying = str(meta.get("underlying_symbol") or state.underlying_symbol).upper()
        current = self._underlying_marks.get(underlying)
        if current is None:
            return
        crossings = state.shadow_crossings
        previous = self._float_or_none(crossings.get("_last_underlying_price"))
        if previous is not None and previous != current:
            for name, raw_level in state.shadow_levels.items():
                level = self._float_or_none(raw_level)
                if level is None:
                    continue
                crossed = (previous - level) * (float(current) - level) <= 0
                if not crossed:
                    continue
                entries = crossings.setdefault(name, [])
                if entries:
                    continue
                entry = {
                    "crossed_at": now.astimezone(timezone.utc).isoformat(),
                    "underlying_price": float(current),
                    "level": float(level),
                    "option_return_pct": state.current_return_pct(
                        self._float_or_zero(pos.get("current_price"))
                    ),
                }
                entries.append(entry)
                self._exit_events.append(
                    {
                        "event_type": "option_shadow_level_crossing",
                        "symbol": symbol,
                        "details": {"level_name": name, **entry},
                    }
                )
        crossings["_last_underlying_price"] = float(current)
        self._persist_tier_state(symbol, state, pos)

    def _request_tiered_exit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        state: TieredExitState,
        stage: str,
        qty: int,
        now: datetime,
    ) -> None:
        if not hasattr(self.order_manager, "close_position_quantity"):
            self._enter_tier_fail_safe(
                symbol,
                pos,
                state,
                "partial_close_interface_unavailable",
                now,
            )
            return
        attempt = state.pending_attempts + 1
        client_order_id = self._tier_client_order_id(state.lifecycle_id, stage, attempt)
        self._exit_events.append(
            {
                "event_type": "option_runner_trail_triggered"
                if stage == "runner_trail"
                else "option_tiered_exit_triggered",
                "symbol": symbol,
                "details": {
                    "stage": stage,
                    "qty": int(qty),
                    "return_pct": state.current_return_pct(
                        self._float_or_zero(pos.get("current_price"))
                    ),
                    "lifecycle_id": state.lifecycle_id,
                },
            }
        )
        try:
            state.begin_pending(
                stage=stage,
                qty=min(int(qty), int(float(pos.get("qty") or state.remaining_qty))),
                client_order_id=client_order_id,
                now=now,
            )
            self._persist_tier_state(symbol, state, pos)
            try:
                result = self.order_manager.close_position_quantity(
                    symbol,
                    state.pending_qty,
                    client_order_id=client_order_id,
                    stage=stage,
                    lifecycle_id=state.lifecycle_id,
                )
            except TypeError as exc:
                if "lifecycle_id" not in str(exc):
                    raise
                result = self.order_manager.close_position_quantity(
                    symbol,
                    state.pending_qty,
                    client_order_id=client_order_id,
                    stage=stage,
                )
        except Exception as exc:
            result = {"success": False, "error": "tiered_exit_exception", "detail": str(exc)}

        if not result.get("success"):
            state.clear_pending(now)
            self._persist_tier_state(symbol, state, pos)
            self._exit_events.append(
                {
                    "event_type": "option_tiered_exit_failed",
                    "symbol": symbol,
                    "details": {
                        "entry_order_id": self._order_meta.get(symbol, {}).get("order_id"),
                        "stage": stage,
                        "qty": qty,
                        "error": result.get("error"),
                        "detail": result.get("detail"),
                    },
                }
            )
            self._restore_or_fail_safe(symbol, pos, state, now)
            return

        order_id = str(result.get("order_id") or "")
        if order_id:
            state.attach_pending_order(order_id, now)
        self._persist_tier_state(symbol, state, pos)
        self._exit_events.append(
            {
                "event_type": "option_partial_exit_requested",
                "symbol": symbol,
                "details": {
                    "stage": stage,
                    "qty": state.pending_qty,
                    "order_id": order_id,
                    "client_order_id": client_order_id,
                    "return_pct": state.current_return_pct(
                        self._float_or_zero(pos.get("current_price"))
                    ),
                },
            }
        )

    def _reconcile_tiered_pending_fill(
        self,
        symbol: str,
        previous_pos: Optional[Dict[str, Any]],
        pos: Dict[str, Any],
        now: datetime,
    ) -> None:
        meta = self._order_meta.get(symbol, {})
        state = parse_tiered_state(meta.get("tiered_exit_state"))
        if state is None or not state.has_pending_exit:
            return
        if hasattr(self.order_manager, "get_exit_order_status"):
            broker_order = self.order_manager.get_exit_order_status(
                symbol,
                state.pending_order_id,
                state.pending_client_order_id,
            ) or {}
            recovered_order_id = str(broker_order.get("order_id") or "")
            if recovered_order_id and recovered_order_id != state.pending_order_id:
                state.attach_pending_order(recovered_order_id, now)
                self._persist_tier_state(symbol, state, pos)
        before_qty = (
            abs(float(previous_pos.get("qty") or 0))
            if previous_pos is not None
            else state.remaining_qty
        )
        current_qty = abs(float(pos.get("qty") or 0))
        filled_qty = min(state.pending_qty, max(0, int(round(before_qty - current_qty))))
        if filled_qty > 0:
            requested_qty = state.pending_qty
            fill = self._latest_exit_fill(symbol) or {}
            fill_price = self._float_or_none(fill.get("filled_avg_price"))
            if fill_price is None:
                fill_price = self._float_or_none(previous_pos and previous_pos.get("current_price"))
            if fill_price is None:
                fill_price = self._float_or_none(pos.get("current_price")) or state.entry_price
            stage = state.pending_stage or "tier"
            state.apply_fill(
                stage=stage,
                qty=filled_qty,
                exit_price=fill_price,
                pending_remaining_qty=max(0, requested_qty - filled_qty),
                now=now,
            )
            self._persist_tier_state(symbol, state, pos)
            self._exit_events.append(
                {
                    "event_type": "option_partial_exit_filled",
                    "symbol": symbol,
                    "details": {
                        "entry_order_id": meta.get("order_id"),
                        "stage": stage,
                        "filled_qty": filled_qty,
                        "filled_avg_price": fill_price,
                        "realized_pnl": (fill_price - state.entry_price) * filled_qty * 100.0,
                        "remaining_qty": state.remaining_qty,
                        "tiered_exit_state": state.to_dict(),
                        "order": fill,
                    },
                }
            )
            if (
                state.remaining_qty > 0
                and not state.has_pending_exit
                and hasattr(self.order_manager, "ensure_protective_stops")
            ):
                self._ensure_protection_for_state(symbol, pos, state, now)
            return

        requested_at = self._parse_datetime(state.pending_requested_at)
        timeout = int(self.options_settings.tiered_exit_fill_timeout_seconds)
        if requested_at is None or (now - requested_at).total_seconds() < timeout:
            return
        cancel_result: Dict[str, Any] = {"status": "absent"}
        if hasattr(self.order_manager, "cancel_exit_order"):
            cancel_result = self.order_manager.cancel_exit_order(
                symbol, state.pending_order_id
            ) or {"status": "query_failed"}
            if cancel_result.get("status") == "filled":
                # Let the next broker position poll reconcile the filled
                # quantity before issuing any retry.
                return
            if cancel_result.get("status") not in {
                "absent",
                "canceled",
                "rejected",
                "expired",
                "replaced",
            }:
                self._enter_tier_fail_safe(
                    symbol,
                    pos,
                    state,
                    "tier_close_cancel_unconfirmed",
                    now,
                )
                return
        stage = state.pending_stage or "tier"
        missing_qty = int(state.pending_qty)
        retry_count = state.pending_attempts
        state.clear_pending(now)
        self._persist_tier_state(symbol, state, pos)
        self._exit_events.append(
            {
                "event_type": "option_tiered_exit_retry",
                "symbol": symbol,
                "details": {"stage": stage, "attempts": retry_count, "reason": "fill_timeout"},
            }
        )
        if retry_count >= 2:
            self._enter_tier_fail_safe(symbol, pos, state, f"{stage}_fill_timeout", now)
        else:
            if self._ensure_protection_for_state(symbol, pos, state, now):
                self._request_tiered_exit(symbol, pos, state, stage, missing_qty, now)

    def _initialize_tiered_state(
        self,
        *,
        symbol: str,
        qty: int,
        entry_price: float,
        position: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> Optional[TieredExitState]:
        settings = self.options_settings
        if not settings or not settings.tiered_exit_enabled:
            return None
        # Startup reconciliation intentionally registers broker positions as
        # legacy.  Only the live entry path may opt a newly filled lifecycle
        # into this paper experiment.
        if not bool(meta.get("tiered_candidate")):
            return None
        if meta.get("tiered_exit_unreadable"):
            return None
        if str(meta.get("underlying_symbol") or "").upper() not in settings.tiered_exit_underlyings:
            return None
        if int(qty) < settings.tiered_min_contracts:
            return None
        existing = parse_tiered_state(meta.get("tiered_exit_state") or meta.get("exit_state"))
        if existing is not None:
            return existing
        option_plan = meta.get("option_plan") if isinstance(meta.get("option_plan"), dict) else {}
        underlying_plan = option_plan.get("underlying_trade_plan") if isinstance(option_plan, dict) else {}
        underlying_plan = underlying_plan if isinstance(underlying_plan, dict) else {}
        supplied_shadow = option_plan.get("shadow_levels") if isinstance(option_plan, dict) else {}
        shadow_levels: Dict[str, Any] = {
            **(supplied_shadow if isinstance(supplied_shadow, dict) else {}),
            "underlying_entry": underlying_plan.get("entry_price"),
            "underlying_stop": underlying_plan.get("stop_loss"),
            "underlying_target": underlying_plan.get("take_profit"),
            "entry_mean": option_plan.get("entry_mu"),
            "entry_sigma": option_plan.get("entry_sigma"),
            "prior_day_high": option_plan.get("prior_day_high"),
            "prior_day_low": option_plan.get("prior_day_low"),
            "prior_day_close": option_plan.get("prior_day_close"),
        }
        original_stop = float(entry_price) * (1.0 - float(settings.stop_loss_pct))
        state = TieredExitState.create(
            lifecycle_id=str(meta.get("trade_pk") or meta.get("order_id") or symbol),
            underlying_symbol=str(meta.get("underlying_symbol") or ""),
            option_symbol=symbol,
            initial_qty=int(qty),
            entry_price=float(entry_price),
            original_stop_price=original_stop,
            tp1_return_pct=settings.tiered_tp1_return_pct,
            tp2_return_pct=settings.tiered_tp2_return_pct,
            tp1_fraction=settings.tiered_tp1_fraction,
            tp2_fraction=settings.tiered_tp2_fraction,
            post_tp1_stop_return_pct=settings.tiered_post_tp1_stop_return_pct,
            runner_floor_return_pct=settings.tiered_runner_floor_return_pct,
            runner_giveback_pct=settings.tiered_runner_giveback_pct,
            shadow_levels=shadow_levels,
        )
        self._exit_events.append(
            {
                "event_type": "option_tiered_exit_initialized",
                "symbol": symbol,
                "details": state.to_dict(),
            }
        )
        return state

    def _persist_tier_state(
        self,
        symbol: str,
        state: TieredExitState,
        pos: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = self._order_meta.setdefault(symbol, {})
        meta["tiered_exit_state"] = state
        if not self.storage or not hasattr(self.storage, "update_position"):
            return
        position = pos or self.tracked_positions.get(symbol) or {}
        try:
            self.storage.update_position(
                {
                    "trade_id": meta.get("trade_pk"),
                    "symbol": symbol,
                    "asset_class": position.get("asset_class") or "option",
                    "underlying_symbol": meta.get("underlying_symbol"),
                    "option_symbol": meta.get("option_symbol") or symbol,
                    "side": position.get("side") or "long",
                    "entry_price": position.get("entry_price") or state.entry_price,
                    "current_price": position.get("current_price"),
                    "stop_loss": state.active_stop_price,
                    "take_profit": state.entry_price * (1.0 + state.tp2_return_pct),
                    "qty": state.remaining_qty,
                    "unrealized_pnl": position.get("unrealized_pl") or 0.0,
                    "exit_state": state.to_dict(),
                }
            )
        except Exception as exc:
            logger.error("Failed to persist tiered exit state for %s: %s", symbol, exc)

    def _ensure_protection_for_state(
        self,
        symbol: str,
        pos: Dict[str, Any],
        state: TieredExitState,
        now: datetime,
    ) -> bool:
        try:
            events = self.order_manager.ensure_protective_stops(
                positions=[pos],
                order_meta=self._order_meta,
            )
            risk_events = list(events or [])
            self._exit_events.extend(risk_events)
            failed = any(
                event.get("event_type") == "option_protective_stop_failed"
                for event in risk_events
            )
            if failed:
                self._enter_tier_fail_safe(symbol, pos, state, "protection_restore_failed", now)
                return False
            self._exit_events.append(
                {
                    "event_type": "option_tiered_protection_replaced",
                    "symbol": symbol,
                    "details": {
                        "active_stop_price": state.active_stop_price,
                        "remaining_qty": state.remaining_qty,
                    },
                }
            )
            return True
        except Exception as exc:
            logger.error("Failed to restore tiered protection for %s: %s", symbol, exc)
            self._enter_tier_fail_safe(symbol, pos, state, "protection_restore_exception", now)
            return False

    def _restore_or_fail_safe(
        self,
        symbol: str,
        pos: Dict[str, Any],
        state: TieredExitState,
        now: datetime,
    ) -> None:
        if self._ensure_protection_for_state(symbol, pos, state, now):
            return
        self._enter_tier_fail_safe(symbol, pos, state, "partial_exit_protection_failed", now)

    def _enter_tier_fail_safe(
        self,
        symbol: str,
        pos: Dict[str, Any],
        state: TieredExitState,
        reason: str,
        now: datetime,
    ) -> None:
        if state.has_pending_exit and hasattr(self.order_manager, "cancel_exit_order"):
            cancel = self.order_manager.cancel_exit_order(
                symbol, state.pending_order_id
            ) or {}
            if cancel.get("status") == "filled":
                self._exit_events.append(
                    {
                        "event_type": "option_tiered_exit_fail_safe",
                        "symbol": symbol,
                        "details": {"reason": "pending_order_filled_during_fail_safe"},
                    }
                )
                return
            if cancel.get("status") not in {
                "absent",
                "canceled",
                "rejected",
                "expired",
                "replaced",
            }:
                self._exit_events.append(
                    {
                        "event_type": "option_tiered_exit_critical",
                        "symbol": symbol,
                        "details": {"reason": "pending_order_cancel_unconfirmed", "cancel": cancel},
                    }
                )
                return
        state.mark_fail_safe(reason, now)
        self._persist_tier_state(symbol, state, pos)
        self._exit_events.append(
            {
                "event_type": "option_tiered_exit_fail_safe",
                "symbol": symbol,
                "details": {"reason": reason, "state": state.to_dict()},
            }
        )
        if symbol in self._closing_symbols:
            return
        try:
            if self.order_manager.close_position(symbol):
                self._closing_symbols.add(symbol)
                self._closing_context[symbol] = {
                    "reason": "tiered_exit_fail_safe",
                    "requested_at": now.isoformat(),
                    "position": pos,
                }
        except Exception as exc:
            logger.critical("Tiered fail-safe close failed for %s: %s", symbol, exc)

    @staticmethod
    def _tier_client_order_id(lifecycle_id: str, stage: str, attempt: int) -> str:
        import hashlib

        digest = hashlib.sha1(str(lifecycle_id).encode("utf-8")).hexdigest()[:10]
        return f"llma-tier-{digest}-{str(stage).lower()}-{int(attempt)}"[:48]

    @staticmethod
    def _tier_state_dict(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state = parse_tiered_state(meta.get("tiered_exit_state")) if meta else None
        return state.to_dict() if state is not None else None

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def get_session_closed(self) -> List[Dict[str, Any]]:
        """Closed trades accrued during this process lifetime (for live-state telemetry)."""
        return list(self.session_closed)

    def _record_session_closed(self, symbol: str, pnl: float, exit_reason: str) -> None:
        self.session_closed.append(
            {
                "symbol": symbol,
                "pnl": float(pnl),
                "exit_reason": exit_reason,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _option_exit_reason(
        self,
        symbol: str,
        pos: Dict[str, Any],
        now: datetime,
        force_close_options: bool,
        force_close_reason: str,
    ) -> Optional[str]:
        if force_close_options:
            return force_close_reason

        meta = self._order_meta.get(str(symbol).upper(), {})
        tier_state = parse_tiered_state(meta.get("tiered_exit_state"))
        pct = self._normalized_pct(pos.get("unrealized_plpc"))
        if tier_state is not None and tier_state.is_active:
            # Profit tiers are submitted by _manage_tiered_positions.  This
            # path only handles a hard stop/time/forced exit so a single mark
            # cannot accidentally submit a partial and then flatten the rest.
            current_price = self._float_or_none(pos.get("current_price"))
            if current_price is not None and current_price > 0:
                pct = tier_state.current_return_pct(current_price)
            if not tier_state.has_pending_exit:
                active_stop_return = tier_state.current_return_pct(tier_state.active_stop_price)
                if pct <= active_stop_return:
                    return "option_tiered_stop" if tier_state.stage != "pre_tp1" else "option_stop_loss"
        elif tier_state is None or tier_state.stage == "fail_safe":
            if pct >= float(self.options_settings.profit_target_pct):
                return "option_profit_target"
            if pct <= -float(self.options_settings.stop_loss_pct):
                return "option_stop_loss"

        hold_minutes = self._hold_minutes(symbol, now)
        dte = option_dte(symbol, now.date())
        time_stop_applies = not self.options_settings.allow_overnight or (
            dte is not None and dte <= 0
        )
        if (
            time_stop_applies
            and hold_minutes is not None
            and hold_minutes >= self.options_settings.max_hold_minutes
        ):
            return "option_time_stop"
        return None

    def _persist_closed_position(self, symbol: str, pos: Dict[str, Any], reason: str) -> None:
        if not self.storage:
            return

        meta = self._order_meta.get(symbol, {})
        trade_pk = meta.get("trade_pk")
        if trade_pk is None:
            return

        try:
            exit_px = self._float_or_none(pos.get("current_price"))
            pnl = self._float_or_zero(pos.get("unrealized_pl"))
            if hasattr(self.storage, "close_trade_by_pk"):
                self.storage.close_trade_by_pk(
                    int(trade_pk),
                    datetime.now(timezone.utc),
                    exit_price=exit_px,
                    pnl=pnl,
                    exit_reason=reason,
                )
            self.storage.delete_position_by_trade_pk(int(trade_pk))
        except Exception as exc:
            logger.error("Failed to persist option exit for %s: %s", symbol, exc)

    def _persist_actual_entry_fills(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            if event.get("event_type") != "option_protective_stop_submitted":
                continue
            symbol = str(event.get("symbol", ""))
            meta = self._order_meta.get(symbol, {})
            trade_pk = meta.get("trade_pk")
            details = event.get("details") or {}
            if details.get("protection_replacement"):
                continue
            if trade_pk is None or details.get("actual_entry_price") is None:
                continue
            try:
                qty = int(float(details.get("actual_filled_qty") or 0))
                entry_price = float(details["actual_entry_price"])
                if self.storage and hasattr(self.storage, "update_trade_entry_fill"):
                    self.storage.update_trade_entry_fill(
                        int(trade_pk),
                        qty=qty,
                        entry_price=entry_price,
                    )
                position = (
                    details.get("position")
                    if isinstance(details.get("position"), dict)
                    else {}
                )
                tier_state = self._initialize_tiered_state(
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    position=position,
                    meta=meta,
                )
                if tier_state is not None:
                    meta["tiered_exit_state"] = tier_state
                self.storage.update_position(
                    {
                        "trade_id": int(trade_pk),
                        "symbol": symbol,
                        "asset_class": meta.get("asset_class") or "option",
                        "underlying_symbol": meta.get("underlying_symbol"),
                        "option_symbol": meta.get("option_symbol") or symbol,
                        "side": position.get("side") or "long",
                        "entry_price": entry_price,
                        "current_price": position.get("current_price") or entry_price,
                        "stop_loss": tier_state.active_stop_price if tier_state else None,
                        "take_profit": (
                            tier_state.entry_price * (1.0 + tier_state.tp2_return_pct)
                            if tier_state
                            else None
                        ),
                        "qty": qty,
                        "unrealized_pnl": position.get("unrealized_pl") or 0.0,
                        "exit_state": tier_state.to_dict() if tier_state else None,
                    }
                ) if self.storage else None
            except Exception as exc:
                logger.error("Failed to persist actual entry fill for %s: %s", symbol, exc)

    def _close_unprotected_positions(
        self,
        events: List[Dict[str, Any]],
        now: datetime,
    ) -> None:
        for event in events:
            if event.get("event_type") != "option_protective_stop_failed":
                continue
            symbol = str(event.get("symbol", ""))
            if not symbol or symbol in self._closing_symbols:
                continue
            logger.error("Closing unprotected option position %s immediately", symbol)
            try:
                if self.order_manager.close_position(symbol):
                    self._closing_symbols.add(symbol)
                    self._closing_context[symbol] = {
                        "reason": "option_protection_failed",
                        "requested_at": now.isoformat(),
                    }
            except Exception as exc:
                logger.critical("Emergency close failed for unprotected %s: %s", symbol, exc)

    def _latest_exit_fill(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not hasattr(self.order_manager, "get_latest_exit_fill"):
            return None
        try:
            fill = self.order_manager.get_latest_exit_fill(symbol)
            return fill if isinstance(fill, dict) else None
        except Exception as exc:
            logger.error("Failed to load actual exit fill for %s: %s", symbol, exc)
            return None

    @classmethod
    def _realized_pnl_from_fill(
        cls,
        pos: Dict[str, Any],
        exit_price: Optional[float],
        exit_qty: Optional[float],
    ) -> Optional[float]:
        entry_price = cls._float_or_none(
            pos.get("avg_entry_price") or pos.get("entry_price")
        )
        qty = abs(exit_qty or cls._float_or_zero(pos.get("qty")))
        if entry_price is None or exit_price is None or qty <= 0:
            return None
        multiplier = 100.0 if cls._is_option_position(pos) else 1.0
        side = str(pos.get("side", "")).lower()
        direction = -1.0 if side in ("short", "sell") else 1.0
        return (exit_price - entry_price) * qty * multiplier * direction

    def _record_stopout_and_overshoot(
        self,
        symbol: str,
        pos: Dict[str, Any],
        pnl: float,
        reason: str,
    ) -> None:
        if reason != "option_stop_loss" or not self.options_settings:
            return
        if hasattr(self.order_manager, "record_stopout"):
            try:
                self.order_manager.record_stopout(symbol, minutes=60)
            except Exception as exc:
                logger.error("Failed to record stopout cooldown for %s: %s", symbol, exc)
        cost_basis = abs(self._float_or_zero(pos.get("cost_basis")))
        if cost_basis <= 0:
            return
        realized_loss_pct = max(0.0, -float(pnl) / cost_basis)
        policy = float(self.options_settings.stop_loss_pct)
        if realized_loss_pct <= policy + 0.05:
            return
        self._exit_events.append(
            {
                "event_type": "option_stop_overshoot",
                "symbol": symbol,
                "details": {
                    "policy_stop_pct": policy,
                    "realized_loss_pct": realized_loss_pct,
                    "overshoot_points": realized_loss_pct - policy,
                    "realized_pnl": pnl,
                    "cost_basis": cost_basis,
                },
            }
        )

    def _hold_minutes(self, symbol: str, now: datetime) -> Optional[float]:
        opened_at = self._order_meta.get(symbol, {}).get("opened_at")
        if isinstance(opened_at, str):
            try:
                opened_at = datetime.fromisoformat(opened_at)
            except ValueError:
                return None
        if not isinstance(opened_at, datetime):
            return None
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        return max(0.0, (now - opened_at.astimezone(timezone.utc)).total_seconds() / 60.0)

    @staticmethod
    def _is_option_position(pos: Dict[str, Any]) -> bool:
        asset_class = str(pos.get("asset_class", "")).lower()
        return asset_class in ("option", "us_option") or bool(pos.get("option_symbol"))

    @staticmethod
    def _normalized_pct(value: Any) -> float:
        try:
            pct = float(value)
        except (TypeError, ValueError):
            return 0.0
        return pct / 100.0 if abs(pct) > 5.0 else pct

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _float_or_zero(cls, value: Any) -> float:
        return cls._float_or_none(value) or 0.0
