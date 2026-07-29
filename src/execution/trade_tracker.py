"""Track open positions and monitor for exits."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import date, datetime, timezone

from src.core.config import OptionsSettings
from src.execution.order_manager import StockOrderManager

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
        self._order_meta[symbol] = {
            "order_id": str(order_id) if order_id else "",
            "trade_pk": int(trade_pk),
            "opened_at": datetime.now(timezone.utc),
            **(metadata or {}),
        }

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
        positions = self.order_manager.get_open_positions()
        fetched_symbols = {str(pos.get("symbol", "")) for pos in positions}
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
            symbol = pos["symbol"]
            self.tracked_positions[symbol] = {
                **pos,
                "last_updated": datetime.now().isoformat(),
            }
            if self.options_settings and self._is_option_position(pos):
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
                        },
                    }
                )

        positions = self._close_option_positions_if_needed(
            positions,
            now=now,
            force_close_options=force_close_options,
            force_close_reason=force_close_reason,
        )

        open_symbols = {pos["symbol"] for pos in positions}
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
            reason = str(context.get("reason") or "position_closed")
            if not context and (fill or {}).get("is_protective_stop"):
                reason = "option_stop_loss"
            logger.info("Position closed: %s (realized/estimated P/L: $%.2f)", symbol, pnl)
            self._record_session_closed(
                symbol=symbol,
                pnl=pnl,
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
                            "actual_exit_price": exit_px,
                            "actual_filled_qty": exit_qty,
                            "realized_pnl": pnl,
                            "position": old_pos,
                        },
                    }
                )
                self._record_stopout_and_overshoot(symbol, old_pos, pnl, reason)

            if self.storage and meta:
                try:
                    if exit_px is None:
                        exit_px = old_pos.get("current_price")
                    if exit_px is not None:
                        try:
                            exit_px = float(exit_px)
                        except (TypeError, ValueError):
                            exit_px = None
                    if exit_px is None and pnl:
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
                                exit_px = entry_f + (pnl / (qty * multiplier))
                            elif side in ("short", "sell"):
                                exit_px = entry_f - (pnl / (qty * multiplier))
                    if meta.get("trade_pk") is not None:
                        trade_pk = int(meta["trade_pk"])
                        if hasattr(self.storage, "close_trade_by_pk"):
                            self.storage.close_trade_by_pk(
                                trade_pk,
                                datetime.now(timezone.utc),
                                exit_price=exit_px,
                                pnl=pnl,
                                exit_reason=reason,
                            )
                        elif meta.get("order_id"):
                            self.storage.save_trade(
                                {
                                    "trade_id": meta["order_id"],
                                    "status": "closed",
                                    "exit_time": datetime.now(timezone.utc),
                                    "exit_price": exit_px,
                                    "pnl": pnl,
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

        pct = self._normalized_pct(pos.get("unrealized_plpc"))
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
        if not self.storage or not hasattr(self.storage, "update_trade_entry_fill"):
            return
        for event in events:
            if event.get("event_type") != "option_protective_stop_submitted":
                continue
            symbol = str(event.get("symbol", ""))
            meta = self._order_meta.get(symbol, {})
            trade_pk = meta.get("trade_pk")
            details = event.get("details") or {}
            if trade_pk is None or details.get("actual_entry_price") is None:
                continue
            try:
                qty = int(float(details.get("actual_filled_qty") or 0))
                entry_price = float(details["actual_entry_price"])
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
                        "stop_loss": None,
                        "take_profit": None,
                        "qty": qty,
                        "unrealized_pnl": position.get("unrealized_pl") or 0.0,
                    }
                )
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
