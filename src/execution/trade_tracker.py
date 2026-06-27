"""Track open positions and monitor for exits."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import datetime, timezone

from src.core.config import OptionsSettings
from src.execution.order_manager import StockOrderManager

if TYPE_CHECKING:
    from src.data.storage import StorageAdapter

logger = logging.getLogger(__name__)


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
        self._exit_events: List[Dict[str, Any]] = []

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

        for pos in positions:
            symbol = pos["symbol"]
            self.tracked_positions[symbol] = {
                **pos,
                "last_updated": datetime.now().isoformat(),
            }

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
            u_pnl = float(old_pos.get("unrealized_pl", 0) or 0)
            logger.info("Position closed: %s (last unrealized P/L: $%.2f)", symbol, u_pnl)

            if self.storage and meta:
                try:
                    exit_px = old_pos.get("current_price")
                    if exit_px is not None:
                        try:
                            exit_px = float(exit_px)
                        except (TypeError, ValueError):
                            exit_px = None
                    if exit_px is None and u_pnl:
                        qty = abs(float(old_pos.get("qty", 0) or 0))
                        entry = old_pos.get("avg_entry_price") or old_pos.get("entry_price")
                        try:
                            entry_f = float(entry) if entry is not None else None
                        except (TypeError, ValueError):
                            entry_f = None
                        side = str(old_pos.get("side", "")).lower()
                        if qty > 0 and entry_f is not None:
                            if side in ("long", "buy"):
                                exit_px = entry_f + (u_pnl / qty)
                            elif side in ("short", "sell"):
                                exit_px = entry_f - (u_pnl / qty)
                    if meta.get("trade_pk") is not None:
                        trade_pk = int(meta["trade_pk"])
                        if hasattr(self.storage, "close_trade_by_pk"):
                            self.storage.close_trade_by_pk(
                                trade_pk,
                                datetime.now(timezone.utc),
                                exit_price=exit_px,
                                pnl=u_pnl,
                                exit_reason="position_closed",
                            )
                        elif meta.get("order_id"):
                            self.storage.save_trade(
                                {
                                    "trade_id": meta["order_id"],
                                    "status": "closed",
                                    "exit_time": datetime.now(timezone.utc),
                                    "exit_price": exit_px,
                                    "pnl": u_pnl,
                                    "exit_reason": "position_closed",
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
                self._persist_closed_position(symbol, pos, reason)
                self.tracked_positions.pop(symbol, None)
                self._order_meta.pop(symbol, None)
            else:
                self._closing_symbols.discard(symbol)
                remaining.append(pos)

        return remaining

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
        if hold_minutes is not None and hold_minutes >= self.options_settings.max_hold_minutes:
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
        return asset_class == "option" or bool(pos.get("option_symbol"))

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
