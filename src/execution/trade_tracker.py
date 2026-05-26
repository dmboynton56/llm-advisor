"""Track open positions and monitor for exits."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import datetime, timezone

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
    ):
        self.order_manager = order_manager
        self.storage = storage
        self.tracked_positions: Dict[str, Dict[str, Any]] = {}
        self._order_meta: Dict[str, Dict[str, Any]] = {}

    def register_open_trade(self, symbol: str, order_id: Optional[str], trade_pk: int) -> None:
        """Link Alpaca order id + internal trades PK for DB updates on exit."""
        self._order_meta[symbol] = {
            "order_id": str(order_id) if order_id else "",
            "trade_pk": int(trade_pk),
        }

    def update_positions(self) -> List[Dict[str, Any]]:
        """
        Update and return current open positions.

        Returns:
            List of position dicts
        """
        positions = self.order_manager.get_open_positions()

        for pos in positions:
            symbol = pos["symbol"]
            self.tracked_positions[symbol] = {
                **pos,
                "last_updated": datetime.now().isoformat(),
            }

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
