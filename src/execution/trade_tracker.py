"""Track open positions and monitor for exits."""
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.execution.order_manager import StockOrderManager


class TradeTracker:
    """Tracks open positions and monitors for exits."""
    
    def __init__(self, order_manager: StockOrderManager):
        """
        Initialize trade tracker.
        
        Args:
            order_manager: StockOrderManager instance
        """
        self.order_manager = order_manager
        self.tracked_positions: Dict[str, Dict[str, Any]] = {}
    
    def update_positions(self) -> List[Dict[str, Any]]:
        """
        Update and return current open positions.
        
        Returns:
            List of position dicts
        """
        positions = self.order_manager.get_open_positions()
        
        # Update tracked positions
        for pos in positions:
            symbol = pos["symbol"]
            self.tracked_positions[symbol] = {
                **pos,
                "last_updated": datetime.now().isoformat(),
            }
        
        # Remove positions that are no longer open
        open_symbols = {pos["symbol"] for pos in positions}
        closed_symbols = set(self.tracked_positions.keys()) - open_symbols
        
        for symbol in closed_symbols:
            old_pos = self.tracked_positions.pop(symbol)
            print(f"  > Position closed: {symbol} (P/L: ${old_pos.get('unrealized_pl', 0):.2f})")
        
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
            pos.get("unrealized_pl", 0.0)
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
        # Bracket orders handle SL/TP automatically, so we just check for closed positions
        current_positions = self.update_positions()
        return current_positions

