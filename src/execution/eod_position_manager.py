"""End-of-day position management for live trading."""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.execution.order_manager import StockOrderManager


def close_all_positions_at_eod(
    order_manager: StockOrderManager,
    trade_tracker: Optional[Any] = None
) -> List[str]:
    """
    Close all open positions at end of day.
    
    Args:
        order_manager: StockOrderManager instance
        trade_tracker: Optional TradeTracker instance
        
    Returns:
        List of closed position symbols
    """
    closed_symbols = []
    
    try:
        # Get all open positions
        positions = order_manager.get_open_positions()
        
        if not positions:
            return []
        
        for pos in positions:
            symbol = pos["symbol"]
            if order_manager.close_position(symbol):
                closed_symbols.append(symbol)
        
        if trade_tracker:
            for _ in range(3):
                remaining = trade_tracker.update_positions()
                if not remaining:
                    break
                time.sleep(2)
        
        return closed_symbols
        
    except Exception as e:
        print(f"  ! EOD: Error closing positions: {e}")
        return closed_symbols


def check_and_close_eod(
    order_manager: Optional[StockOrderManager],
    trade_tracker: Optional[Any],
    current_time: datetime,
    eod_close_time: str
) -> bool:
    """
    Check if it's time to close positions and do so if needed.
    
    Args:
        order_manager: StockOrderManager instance (can be None)
        trade_tracker: Optional TradeTracker instance
        current_time: Current time
        eod_close_time: End-of-day close time (HH:MM)
        
    Returns:
        True if positions were closed, False otherwise
    """
    if not order_manager:
        return False
    
    # Parse EOD time
    hour, minute = map(int, eod_close_time.split(":"))
    eod_datetime = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    if current_time < eod_datetime:
        return False
    
    try:
        open_now = order_manager.get_open_positions()
    except Exception:
        return False
    
    if not open_now:
        return True
    
    closed = close_all_positions_at_eod(order_manager, trade_tracker)
    if closed:
        print(f"  ✓ EOD: Closed {len(closed)} positions")
    try:
        return len(order_manager.get_open_positions()) == 0
    except Exception:
        return False
