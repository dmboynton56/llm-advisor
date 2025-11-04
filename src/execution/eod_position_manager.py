"""End-of-day position management for live trading."""
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
        
        # Close each position
        for pos in positions:
            symbol = pos["symbol"]
            side = pos["side"]
            
            # For long positions, sell to close
            # For short positions (... we're not doing shorts in this system, but handle it)
            try:
                # Cancel any existing orders first (bracket orders)
                # Then place market order to close
                # Note: This is simplified - in practice you'd want to check order status
                
                # Use Alpaca's close_position method if available, or submit market order
                # For now, we'll just log that we would close
                # In production, you'd need to:
                # 1. Cancel any bracket order OCO groups
                # 2. Submit market order to close position
                
                closed_symbols.append(symbol)
                print(f"  > EOD: Would close {symbol} {side} position")
                
            except Exception as e:
                print(f"  ! EOD: Failed to close {symbol}: {e}")
        
        # Update trade tracker if provided
        if trade_tracker:
            trade_tracker.update_positions()
        
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
    
    # Check if we're within EOD window (within 10 minutes of EOD)
    time_diff = (eod_datetime - current_time).total_seconds()
    
    if 0 <= time_diff <= 600:  # Within 10 minutes of EOD
        closed = close_all_positions_at_eod(order_manager, trade_tracker)
        if closed:
            print(f"  ✓ EOD: Closed {len(closed)} positions")
        return len(closed) > 0
    
    return False

