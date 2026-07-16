"""End-of-day position management for live trading."""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.execution.order_manager import StockOrderManager
from src.execution.trade_tracker import option_dte


def _is_option_position(position: Dict[str, Any]) -> bool:
    return str(position.get("asset_class") or "").lower() in ("option", "us_option") or bool(
        position.get("option_symbol")
    )


def should_flatten_at_eod(
    position: Dict[str, Any],
    current_time: datetime,
    allow_overnight: bool,
    eod_flatten_max_dte: int,
) -> bool:
    """Stocks always flatten; overnight-enabled options flatten only near expiry."""
    if not _is_option_position(position):
        return True
    if not allow_overnight:
        return True
    symbol = str(position.get("option_symbol") or position.get("symbol") or "")
    dte = option_dte(symbol, current_time.date())
    # Unknown option expiries are flattened because their overnight risk cannot be classified.
    return dte is None or dte <= eod_flatten_max_dte


def close_all_positions_at_eod(
    order_manager: StockOrderManager,
    trade_tracker: Optional[Any] = None,
    *,
    current_time: Optional[datetime] = None,
    allow_overnight: bool = False,
    eod_flatten_max_dte: int = 0,
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
        
        policy_time = current_time or datetime.now().astimezone()
        for pos in positions:
            if not should_flatten_at_eod(
                pos, policy_time, allow_overnight, eod_flatten_max_dte
            ):
                continue
            symbol = pos["symbol"]
            if order_manager.close_position(symbol):
                closed_symbols.append(symbol)
        
        if trade_tracker:
            for _ in range(3):
                remaining = trade_tracker.update_positions()
                closable_remaining = [
                    pos
                    for pos in remaining
                    if should_flatten_at_eod(
                        pos, policy_time, allow_overnight, eod_flatten_max_dte
                    )
                ]
                if not closable_remaining:
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
    eod_close_time: str,
    *,
    allow_overnight: bool = False,
    eod_flatten_max_dte: int = 0,
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

    if not any(
        should_flatten_at_eod(pos, current_time, allow_overnight, eod_flatten_max_dte)
        for pos in open_now
    ):
        return True
    
    closed = close_all_positions_at_eod(
        order_manager,
        trade_tracker,
        current_time=current_time,
        allow_overnight=allow_overnight,
        eod_flatten_max_dte=eod_flatten_max_dte,
    )
    if closed:
        print(f"  ✓ EOD: Closed {len(closed)} positions")
    try:
        return not any(
            should_flatten_at_eod(pos, current_time, allow_overnight, eod_flatten_max_dte)
            for pos in order_manager.get_open_positions()
        )
    except Exception:
        return False
