"""Mock order manager for backtesting - simulates trades without API calls."""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


@dataclass
class SimulatedTrade:
    """Simulated trade result."""
    order_id: str
    symbol: str
    side: str
    qty: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "stop_loss", "take_profit", "end_of_day"
    pnl: Optional[float] = None
    status: str = "filled"  # filled, closed


class MockOrderManager:
    """Simulates trade execution for backtesting."""
    
    def __init__(self, initial_equity: float = 100000.0, eod_close_time: str = "15:50"):
        """
        Initialize mock order manager.
        
        Args:
            initial_equity: Starting account equity
            eod_close_time: End-of-day close time (HH:MM)
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.start_of_day_equity = initial_equity  # Track daily starting equity
        self.trades: List[SimulatedTrade] = []
        self.open_positions: Dict[str, SimulatedTrade] = {}
        self.eod_close_time = eod_close_time
        self.order_counter = 0
    
    def get_account_equity(self) -> float:
        """Return current equity (updated by simulated trades)."""
        return self.current_equity
    
    def execute_stock_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr_5m: Optional[float] = None,
        qty: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Simulate trade execution.
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            atr_5m: Optional ATR for position sizing
            qty: Optional quantity (if None, calculated from risk)
            
        Returns:
            Order dict if successful, None otherwise
        """
        # Validate risk:reward ratio
        from src.execution.risk_calculator import validate_risk_reward
        if not validate_risk_reward(entry_price, stop_loss, take_profit, 1.5):
            return None
        
        # Calculate position size if not provided
        if qty is None:
            from src.execution.risk_calculator import calculate_position_size
            qty = calculate_position_size(
                account_equity=self.current_equity,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                max_risk_percent=1.0,  # Default 1%
                atr_5m=atr_5m
            )
        
        if qty <= 0:
            return None
        
        # Don't allow multiple positions in same symbol (simplified)
        if symbol in self.open_positions:
            return None
        
        # Create simulated trade
        self.order_counter += 1
        trade = SimulatedTrade(
            order_id=f"MOCK_{self.order_counter:06d}",
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.trades.append(trade)
        self.open_positions[symbol] = trade
        
        return {
            "order_id": trade.order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "filled",
            "submitted_at": trade.entry_time.isoformat(),
        }
    
    def check_exits(self, current_prices: Dict[str, float], current_time: datetime) -> List[str]:
        """
        Check if any open positions should exit (stop loss / take profit).
        
        Args:
            current_prices: Current prices for all symbols
            current_time: Current simulated time
            
        Returns:
            List of symbols that were closed
        """
        closed_symbols = []
        for symbol, trade in list(self.open_positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Check stop loss and take profit
            if trade.side == "long":
                if current_price <= trade.stop_loss:
                    self._close_trade(trade, current_price, "stop_loss", current_time)
                    closed_symbols.append(symbol)
                elif current_price >= trade.take_profit:
                    self._close_trade(trade, current_price, "take_profit", current_time)
                    closed_symbols.append(symbol)
            else:  # short
                if current_price >= trade.stop_loss:
                    self._close_trade(trade, current_price, "stop_loss", current_time)
                    closed_symbols.append(symbol)
                elif current_price <= trade.take_profit:
                    self._close_trade(trade, current_price, "take_profit", current_time)
                    closed_symbols.append(symbol)
        
        return closed_symbols
    
    def close_all_positions_at_eod(
        self,
        current_time: datetime,
        current_prices: Dict[str, float]
    ) -> int:
        """
        Close all open positions at end of day.
        
        Args:
            current_time: Current simulated time
            current_prices: Current prices for all symbols
            
        Returns:
            Number of positions closed
        """
        closed_count = 0
        for symbol, trade in list(self.open_positions.items()):
            exit_price = current_prices.get(symbol, trade.entry_price)  # Use current price or entry as fallback
            self._close_trade(trade, exit_price, "end_of_day", current_time)
            closed_count += 1
        return closed_count
    
    def _close_trade(
        self,
        trade: SimulatedTrade,
        exit_price: float,
        reason: str,
        exit_time: datetime
    ):
        """Close a simulated trade."""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.status = "closed"
        
        # Calculate P/L
        if trade.side == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.qty
        else:  # short
            trade.pnl = (trade.entry_price - exit_price) * trade.qty
        
        # Update equity
        self.current_equity += trade.pnl
        
        # Remove from open positions
        del self.open_positions[trade.symbol]
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        return [
            {
                "symbol": trade.symbol,
                "qty": trade.qty,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "current_price": trade.entry_price,  # Simplified - would use actual price in real scenario
                "unrealized_pl": 0.0,  # Would calculate from current price
            }
            for trade in self.open_positions.values()
        ]
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all simulated trades."""
        closed_trades = [t for t in self.trades if t.exit_time]
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
        winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl and t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
        
        return {
            "total_trades": len(self.trades),
            "closed_trades": len(closed_trades),
            "open_positions": len(self.open_positions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_pnl": total_pnl,
            "average_win": sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0,
            "average_loss": sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0,
            "final_equity": self.current_equity,
            "return_pct": ((self.current_equity - self.initial_equity) / self.initial_equity) * 100,
            "daily_return_pct": ((self.current_equity - self.start_of_day_equity) / self.start_of_day_equity) * 100 if self.start_of_day_equity > 0 else 0.0,
            "win_rate": win_rate,
            "trades": [asdict(t) for t in self.trades]
        }
    
    def reset_for_new_day(self):
        """Reset for new trading day (for multi-day backtests)."""
        # Save end-of-day state
        self.start_of_day_equity = self.current_equity
        # Keep trade history, but could optionally clear open positions
        # For intraday-only, we'd close all positions first


def execute_trade_from_signal(
    signal: Any,  # SignalEvent from threshold_evaluator
    state: Any,  # SymbolState
    order_manager: MockOrderManager
) -> Optional[Dict[str, Any]]:
    """
    Execute trade from signal event and symbol state (mock version).
    
    Args:
        signal: SignalEvent
        state: SymbolState with trade plan
        order_manager: MockOrderManager instance
        
    Returns:
        Order dict if successful, None otherwise
    """
    if not state.trade:
        return None
    
    return order_manager.execute_stock_trade(
        symbol=signal.symbol,
        side=state.trade.side,
        entry_price=state.trade.entry_price,
        stop_loss=state.trade.sl_price,
        take_profit=state.trade.tp_price,
        atr_5m=state.atr_5m,
    )

