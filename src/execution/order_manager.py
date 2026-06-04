"""Stock order manager with bracket orders for STDEV trading.

Refactored from options-focused version to stock trading.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from src.execution.risk_calculator import (
    calculate_position_size,
    validate_risk_reward,
    max_shares_for_buying_power,
)
from src.core.config import Settings
from src.utils.env_sanitize import getenv_strip

load_dotenv()


def is_execution_success(result: Optional[Dict[str, Any]]) -> bool:
    """True when order_manager returned a filled/submitted order payload."""
    return bool(result and result.get("order_id") and not result.get("error"))


def execution_failure_reason(result: Optional[Dict[str, Any]]) -> str:
    if not result:
        return "order_manager_unavailable"
    return str(result.get("error") or "unknown")


class StockOrderManager:
    """Manages stock trade execution via Alpaca bracket orders."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, paper: bool = True):
        """
        Initialize order manager.
        
        Args:
            api_key: Alpaca API key (defaults to env var)
            api_secret: Alpaca API secret (defaults to env var)
            paper: Use paper trading account (default: True)
        """
        api_key = api_key or getenv_strip("ALPACA_API_KEY")
        api_secret = api_secret or getenv_strip("ALPACA_SECRET_KEY")
        
        if not api_key or not api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")
        
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper
        )
        self.settings = Settings.load()
    
    def get_account_equity(self) -> float:
        """Get current account equity."""
        account = self.trading_client.get_account()
        return float(account.equity)

    def get_buying_power(self) -> float:
        """Get current account buying power."""
        account = self.trading_client.get_account()
        return float(account.buying_power)
    
    def close_position(self, symbol: str) -> bool:
        """Close an open position via Alpaca (market close)."""
        try:
            self.trading_client.close_position(symbol)
            print(f"  > Closed position {symbol}")
            return True
        except Exception as e:
            print(f"  ! Failed to close position {symbol}: {e}")
            return False
    
    @staticmethod
    def _round_price(price: float) -> float:
        """Alpaca equity orders require penny increments."""
        return round(float(price), 2)

    @staticmethod
    def _failure(error: str, **extra: Any) -> Dict[str, Any]:
        return {"success": False, "error": error, **extra}

    def _normalize_side(self, side: str) -> str:
        """Normalize trade side values (e.g., 'long' -> 'buy')."""
        if not side:
            raise ValueError("Trade side is required")
        side_lower = side.lower()
        if side_lower in ("buy", "long"):
            return "buy"
        if side_lower in ("sell", "short"):
            return "sell"
        raise ValueError(f"Unsupported trade side: {side}")
    
    def execute_stock_trade(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr_5m: Optional[float] = None,
        qty: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute stock trade via bracket order.
        
        Returns:
            Order dict on success; failure dict with ``error`` key on reject
        """
        stop_loss = self._round_price(stop_loss)
        take_profit = self._round_price(take_profit)

        if not validate_risk_reward(
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            min_rr_ratio=self.settings.risk.min_risk_reward_ratio
        ):
            print(
                f"  ! Trade rejected: Risk:reward ratio below minimum "
                f"({self.settings.risk.min_risk_reward_ratio})"
            )
            return self._failure("rr_below_min")

        if qty is None:
            equity = self.get_account_equity()
            buying_power = self.get_buying_power()
            bp_cap = max_shares_for_buying_power(
                buying_power,
                entry_price,
                self.settings.risk.max_position_notional_pct,
            )
            qty = calculate_position_size(
                account_equity=equity,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                max_risk_percent=self.settings.risk.max_risk_per_trade_percent,
                atr_5m=atr_5m,
                max_shares=bp_cap,
            )
            if qty <= 0 and bp_cap <= 0:
                print(
                    "  ! Trade rejected: insufficient buying power for minimum risk-sized position"
                )
                return self._failure("insufficient_buying_power")
        
        if qty <= 0:
            print(f"  ! Trade rejected: Invalid position size ({qty})")
            return self._failure("invalid_qty", qty=qty)
        
        normalized_side = self._normalize_side(side)
        order_side = OrderSide.BUY if normalized_side == "buy" else OrderSide.SELL
        
        bracket_order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_loss),
            take_profit=TakeProfitRequest(limit_price=take_profit)
        )
        
        try:
            print(f"  > Submitting {normalized_side.upper()} bracket order for {qty} shares of {symbol}...")
            print(f"    Entry: Market (target: ${entry_price:.2f})")
            print(f"    Stop Loss: ${stop_loss:.2f}")
            print(f"    Take Profit: ${take_profit:.2f}")
            
            order = self.trading_client.submit_order(order_data=bracket_order)
            
            print("  --- TRADE EXECUTED SUCCESSFULLY ---")
            print(f"    Order ID: {order.id}")
            print(f"    Symbol: {order.symbol}")
            print(f"    Qty: {order.qty}")
            print(f"    Status: {order.status}")
            
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "qty": order.qty,
                "side": normalized_side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "status": order.status,
                "submitted_at": datetime.now().isoformat(),
            }
        
        except Exception as e:
            print(f"  --- TRADE FAILED TO EXECUTE ---")
            print(f"  ! ERROR: {e}")
            err_text = str(e).lower()
            if "insufficient buying power" in err_text:
                return self._failure("insufficient_buying_power", detail=str(e))
            return self._failure("alpaca_submit_failed", detail=str(e))
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            print(f"  > Cancelled order {order_id}")
            return True
        except Exception as e:
            print(f"  ! Failed to cancel order {order_id}: {e}")
            return False
    
    def get_open_positions(self) -> list[Dict[str, Any]]:
        """Get all open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side,
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                }
                for pos in positions
            ]
        except Exception as e:
            print(f"  ! Failed to get positions: {e}")
            return []


def execute_trade_from_signal(
    signal: Any,  # SignalEvent from threshold_evaluator
    state: Any,  # SymbolState
    order_manager: StockOrderManager
) -> Optional[Dict[str, Any]]:
    """
    Execute trade from signal event and symbol state.
    
    Returns:
        Order dict on success; failure dict with ``error`` on reject
    """
    if hasattr(order_manager, "execute_signal_trade"):
        return order_manager.execute_signal_trade(signal, state)

    if not state.trade:
        print(f"  ! No trade plan in state for {signal.symbol}")
        return StockOrderManager._failure("no_trade_plan")
    
    return order_manager.execute_stock_trade(
        symbol=signal.symbol,
        side=state.trade.side,
        entry_price=state.trade.entry_price,
        stop_loss=state.trade.sl_price,
        take_profit=state.trade.tp_price,
        atr_5m=state.atr_5m,
    )
