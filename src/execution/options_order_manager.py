"""Paper-only options order manager for STDEV signals."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from src.core.config import OptionsSettings, RiskSettings, Settings
from src.data.alpaca_options_client import AlpacaOptionsClient
from src.execution.options_strategy_mapper import OptionTradePlan, OptionsStrategyMapper
from src.utils.env_sanitize import getenv_strip

load_dotenv()


class OptionsOrderManager:
    """Executes paper option trades from stock-derived signals."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
        settings: Optional[Settings] = None,
        options_client: Optional[AlpacaOptionsClient] = None,
    ):
        self.settings = settings or Settings.load()
        self.options_settings: OptionsSettings = self.settings.options
        self.risk_settings: RiskSettings = self.settings.risk

        if self.options_settings.paper_only and not paper:
            raise RuntimeError("Options engine is paper-only; set ALPACA_PAPER_TRADING=true")

        api_key = api_key or getenv_strip("ALPACA_API_KEY")
        api_secret = api_secret or getenv_strip("ALPACA_SECRET_KEY")
        if not api_key or not api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")

        self.trading_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper)
        self.options_client = options_client or AlpacaOptionsClient(
            api_key=api_key,
            api_secret=api_secret,
            paper=paper,
            feed=self.options_settings.data_feed,
        )
        self.mapper = OptionsStrategyMapper(self.options_settings, self.risk_settings)

    @staticmethod
    def _failure(error: str, **extra: Any) -> Dict[str, Any]:
        return {"success": False, "error": error, **extra}

    def get_account_equity(self) -> float:
        account = self.trading_client.get_account()
        return float(account.equity)

    def get_buying_power(self) -> float:
        account = self.trading_client.get_account()
        return float(getattr(account, "options_buying_power", None) or account.buying_power)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        try:
            positions = self.trading_client.get_all_positions()
            return [self._position_to_dict(pos) for pos in positions]
        except Exception as exc:
            print(f"  ! Failed to get positions: {exc}")
            return []

    def close_position(self, symbol: str) -> bool:
        try:
            self.trading_client.close_position(symbol)
            print(f"  > Closed position {symbol}")
            return True
        except Exception as exc:
            print(f"  ! Failed to close position {symbol}: {exc}")
            return False

    def execute_signal_trade(self, signal: Any, state: Any) -> Optional[Dict[str, Any]]:
        """Build and submit an option order for a stock signal."""
        if not state.trade:
            return self._failure("no_trade_plan")

        try:
            account_equity = self.get_account_equity()
            plan = self.mapper.build_trade_plan(
                signal=signal,
                state=state,
                options_client=self.options_client,
                account_equity=account_equity,
            )
        except Exception as exc:
            return self._failure(
                "option_plan_failed",
                detail=str(exc),
                diagnostics=getattr(self.mapper, "last_rejection", None),
            )

        if plan is None:
            return self._failure(
                "no_option_candidate",
                diagnostics=getattr(self.mapper, "last_rejection", None),
            )

        buying_power = self.get_buying_power()
        if plan.max_loss > buying_power:
            return self._failure(
                "insufficient_options_buying_power",
                required=plan.max_loss,
                available=buying_power,
                option_plan=plan.to_dict(),
            )

        return self.execute_option_trade(plan)

    def execute_option_trade(self, plan: OptionTradePlan) -> Optional[Dict[str, Any]]:
        if plan.side != "buy" or plan.position_intent != "buy_to_open":
            return self._failure("unsupported_option_order", option_plan=plan.to_dict())

        order_request = LimitOrderRequest(
            symbol=plan.option_symbol,
            qty=plan.qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE,
            limit_price=plan.limit_price,
            position_intent=PositionIntent.BUY_TO_OPEN,
        )

        try:
            print(
                f"  > Submitting PAPER option BUY {plan.qty} {plan.option_symbol} "
                f"@ limit ${plan.limit_price:.2f} ({plan.underlying_symbol} {plan.setup_type})"
            )
            order = self.trading_client.submit_order(order_data=order_request)
            return {
                "success": True,
                "asset_class": "option",
                "order_id": order.id,
                "symbol": order.symbol,
                "underlying_symbol": plan.underlying_symbol,
                "option_symbol": plan.option_symbol,
                "qty": order.qty,
                "side": "buy",
                "status": order.status,
                "limit_price": plan.limit_price,
                "submitted_at": datetime.now().isoformat(),
                "option_plan": plan.to_dict(),
            }
        except Exception as exc:
            return self._failure(
                "alpaca_option_submit_failed",
                detail=str(exc),
                option_plan=plan.to_dict(),
            )

    @staticmethod
    def _position_to_dict(pos: Any) -> Dict[str, Any]:
        symbol = str(getattr(pos, "symbol", ""))
        qty = float(getattr(pos, "qty", 0) or 0)
        market_value = float(getattr(pos, "market_value", 0) or 0)
        cost_basis = float(getattr(pos, "cost_basis", 0) or 0)
        raw_asset_class = str(getattr(pos, "asset_class", "option"))
        is_option = "option" in raw_asset_class.lower()
        multiplier = 100.0 if is_option else 1.0
        entry_price = abs(cost_basis) / (abs(qty) * multiplier) if qty else None
        current_price = abs(market_value) / (abs(qty) * multiplier) if qty else None
        return {
            "symbol": symbol,
            "option_symbol": symbol if is_option else None,
            "qty": qty,
            "side": getattr(pos, "side", ""),
            "market_value": market_value,
            "cost_basis": cost_basis,
            "unrealized_pl": float(getattr(pos, "unrealized_pl", 0) or 0),
            "unrealized_plpc": float(getattr(pos, "unrealized_plpc", 0) or 0),
            "entry_price": entry_price,
            "current_price": current_price,
            "asset_class": "option" if is_option else raw_asset_class,
        }
