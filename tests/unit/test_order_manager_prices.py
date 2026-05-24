"""Bracket order price normalization for Alpaca."""
from types import SimpleNamespace

from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

from src.execution.order_manager import StockOrderManager


def test_round_price_to_penny() -> None:
    assert StockOrderManager._round_price(740.366) == 740.37
    assert StockOrderManager._round_price(278.9615) == 278.96


def test_execute_stock_trade_submits_paper_bracket_order() -> None:
    submitted = {}

    class FakeTradingClient:
        def submit_order(self, order_data):
            submitted["order_data"] = order_data
            return SimpleNamespace(
                id="paper-order-1",
                symbol=order_data.symbol,
                qty=order_data.qty,
                status="accepted",
            )

    manager = StockOrderManager.__new__(StockOrderManager)
    manager.trading_client = FakeTradingClient()
    manager.settings = SimpleNamespace(
        risk=SimpleNamespace(min_risk_reward_ratio=1.5, max_risk_per_trade_percent=1.0)
    )

    result = manager.execute_stock_trade(
        symbol="SPY",
        side="long",
        entry_price=500.0,
        stop_loss=498.333,
        take_profit=503.0,
        qty=2,
    )

    assert result is not None
    assert result["order_id"] == "paper-order-1"
    order_data = submitted["order_data"]
    assert order_data.symbol == "SPY"
    assert order_data.qty == 2
    assert order_data.side == OrderSide.BUY
    assert order_data.time_in_force == TimeInForce.DAY
    assert order_data.order_class == OrderClass.BRACKET
    assert order_data.stop_loss.stop_price == 498.33
    assert order_data.take_profit.limit_price == 503.0
