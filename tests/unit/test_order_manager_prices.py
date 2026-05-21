"""Bracket order price normalization for Alpaca."""
from src.execution.order_manager import StockOrderManager


def test_round_price_to_penny() -> None:
    assert StockOrderManager._round_price(740.366) == 740.37
    assert StockOrderManager._round_price(278.9615) == 278.96
