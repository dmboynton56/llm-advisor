"""Options order manager safety and request-shape tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, PositionIntent, TimeInForce

from src.core.config import OptionsSettings, RiskSettings, Settings, TradingSettings
from src.execution.options_order_manager import OptionsOrderManager
from src.execution.options_strategy_mapper import OptionTradePlan


def _settings() -> Settings:
    return Settings(
        trading=TradingSettings(instrument="options"),
        risk=RiskSettings(),
        options=OptionsSettings(paper_only=True),
    )


def _plan() -> OptionTradePlan:
    return OptionTradePlan(
        underlying_symbol="SPY",
        option_symbol="SPY260116C00500000",
        strategy_type="single_long",
        contract_type="call",
        side="buy",
        position_intent="buy_to_open",
        qty=1,
        limit_price=2.05,
        estimated_premium=205.0,
        max_loss=205.0,
        expiration_date="2026-01-16",
        dte=14,
        strike_price=500.0,
        delta=0.45,
        implied_volatility=0.22,
        bid_price=2.00,
        ask_price=2.10,
        mid_price=2.05,
        bid_ask_spread_pct=0.0488,
        open_interest=500,
        setup_type="MR",
        signal_side="long",
        z_score=-1.0,
    )


def test_options_manager_refuses_live_when_paper_only() -> None:
    with pytest.raises(RuntimeError, match="paper-only"):
        OptionsOrderManager(paper=False, settings=_settings())


def test_execute_option_trade_submits_limit_buy_to_open() -> None:
    submitted = {}

    class FakeTradingClient:
        def submit_order(self, order_data):
            submitted["order_data"] = order_data
            return SimpleNamespace(id="opt-order-1", symbol=order_data.symbol, qty=order_data.qty, status="accepted")

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.trading_client = FakeTradingClient()

    result = manager.execute_option_trade(_plan())

    assert result is not None
    assert result["order_id"] == "opt-order-1"
    assert result["asset_class"] == "option"
    assert result["option_plan"]["underlying_symbol"] == "SPY"
    order_data = submitted["order_data"]
    assert order_data.symbol == "SPY260116C00500000"
    assert order_data.qty == 1
    assert order_data.side == OrderSide.BUY
    assert order_data.type == OrderType.LIMIT
    assert order_data.time_in_force == TimeInForce.DAY
    assert order_data.order_class == OrderClass.SIMPLE
    assert order_data.position_intent == PositionIntent.BUY_TO_OPEN
    assert order_data.limit_price == 2.05


def test_position_to_dict_reports_option_premium_per_contract() -> None:
    pos = SimpleNamespace(
        symbol="SPY260116C00500000",
        qty="2",
        market_value="520",
        cost_basis="400",
        unrealized_pl="120",
        unrealized_plpc="0.30",
        side="long",
        asset_class="option",
    )

    out = OptionsOrderManager._position_to_dict(pos)

    assert out["entry_price"] == 2.0
    assert out["current_price"] == 2.6
    assert out["unrealized_pl"] == 120.0
    assert out["unrealized_plpc"] == 0.30


def test_position_to_dict_does_not_scale_non_option_positions() -> None:
    pos = SimpleNamespace(
        symbol="SPY",
        qty="2",
        market_value="1000",
        cost_basis="900",
        unrealized_pl="100",
        unrealized_plpc="0.1111",
        side="long",
        asset_class="us_equity",
    )

    out = OptionsOrderManager._position_to_dict(pos)

    assert out["option_symbol"] is None
    assert out["asset_class"] == "us_equity"
    assert out["entry_price"] == 450.0
    assert out["current_price"] == 500.0


def test_execute_signal_trade_returns_option_candidate_diagnostics() -> None:
    diagnostics = {"reason": "all_candidates_filtered", "filter_rejections": {"spread_too_wide": 1}}

    class FakeMapper:
        last_rejection = diagnostics

        def build_trade_plan(self, **kwargs):
            return None

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.mapper = FakeMapper()
    manager.options_client = object()
    manager.get_account_equity = lambda: 100000.0

    state = SimpleNamespace(trade=object())
    signal = SimpleNamespace(symbol="SPY", side="long", entry_price=500.0)

    result = manager.execute_signal_trade(signal, state)

    assert result["success"] is False
    assert result["error"] == "no_option_candidate"
    assert result["diagnostics"] == diagnostics
