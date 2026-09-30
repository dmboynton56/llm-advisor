"""Options order manager safety and request-shape tests."""
from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from dataclasses import replace

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
    manager.settings = _settings()
    manager._stopout_cooldowns = {}
    manager.get_open_positions = lambda **kwargs: []
    manager.get_open_orders = lambda **kwargs: []

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
    manager.settings = _settings()
    manager.mapper = FakeMapper()
    manager.options_client = object()
    manager.get_account_equity = lambda: 100000.0

    state = SimpleNamespace(trade=object())
    signal = SimpleNamespace(symbol="SPY", side="long", entry_price=500.0)

    result = manager.execute_signal_trade(signal, state)

    assert result["success"] is False
    assert result["error"] == "no_option_candidate"
    assert result["diagnostics"] == diagnostics


@pytest.mark.parametrize("reason", ["duplicate_option_contract", "underlying_exposure"])
def test_execute_signal_trade_stops_after_exposure_guard(reason: str) -> None:
    plans_built = []

    class FakeMapper:
        last_rejection = None

        def build_trade_plan(self, **kwargs):
            plans_built.append(_plan())
            return plans_built[-1]

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager.mapper = FakeMapper()
    manager.options_client = object()
    manager.get_account_equity = lambda: 100000.0
    manager.get_buying_power = lambda: 100000.0
    manager._entry_guard = lambda plan: {"success": False, "error": reason}
    manager.trading_client = SimpleNamespace(
        submit_order=lambda **kwargs: pytest.fail("blocked entry must never be submitted")
    )
    state = SimpleNamespace(
        trade=SimpleNamespace(
            selected_option_symbols=[],
            excluded_option_symbols=[],
            alternate_contract_attempted=False,
            guard_failure_reasons=[],
        )
    )
    signal = SimpleNamespace(symbol="SPY", side="long", entry_price=500.0)

    result = manager.execute_signal_trade(signal, state)

    assert result["success"] is False
    assert result["terminal_for_signal"] is True
    assert result["terminal_outcome"] == "execution_guard_failed"
    assert len(plans_built) == 1
    assert state.trade.alternate_contract_attempted is False
    assert state.trade.excluded_option_symbols == []
    assert state.trade.guard_failure_reasons == [reason]


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "GOOG", "TSLA", "UNKNOWN"])
def test_monitor_only_signal_never_fetches_candidates(symbol: str) -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    result = manager.execute_signal_trade(
        SimpleNamespace(symbol=symbol), SimpleNamespace(trade=object())
    )
    assert result["error"] == "monitor_only_symbol"
    assert result["terminal_for_signal"] is True


def test_ensure_protective_stop_uses_actual_position_fill() -> None:
    submitted = {}

    class FakeTradingClient:
        def get_orders(self, filter):
            return []

        def submit_order(self, order_data):
            submitted["order"] = order_data
            return SimpleNamespace(id="stop-1")

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.trading_client = FakeTradingClient()
    manager.options_settings = OptionsSettings(stop_loss_pct=0.35)
    manager._risk_events = []

    events = manager.ensure_protective_stops(
        positions=[
            {
                "symbol": "SPY260116C00500000",
                "asset_class": "option",
                "qty": 3,
                "entry_price": 2.0,
            }
        ],
        order_meta={"SPY260116C00500000": {"order_id": "entry-1"}},
    )

    request = submitted["order"]
    assert request.side == OrderSide.SELL
    assert request.type == OrderType.STOP
    assert request.position_intent == PositionIntent.SELL_TO_CLOSE
    assert request.qty == 3
    assert request.stop_price == 1.30
    assert events[0]["details"]["actual_filled_qty"] == 3
    assert events[0]["details"]["entry_order_id"] == "entry-1"


def test_partial_close_uses_exact_sell_to_close_market_request() -> None:
    submitted = {}

    class FakeTradingClient:
        def get_orders(self, filter):
            return []

        def submit_order(self, order_data):
            submitted["order"] = order_data
            return SimpleNamespace(id="tier-order-1", status="accepted")

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.trading_client = FakeTradingClient()
    manager._risk_events = []
    manager._last_exit_orders = {}

    result = manager.close_position_quantity(
        "SPY260116C00500000",
        2,
        lifecycle_id="life-1",
        stage="tp1",
        client_order_id="llma-tier-life-1-tp1-1",
    )

    assert result["success"] is True
    request = submitted["order"]
    assert request.qty == 2
    assert request.side == OrderSide.SELL
    assert request.type == OrderType.MARKET
    assert request.order_class == OrderClass.SIMPLE
    assert request.position_intent == PositionIntent.SELL_TO_CLOSE
    assert request.client_order_id == "llma-tier-life-1-tp1-1"


def test_entry_guard_blocks_same_contract_and_underlying() -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager._stopout_cooldowns = {}
    manager.get_open_orders = lambda symbols=None, **kwargs: []
    manager.get_open_positions = lambda **kwargs: [
        {
            "symbol": "SPY260116C00500000",
            "asset_class": "option",
            "qty": 1,
        }
    ]

    duplicate = manager._entry_guard(_plan())
    assert duplicate["error"] == "duplicate_option_contract"

    other_call = replace(_plan(), option_symbol="SPY260116C00510000")
    exposure = manager._entry_guard(other_call)
    assert exposure["error"] == "underlying_exposure"


@pytest.mark.parametrize("held_right,desired_right", [("C", "P"), ("P", "C"), ("C", "C"), ("P", "P")])
@pytest.mark.parametrize("pending", [False, True])
def test_entry_guard_blocks_same_underlying_in_either_direction(
    held_right: str, desired_right: str, pending: bool
) -> None:
    held = f"QQQ260930{held_right}00745000"
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager._stopout_cooldowns = {}
    manager.get_open_positions = lambda **kwargs: [] if pending else [{"symbol": held, "qty": 4}]
    manager.get_open_orders = lambda **kwargs: (
        [SimpleNamespace(symbol=held, side="buy", status="partially_filled")] if pending else []
    )
    plan = replace(
        _plan(), underlying_symbol="QQQ", option_symbol=f"QQQ261007{desired_right}00748000",
        contract_type="call" if desired_right == "C" else "put",
        signal_side="long" if desired_right == "C" else "short",
    )

    result = manager._entry_guard(plan)

    assert result["error"] == "underlying_exposure"
    assert result["existing_exposure"][0]["symbol"] == held


@pytest.mark.parametrize("side,status", [("sell", "accepted"), ("buy", "canceled"), ("buy", "rejected")])
def test_entry_guard_ignores_exit_orders_and_inactive_entries(side: str, status: str) -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager._stopout_cooldowns = {}
    manager.get_open_positions = lambda **kwargs: []
    manager.get_open_orders = lambda **kwargs: [
        SimpleNamespace(symbol="SPY260116P00510000", side=side, status=status)
    ]
    assert manager._entry_guard(_plan()) is None


def test_entry_guard_allows_another_underlying_below_capacity() -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager._stopout_cooldowns = {}
    manager.get_open_positions = lambda **kwargs: [{"symbol": "QQQ260930P00745000", "qty": 4}]
    manager.get_open_orders = lambda **kwargs: []
    assert manager._entry_guard(_plan()) is None


def test_direct_option_submission_checks_existing_underlying() -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager._stopout_cooldowns = {}
    manager.get_open_positions = lambda **kwargs: [{"symbol": "SPY260116P00510000", "qty": 1}]
    manager.get_open_orders = lambda **kwargs: []
    manager.trading_client = SimpleNamespace(
        submit_order=lambda **kwargs: pytest.fail("opposing contract must not be submitted")
    )
    result = manager.execute_option_trade(_plan())
    assert result["error"] == "underlying_exposure"
    assert result["terminal_for_signal"] is True


def test_direct_option_submission_blocks_monitor_only_underlying() -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    plan = replace(_plan(), underlying_symbol="TSLA", option_symbol="TSLA260930C00450000")
    assert manager.execute_option_trade(plan)["error"] == "monitor_only_symbol"


def test_entry_guard_fails_closed_when_open_order_query_fails() -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager.get_open_positions = lambda **kwargs: []

    def unavailable_orders(**kwargs):
        raise RuntimeError("broker unavailable")

    manager.get_open_orders = unavailable_orders
    manager.trading_client = SimpleNamespace(
        submit_order=lambda **kwargs: pytest.fail("unknown exposure must not be submitted")
    )
    assert manager.execute_option_trade(_plan())["error"] == "broker_order_query_failed"


def test_entry_guard_does_not_treat_a_failed_position_lookup_as_flat() -> None:
    class UnavailableBroker:
        def get_all_positions(self):
            raise RuntimeError("broker unavailable")

        def submit_order(self, **kwargs):
            pytest.fail("unknown position exposure must not be submitted")

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager.trading_client = UnavailableBroker()

    result = manager.execute_option_trade(_plan())

    assert result["error"] == "broker_position_query_failed"
    assert result["success"] is False


def test_entry_guard_honors_persisted_stopout_cooldown() -> None:
    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = _settings()
    manager._stopout_cooldowns = {
        "SPY": datetime.now(timezone.utc).replace(microsecond=0)
    }
    manager._stopout_cooldowns["SPY"] += timedelta(minutes=30)
    manager.get_open_orders = lambda symbols=None, **kwargs: []
    manager.get_open_positions = lambda **kwargs: []

    result = manager._entry_guard(_plan())

    assert result["error"] == "stopout_cooldown"
