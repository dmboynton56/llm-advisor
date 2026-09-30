"""Shadow decisions exercise real validation and entry checks, never submission."""
from types import SimpleNamespace

import pytest

from src.core.config import Settings
from src.execution.options_order_manager import OptionsOrderManager
from src.execution.options_strategy_mapper import OptionTradePlan
from src.live.shadow_trading import evaluate_shadow_trade


@pytest.mark.parametrize("block,expected", [
    (None, "would_buy"), ("validation", "skipped"), ("exposure", "skipped"),
    ("capacity", "skipped"), ("buying_power", "skipped"), ("candidate", "skipped"),
    ("broker", "error"), ("llm_parse", "error"), ("context", "error"),
])
def test_full_stock_preview_records_decision_without_order(block, expected) -> None:
    settings = Settings()
    plan = OptionTradePlan(
        underlying_symbol="TSLA", option_symbol="TSLA261009C00300000",
        strategy_type="single_long", contract_type="call", side="buy", position_intent="buy_to_open",
        qty=1, limit_price=2.05, estimated_premium=205, max_loss=205,
        expiration_date="2026-10-09", dte=10, strike_price=300, delta=0.45,
        implied_volatility=0.22, bid_price=2.00, ask_price=2.10, mid_price=2.05,
        bid_ask_spread_pct=0.0488, open_interest=500, setup_type="TC", signal_side="long", z_score=2.5,
    )
    signal = SimpleNamespace(symbol="TSLA", setup_type="TC", side="long", signal_uid="shadow-1")
    state = SimpleNamespace(
        trade=SimpleNamespace(
            entry_price=300, sl_price=299, tp_price=302,
            selected_option_symbols=[], excluded_option_symbols=[],
        ),
        last_z=2.5, atr_percentile=80, htf_bias="bullish", status="tc_triggered",
    )
    context = SimpleNamespace(symbols={"TSLA": SimpleNamespace(
        bias_available=False, bias_error="model_load_failed", model_output={"error": "model_load_failed"},
        news_summary="No material news",
    )})
    if block == "context":
        context = None
    calls = []

    def call_llm(prompt, schema):
        calls.append("validation")
        return SimpleNamespace(content={
            "should_execute": "false" if block == "llm_parse" else block != "validation",
            "confidence": 65, "reasoning": "Test verdict", "risk_assessment": "medium", "veto_flags": [],
        })

    def candidates(**kwargs):
        calls.append("options")
        return None if block == "candidate" else plan

    def positions(**kwargs):
        assert kwargs["raise_on_error"] is True
        if block == "broker":
            raise RuntimeError("broker unavailable")
        if block == "capacity":
            return [{"symbol": "SPY261009C00500000"}] * settings.trading.max_concurrent_trades
        return [{"symbol": "TSLA261002P00290000"}] if block == "exposure" else []

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = settings
    manager.options_client = object()
    manager.mapper = SimpleNamespace(build_trade_plan=candidates, last_rejection={"reason": "liquidity"})
    manager._stopout_cooldowns = {}
    manager.get_account_equity = lambda: 100000
    manager.get_buying_power = lambda: 0 if block == "buying_power" else 100000
    manager.get_open_positions = positions
    manager.get_open_orders = lambda **kwargs: []
    manager.trading_client = SimpleNamespace(
        submit_order=lambda **kwargs: pytest.fail("A shadow decision must never submit an order"),
    )
    result = evaluate_shadow_trade(signal, state, context, SimpleNamespace(call_structured=call_llm), manager, settings)
    assert result["action"] == expected
    assert result["would_execute"] is (expected == "would_buy")
    assert not settings.trading.allows_entry("TSLA")
    if expected == "would_buy":
        assert result["option_plan"]["option_symbol"] == plan.option_symbol
        assert result["option_plan"]["estimated_premium"] == 205
    if block in ("validation", "llm_parse", "context"):
        assert "options" not in calls
    if block == "exposure":
        assert result["reason"] == "underlying_exposure"
