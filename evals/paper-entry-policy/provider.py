"""Exercise the real paper entry boundary against simulated broker state.

No TradingClient is constructed and no network calls are made.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import Settings, TradingSettings
from src.execution.options_order_manager import OptionsOrderManager
from src.execution.options_strategy_mapper import OptionTradePlan
from src.live.shadow_trading import evaluate_shadow_trade


def call_api(prompt: str, options: dict, context: dict) -> dict:
    scenario = context["vars"]["case"]
    underlying = scenario.get("underlying", "QQQ")
    contract_type = scenario.get("contract_type", "call")
    right = "C" if contract_type == "call" else "P"
    plan = OptionTradePlan(
        underlying_symbol=underlying,
        option_symbol=scenario.get("option_symbol", f"{underlying}261007{right}00748000"),
        strategy_type="single_long", contract_type=contract_type,
        side="buy", position_intent="buy_to_open", qty=1,
        limit_price=2.05, estimated_premium=205.0, max_loss=205.0,
        expiration_date="2026-10-07", dte=8, strike_price=748.0,
        delta=0.45 if right == "C" else -0.45, implied_volatility=0.22,
        bid_price=2.00, ask_price=2.10, mid_price=2.05,
        bid_ask_spread_pct=0.0488, open_interest=500,
        setup_type="MR", signal_side="long" if right == "C" else "short", z_score=-1.0,
    )
    trading = TradingSettings()
    if "entry_symbols" in scenario:
        trading.watchlist = scenario["entry_symbols"]
    if "monitor_symbols" in scenario:
        trading.monitor_only_symbols = scenario["monitor_symbols"]
    if scenario.get("trial") == "before_start":
        trading.experimental_paper_start = datetime.now(timezone.utc).date() + timedelta(days=1)
    elif scenario.get("trial") == "after_start":
        trading.experimental_paper_start = datetime.now(timezone.utc).date() - timedelta(days=1)

    manager = OptionsOrderManager.__new__(OptionsOrderManager)
    manager.settings = Settings(trading=trading)
    manager._stopout_cooldowns = (
        {underlying: datetime.now(timezone.utc) + timedelta(minutes=60)}
        if scenario.get("stopout") else {}
    )
    def open_positions(**kwargs):
        if scenario.get("broker_position_error"):
            raise RuntimeError("Simulated broker position query failure")
        return scenario.get("positions", [])

    manager.get_open_positions = open_positions

    def open_orders(**kwargs):
        if scenario.get("broker_order_error"):
            raise RuntimeError("Simulated broker order query failure")
        return [SimpleNamespace(**order) for order in scenario.get("orders", [])]

    submissions = []

    def submit_order(order_data):
        submissions.append(order_data)
        return SimpleNamespace(
            id="fixture-entry", symbol=order_data.symbol, qty=order_data.qty, status="accepted"
        )

    manager.get_open_orders = open_orders
    manager.trading_client = SimpleNamespace(submit_order=submit_order)
    if scenario.get("operation") == "shadow":
        signal = SimpleNamespace(symbol=underlying, setup_type="TC", side="long", signal_uid="eval-shadow")
        state = SimpleNamespace(
            trade=SimpleNamespace(entry_price=100, sl_price=99, tp_price=102, selected_option_symbols=[]),
            last_z=2.5, atr_percentile=80, htf_bias="bullish", status="tc_triggered",
        )
        if scenario.get("bad_rr"):
            state.trade.tp_price = 100.5
        bias = SimpleNamespace(
            bias_available=not scenario.get("missing_ml"),
            bias_error="model_load_failed" if scenario.get("missing_ml") else None,
            model_output={}, daily_bias="choppy", confidence=0, news_summary="Fixture news",
        )
        llm_client = SimpleNamespace(call_structured=lambda *args: SimpleNamespace(content={
            "should_execute": scenario.get("llm_approval", True), "confidence": 65,
            "reasoning": "Fixture verdict", "risk_assessment": "medium",
            "veto_flags": scenario.get("veto_flags", []),
        }))
        manager.mapper = SimpleNamespace(
            build_trade_plan=lambda **kwargs: None if scenario.get("no_candidate") else plan,
            last_rejection={"reason": "liquidity"},
        )
        manager.options_client = object()
        manager.get_account_equity = lambda: 100000
        manager.get_buying_power = lambda: scenario.get("buying_power", 100000)
        result = evaluate_shadow_trade(
            signal, state, SimpleNamespace(symbols={underlying: bias}),
            llm_client, manager, manager.settings,
        )
        decision = result["action"]
    else:
        result = manager.execute_option_trade(plan)
        decision = result.get("error", "entry_allowed")
    return {"output": f"{decision}:{len(submissions)}"}
