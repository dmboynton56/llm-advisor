"""Options strategy mapping tests."""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

from src.core.config import OptionsSettings, RiskSettings
from src.data.alpaca_options_client import OptionContract, OptionGreeks, OptionQuote, OptionSnapshot
from src.execution.options_strategy_mapper import OptionsStrategyMapper


def _snapshot(
    symbol: str,
    delta: float,
    bid: float = 2.00,
    ask: float = 2.10,
    open_interest: int = 500,
) -> OptionSnapshot:
    return OptionSnapshot(
        contract=OptionContract(
            symbol=symbol,
            underlying_symbol="SPY",
            contract_type="call",
            expiration_date=date.today() + timedelta(days=14),
            strike_price=500.0,
            tradable=True,
            open_interest=open_interest,
        ),
        quote=OptionQuote(bid_price=bid, ask_price=ask),
        implied_volatility=0.22,
        greeks=OptionGreeks(delta=delta),
    )


def test_mapper_selects_liquid_delta_matched_long_call() -> None:
    class FakeOptionsClient:
        def find_candidates(self, **kwargs):
            assert kwargs["contract_type"] == "call"
            return [
                _snapshot("SPY260116C00500000", delta=0.45),
                _snapshot("SPY260116C00510000", delta=0.20),
            ]

    mapper = OptionsStrategyMapper(
        OptionsSettings(
            min_delta=0.30,
            max_delta=0.60,
            max_premium_per_trade=500.0,
            max_bid_ask_spread_pct=0.20,
            min_open_interest=100,
        ),
        RiskSettings(max_risk_per_trade_percent=1.0),
    )
    signal = SimpleNamespace(
        symbol="SPY",
        side="long",
        setup_type="MR",
        entry_price=500.0,
        z_score=-1.0,
    )

    plan = mapper.build_trade_plan(signal, state=object(), options_client=FakeOptionsClient(), account_equity=100000)

    assert plan is not None
    assert plan.option_symbol == "SPY260116C00500000"
    assert plan.contract_type == "call"
    assert plan.side == "buy"
    assert plan.position_intent == "buy_to_open"
    assert plan.qty == 2
    assert plan.limit_price == 2.09
    assert plan.max_loss == 418.0


def test_mapper_rejects_wide_spread_candidate() -> None:
    class FakeOptionsClient:
        def find_candidates(self, **kwargs):
            return [_snapshot("SPY260116P00490000", delta=-0.45, bid=1.00, ask=1.80)]

    mapper = OptionsStrategyMapper(
        OptionsSettings(max_bid_ask_spread_pct=0.20, min_open_interest=100),
        RiskSettings(),
    )
    signal = SimpleNamespace(
        symbol="SPY",
        side="short",
        setup_type="TC",
        entry_price=500.0,
        z_score=2.0,
    )

    assert mapper.build_trade_plan(signal, object(), FakeOptionsClient(), 100000) is None
