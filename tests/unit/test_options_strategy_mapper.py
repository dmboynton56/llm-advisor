"""Options strategy mapping tests."""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

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
    assert plan.selection_tier == "primary"


def test_mapper_uses_fallback_profile_when_primary_filters_candidate() -> None:
    class FakeOptionsClient:
        def __init__(self):
            self.calls = []
            self.last_candidate_diagnostics = {}

        def find_candidates(self, **kwargs):
            self.calls.append(kwargs)
            self.last_candidate_diagnostics = {
                "contracts_returned": 1,
                "snapshots_returned": 1,
                "call_number": len(self.calls),
            }
            return [_snapshot("SPY260116C00500000", delta=0.28, bid=2.30, ask=2.50, open_interest=50)]

    mapper = OptionsStrategyMapper(
        OptionsSettings(
            fallback_enabled=True,
            max_premium_per_trade=200.0,
            max_bid_ask_spread_pct=0.15,
            min_open_interest=100,
            fallback_max_premium_per_trade=300.0,
            fallback_max_bid_ask_spread_pct=0.25,
            fallback_min_open_interest=25,
            fallback_contract_limit=250,
            fallback_strike_window_pct=0.15,
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
    client = FakeOptionsClient()

    plan = mapper.build_trade_plan(signal, state=object(), options_client=client, account_equity=100000)

    assert plan is not None
    assert plan.selection_tier == "fallback"
    assert plan.qty == 1
    assert plan.limit_price == 2.45
    assert len(client.calls) == 2
    assert client.calls[0]["limit"] == 100
    assert client.calls[0]["strike_window_pct"] == 0.10
    assert client.calls[1]["limit"] == 250
    assert client.calls[1]["strike_window_pct"] == 0.15


def test_mapper_reports_both_profiles_when_fallback_also_rejects() -> None:
    class FakeOptionsClient:
        def __init__(self):
            self.last_candidate_diagnostics = {}

        def find_candidates(self, **kwargs):
            self.last_candidate_diagnostics = {
                "contracts_returned": 1,
                "snapshots_returned": 1,
                "limit": kwargs["limit"],
            }
            return [_snapshot("SPY260116C00500000", delta=0.20, bid=1.00, ask=1.05, open_interest=500)]

    mapper = OptionsStrategyMapper(
        OptionsSettings(fallback_enabled=True),
        RiskSettings(max_risk_per_trade_percent=1.0),
    )
    signal = SimpleNamespace(
        symbol="SPY",
        side="long",
        setup_type="MR",
        entry_price=500.0,
        z_score=-1.0,
    )

    assert mapper.build_trade_plan(signal, object(), FakeOptionsClient(), 100000) is None
    assert mapper.last_rejection is not None
    assert mapper.last_rejection["reason"] == "no_option_candidate_after_fallback"
    attempts = mapper.last_rejection["attempted_profiles"]
    assert [attempt["selection_tier"] for attempt in attempts] == ["primary", "fallback"]
    assert attempts[0]["filter_rejections"]["delta_out_of_range"] == 1
    assert attempts[1]["search"]["filters"]["min_delta"] == 0.25


def test_mapper_rejects_wide_spread_candidate() -> None:
    class FakeOptionsClient:
        last_candidate_diagnostics = {"contracts_returned": 1, "snapshots_returned": 1}

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
    assert mapper.last_rejection is not None
    assert mapper.last_rejection["reason"] == "all_candidates_filtered"
    assert mapper.last_rejection["filter_rejections"]["spread_too_wide"] == 1
    assert mapper.last_rejection["candidate_source"]["contracts_returned"] == 1


def test_mapper_records_risk_budget_rejection_after_price_buffer() -> None:
    class FakeOptionsClient:
        last_candidate_diagnostics = {"contracts_returned": 1, "snapshots_returned": 1}

        def find_candidates(self, **kwargs):
            return [_snapshot("SPY260116C00500000", delta=0.45, bid=1.96, ask=1.98)]

    mapper = OptionsStrategyMapper(
        OptionsSettings(
            max_premium_per_trade=200.0,
            order_price_buffer_pct=0.02,
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

    assert mapper.build_trade_plan(signal, object(), FakeOptionsClient(), 100000) is None
    assert mapper.last_rejection is not None
    assert mapper.last_rejection["reason"] == "premium_exceeds_risk_budget"
    assert mapper.last_rejection["risk_budget"] == 200.0
    assert mapper.last_rejection["best_candidate"]["option_symbol"] == "SPY260116C00500000"


def test_mapper_records_candidate_fetch_error_diagnostics() -> None:
    class BrokenOptionsClient:
        last_candidate_diagnostics = {"contracts_returned": 0}

        def find_candidates(self, **kwargs):
            raise RuntimeError("snapshot endpoint unavailable")

    mapper = OptionsStrategyMapper(OptionsSettings(), RiskSettings())
    signal = SimpleNamespace(
        symbol="SPY",
        side="long",
        setup_type="MR",
        entry_price=500.0,
        z_score=-1.0,
    )

    with pytest.raises(RuntimeError, match="snapshot endpoint unavailable"):
        mapper.build_trade_plan(signal, object(), BrokenOptionsClient(), 100000)

    assert mapper.last_rejection is not None
    assert mapper.last_rejection["reason"] == "candidate_fetch_error"
    assert mapper.last_rejection["candidate_source"]["contracts_returned"] == 0
