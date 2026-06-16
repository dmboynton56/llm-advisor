"""Alpaca options data client tests."""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

from src.data.alpaca_options_client import AlpacaOptionsClient, OptionContract


def _raw_contract(symbol: str):
    return SimpleNamespace(
        symbol=symbol,
        underlying_symbol="SPY",
        type="call",
        expiration_date=date.today() + timedelta(days=14),
        strike_price="500",
        tradable=True,
        open_interest="100",
        close_price=None,
    )


def test_fetch_contracts_paginates_above_alpaca_request_limit() -> None:
    requests = []

    class FakeTradingClient:
        def get_option_contracts(self, request):
            requests.append(request)
            if len(requests) == 1:
                return SimpleNamespace(
                    option_contracts=[_raw_contract(f"SPY_OPT_{idx}") for idx in range(100)],
                    next_page_token="page-2",
                )
            return SimpleNamespace(
                option_contracts=[_raw_contract(f"SPY_OPT_{idx}") for idx in range(100, 150)],
                next_page_token=None,
            )

    client = AlpacaOptionsClient.__new__(AlpacaOptionsClient)
    client.trading_client = FakeTradingClient()

    contracts = client.fetch_contracts(
        underlying_symbol="SPY",
        contract_type="call",
        expiration_date_gte=date.today() + timedelta(days=7),
        expiration_date_lte=date.today() + timedelta(days=21),
        strike_price_gte=450,
        strike_price_lte=550,
        limit=150,
    )

    assert len(contracts) == 150
    assert requests[0].limit == 100
    assert requests[0].page_token is None
    assert requests[1].limit == 50
    assert requests[1].page_token == "page-2"


def test_fetch_snapshots_batches_above_symbol_limit() -> None:
    requests = []

    class FakeDataClient:
        def get_option_snapshot(self, request):
            requests.append(request)
            return {
                symbol: SimpleNamespace(
                    latest_quote=SimpleNamespace(bid_price="1.00", ask_price="1.10"),
                    greeks=SimpleNamespace(delta="0.45"),
                    implied_volatility="0.20",
                )
                for symbol in request.symbol_or_symbols
            }

    client = AlpacaOptionsClient.__new__(AlpacaOptionsClient)
    client.data_client = FakeDataClient()
    client.feed = None
    contracts = [
        OptionContract(
            symbol=f"SPY_OPT_{idx}",
            underlying_symbol="SPY",
            contract_type="call",
            expiration_date=date.today() + timedelta(days=14),
            strike_price=500.0,
            tradable=True,
            open_interest=100,
        )
        for idx in range(150)
    ]

    snapshots = client.fetch_snapshots(contracts)

    assert len(snapshots) == 150
    assert len(requests) == 2
    assert len(requests[0].symbol_or_symbols) == 100
    assert len(requests[1].symbol_or_symbols) == 50
