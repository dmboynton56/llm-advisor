"""Alpaca market-data wrapper resilience tests."""
from __future__ import annotations

import pytest
import requests
from datetime import datetime, timezone
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.data.alpaca_client import AlpacaDataClient, AlpacaDataUnavailable


def test_init_omits_sdk_retry_kwargs_when_client_does_not_support_them(monkeypatch) -> None:
    captured = {}

    class LegacyStockClient:
        def __init__(
            self,
            api_key=None,
            secret_key=None,
            oauth_token=None,
            use_basic_auth=False,
            raw_data=False,
            url_override=None,
            sandbox=False,
        ):
            captured.update(
                {
                    "api_key": api_key,
                    "secret_key": secret_key,
                    "oauth_token": oauth_token,
                    "use_basic_auth": use_basic_auth,
                    "raw_data": raw_data,
                    "url_override": url_override,
                    "sandbox": sandbox,
                }
            )
            self._session = None

    monkeypatch.setattr("src.data.alpaca_client.StockHistoricalDataClient", LegacyStockClient)

    AlpacaDataClient(api_key="key", api_secret="secret")

    assert captured == {
        "api_key": "key",
        "secret_key": "secret",
        "oauth_token": None,
        "use_basic_auth": False,
        "raw_data": False,
        "url_override": None,
        "sandbox": False,
    }


def test_fetch_bars_raises_bounded_unavailable_on_connect_timeout() -> None:
    class TimeoutBarsClient:
        def get_stock_bars(self, request):
            raise requests.exceptions.ConnectTimeout("connect timed out")

    client = AlpacaDataClient.__new__(AlpacaDataClient)
    client.client = TimeoutBarsClient()
    client.feed = DataFeed.IEX
    client.max_attempts = 1
    client.retry_wait_seconds = 0.0

    with pytest.raises(AlpacaDataUnavailable, match="Alpaca stock bars unavailable"):
        client.fetch_bars(
            ["SPY"],
            start=datetime(2026, 6, 5, 13, 30, tzinfo=timezone.utc),
            end=datetime(2026, 6, 5, 13, 31, tzinfo=timezone.utc),
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        )
