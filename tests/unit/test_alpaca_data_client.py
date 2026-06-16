"""Alpaca market-data wrapper resilience tests."""
from __future__ import annotations

import pytest
import requests
from datetime import datetime, timezone
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.data.alpaca_client import AlpacaDataClient, AlpacaDataUnavailable


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
