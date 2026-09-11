"""Alpaca market-data wrapper resilience tests."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import pytest
import requests
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.data.alpaca_client import (
    AlpacaDataClient,
    AlpacaDataUnavailable,
    fetch_seed_window_bars,
    seed_fetch_deadline_seconds,
)


START = datetime(2026, 6, 5, 13, 30, tzinfo=timezone.utc)
END = datetime(2026, 6, 5, 13, 31, tzinfo=timezone.utc)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: List[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class _ScriptedWindowClient:
    def __init__(self, outcomes: List[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: List[tuple] = []

    def fetch_window_bars(self, symbols, start, end, **kwargs):
        self.calls.append((tuple(symbols), dict(kwargs)))
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _bars(symbol: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    return {symbol: {"bars_1m": [{"t": "2026-06-05T18:00:00+00:00", "c": 1.0}], "bars_5m": []}}


class _TimeoutAPIError(APIError):
    def __init__(self) -> None:
        Exception.__init__(self, '{"message":"backend request timeout"}')

    def __str__(self) -> str:
        return '{"message":"backend request timeout"}'


class _UnauthorizedAPIError(APIError):
    def __init__(self) -> None:
        Exception.__init__(self, "unauthorized")

    def __str__(self) -> str:
        return "unauthorized"


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
            start=START,
            end=END,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        )


def test_fetch_bars_retries_504_then_succeeds() -> None:
    class FlakyBarsClient:
        def __init__(self) -> None:
            self.calls = 0

        def get_stock_bars(self, request):
            self.calls += 1
            if self.calls == 1:
                raise _TimeoutAPIError()
            idx = pd.MultiIndex.from_tuples(
                [("SPY", pd.Timestamp("2026-06-05 13:30", tz="UTC"))],
                names=["symbol", "timestamp"],
            )
            df = pd.DataFrame(
                {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
                index=idx,
            )
            return type("Bars", (), {"df": df})()

    sdk = FlakyBarsClient()
    client = AlpacaDataClient.__new__(AlpacaDataClient)
    client.client = sdk
    client.feed = DataFeed.IEX
    client.max_attempts = 2
    client.retry_wait_seconds = 0.0

    result = client.fetch_bars(
        ["SPY"],
        start=START,
        end=END,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
    )

    assert sdk.calls == 2
    assert list(result["SPY"]["close"]) == [1.0]


def test_fetch_bars_does_not_retry_non_timeout_api_error() -> None:
    class AuthBarsClient:
        def get_stock_bars(self, request):
            raise _UnauthorizedAPIError()

    client = AlpacaDataClient.__new__(AlpacaDataClient)
    client.client = AuthBarsClient()
    client.feed = DataFeed.IEX
    client.max_attempts = 3
    client.retry_wait_seconds = 0.0

    with pytest.raises(APIError, match="unauthorized"):
        client.fetch_bars(
            ["SPY"],
            start=START,
            end=END,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        )


def test_fetch_window_bars_skips_unused_timeframe() -> None:
    client = AlpacaDataClient.__new__(AlpacaDataClient)
    calls = []

    def fake_fetch_bars(symbols, start, end, timeframe, adjustment=None):
        calls.append(timeframe)
        return {sym: pd.DataFrame() for sym in symbols}

    client.fetch_bars = fake_fetch_bars

    result = client.fetch_window_bars(
        ["SPY", "QQQ"],
        START,
        END,
        include_1m=True,
        include_5m=False,
    )

    assert calls == [TimeFrame(1, TimeFrameUnit.Minute)]
    assert result["SPY"]["bars_1m"] == []
    assert result["SPY"]["bars_5m"] == []
    assert result["QQQ"]["bars_5m"] == []


def test_fetch_window_bars_rejects_empty_timeframe_selection() -> None:
    client = AlpacaDataClient.__new__(AlpacaDataClient)
    with pytest.raises(ValueError, match="include_1m or include_5m"):
        client.fetch_window_bars(["SPY"], START, END, include_1m=False, include_5m=False)


def test_seed_fetch_retries_then_succeeds() -> None:
    clock = _Clock()
    unavailable = AlpacaDataUnavailable('backend request timeout')
    client = _ScriptedWindowClient([unavailable, _bars("SPY")])

    result = fetch_seed_window_bars(
        client,
        ["SPY"],
        START,
        END,
        include_1m=True,
        include_5m=False,
        deadline_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert result["SPY"]["bars_1m"][0]["c"] == 1.0
    assert clock.sleeps == [2.0]
    assert [call[0] for call in client.calls] == [("SPY",), ("SPY",)]
    assert client.calls[0][1]["include_1m"] is True
    assert client.calls[0][1]["include_5m"] is False


def test_seed_fetch_fail_closed_after_deadline() -> None:
    clock = _Clock()
    client = _ScriptedWindowClient(
        [AlpacaDataUnavailable("timeout")] * 8
    )

    with pytest.raises(AlpacaDataUnavailable, match="Alpaca seed bars unavailable after"):
        fetch_seed_window_bars(
            client,
            ["SPY"],
            START,
            END,
            deadline_seconds=5,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert clock.sleeps[0] == 2.0
    assert clock.t >= 5.0
    assert len(client.calls) >= 2


def test_seed_fetch_per_symbol_fallback_after_batch_504() -> None:
    clock = _Clock()
    batch_fail = AlpacaDataUnavailable("backend request timeout")
    client = _ScriptedWindowClient(
        [batch_fail, _bars("SPY"), _bars("QQQ")]
    )

    result = fetch_seed_window_bars(
        client,
        ["SPY", "QQQ"],
        START,
        END,
        include_1m=True,
        include_5m=False,
        deadline_seconds=30,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert result["SPY"]["bars_1m"][0]["c"] == 1.0
    assert result["QQQ"]["bars_1m"][0]["c"] == 1.0
    assert clock.sleeps == []
    assert [call[0] for call in client.calls] == [("SPY", "QQQ"), ("SPY",), ("QQQ",)]


def test_seed_fetch_deadline_env_override(monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_SEED_FETCH_DEADLINE_SECONDS", raising=False)
    assert seed_fetch_deadline_seconds() == 900.0
    monkeypatch.setenv("ALPACA_SEED_FETCH_DEADLINE_SECONDS", "120")
    assert seed_fetch_deadline_seconds() == 120.0
    monkeypatch.setenv("ALPACA_SEED_FETCH_DEADLINE_SECONDS", "nope")
    assert seed_fetch_deadline_seconds() == 900.0


def test_live_loop_startup_uses_seed_retry_helper() -> None:
    """GH#157: startup prev-day bars must not use the in-loop 2-attempt fetch."""
    from pathlib import Path

    source = Path("src/live/loop.py").read_text(encoding="utf-8")
    assert "prev_day_bars = fetch_seed_window_bars(" in source
    assert "premarket_bars = fetch_seed_window_bars(" in source
    assert "include_5m=False" in source
    assert "include_1m=False" in source

