"""Monitoring expands independently of the entry universe and ML availability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import run_premarket
from src.core.config import Settings, TradingSettings
from src.premarket.bias_gatherer import PremarketBias, PremarketContext


def test_scheduler_entry_list_still_monitors_the_first_expansion() -> None:
    trading = TradingSettings(watchlist=["SPY", "QQQ", "IWM"])
    symbols = trading.monitoring_symbols()
    assert symbols == ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOG", "TSLA"]
    assert [symbol for symbol in symbols if trading.allows_entry(symbol)] == ["SPY", "QQQ", "IWM"]


def test_explicit_entry_list_does_not_promote_monitor_symbols() -> None:
    trading = TradingSettings(watchlist=["SPY", "TSLA"])
    assert trading.allows_entry("SPY")
    assert not trading.allows_entry("TSLA")
    assert not trading.allows_entry("UNKNOWN")


def test_symbol_settings_normalize_and_allow_explicit_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATCHLIST", " spy, tsla,SPY, ")
    monkeypatch.setenv("MONITOR_ONLY_SYMBOLS", " aapl, MSFT,goog,aapl, ")
    trading = Settings.load().trading
    assert trading.monitoring_symbols() == ["SPY", "TSLA", "AAPL", "MSFT", "GOOG"]
    assert trading.allows_entry("tsla")
    assert not trading.allows_entry("AAPL")


@pytest.mark.parametrize("missing_entry_bias", [False, True])
def test_premarket_records_monitor_symbols_without_hiding_entry_bias_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_entry_bias: bool
) -> None:
    settings = Settings()
    expected = settings.trading.monitoring_symbols()
    requested = []
    alerts = []

    def fake_gather(**kwargs):
        requested.append(kwargs["symbols"])
        biases = {}
        for symbol in kwargs["symbols"]:
            available = settings.trading.allows_entry(symbol) and not (missing_entry_bias and symbol == "SPY")
            biases[symbol] = PremarketBias(
                symbol=symbol, daily_bias="neutral", confidence=0,
                model_output={} if available else {"error": "model_load_failed"},
                news_summary="Test headline", premarket_price=100.0,
                bias_available=available, needs_bias=available,
                bias_error=None if available else "model_load_failed",
            )
        return PremarketContext(date="2026-09-29", symbols=biases, market_context={})

    def fake_snapshots(**kwargs):
        requested.append(kwargs["symbols"])
        return {"symbols": [{"symbol": symbol} for symbol in kwargs["symbols"]]}

    monkeypatch.setattr(run_premarket.Settings, "load", lambda: settings)
    monkeypatch.setattr(run_premarket, "gather_premarket_bias", fake_gather)
    monkeypatch.setattr(run_premarket, "build_premarket_snapshots", fake_snapshots)
    monkeypatch.setattr(run_premarket, "send_discord_alert", alerts.append)
    monkeypatch.setattr(sys, "argv", [
        "run_premarket.py", "--date", "2026-09-29", "--symbols", "SPY", "QQQ", "IWM",
        "--output", str(tmp_path),
    ])

    run_premarket.main()

    assert requested == [expected, expected]
    artifact = json.loads((tmp_path / "premarket_context.json").read_text())
    assert list(artifact["premarket_context"]["symbols"]) == expected
    assert artifact["premarket_context"]["symbols"]["TSLA"]["bias_available"] is False
    assert [row["symbol"] for row in artifact["snapshots"]["symbols"]] == expected
    degraded_alerts = [message for message in alerts if "degraded mode" in message]
    assert len(degraded_alerts) == int(missing_entry_bias)
    if missing_entry_bias:
        assert "SPY: model_load_failed" in degraded_alerts[0]
        assert "TSLA" not in degraded_alerts[0]
