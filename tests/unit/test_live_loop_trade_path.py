"""Tests for live loop trade execution and session summary."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.live.loop import (
    append_order_event,
    build_live_session_summary,
    execute_trade,
    write_live_session_summary,
)
from src.live.state_manager import SymbolState, TradePlan
from src.live.threshold_evaluator import SignalEvent
from src.features.stdev_features import RollingStats
from config.thresholds import STDEVThresholds
from src.data.storage import Storage


def _minimal_state(symbol: str = "SPY") -> SymbolState:
    th = STDEVThresholds()
    rolling = RollingStats.from_seed([100.0, 101.0, 100.5, 99.8, 100.2], window=120)
    st = SymbolState(
        symbol=symbol,
        rolling=rolling,
        htf_bias="neutral",
        ema_slope_hourly=0.0,
        atr_percentile=50.0,
        atr_5m=1.0,
        thresholds=th,
    )
    st.trade = TradePlan(
        setup="MR",
        side="long",
        entry_price=100.0,
        sl_price=99.0,
        tp_price=102.0,
        triggered_at=datetime.now(timezone.utc),
    )
    return st


def test_execute_trade_returns_order_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_execute(signal, state, order_manager):
        return {"order_id": "ord-1", "qty": "3", "symbol": signal.symbol}

    monkeypatch.setattr(
        "src.execution.order_manager.execute_trade_from_signal",
        fake_execute,
    )
    sig = SignalEvent(
        symbol="SPY",
        setup_type="MR",
        side="long",
        entry_price=100.0,
        z_score=-0.5,
        thresholds_used={},
        timestamp=datetime.now(timezone.utc),
    )
    st = _minimal_state()
    out = execute_trade(sig, st, order_manager=object())
    assert out is not None
    assert out.get("order_id") == "ord-1"


def test_execute_trade_returns_none_without_manager() -> None:
    sig = SignalEvent(
        symbol="SPY",
        setup_type="MR",
        side="long",
        entry_price=100.0,
        z_score=-0.5,
        thresholds_used={},
        timestamp=datetime.now(timezone.utc),
    )
    st = _minimal_state()
    assert execute_trade(sig, st, None) is None


def test_build_live_session_summary_from_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    storage = Storage.create(env="dev", db_path=str(db_path))
    now = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)
    storage.save_trade(
        {
            "trade_id": "alp-1",
            "symbol": "SPY",
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 99.0,
            "take_profit": 103.0,
            "qty": 1,
            "status": "closed",
            "entry_time": now,
            "exit_time": now,
            "exit_price": 101.0,
            "pnl": 1.0,
            "exit_reason": "take_profit",
        }
    )
    summary = build_live_session_summary(
        storage=storage,
        date_str="2026-05-20",
        loop_count=10,
        session_end_reason="session_complete",
        order_manager=None,
    )
    assert summary["total_trades"] >= 1
    assert summary["closed_trades"] >= 1
    assert summary["total_pnl"] == pytest.approx(1.0)
    assert len(summary["trades"]) >= 1


def test_write_live_session_summary_degrades_when_storage_query_fails(tmp_path: Path) -> None:
    class BrokenStorage:
        def get_trades(self, *args, **kwargs):
            raise RuntimeError("warehouse unavailable")

    write_live_session_summary(
        output_dir=tmp_path,
        date_str="2026-05-22",
        loop_count=124,
        session_end_reason="session_complete",
        storage=BrokenStorage(),
        order_manager=None,
    )

    payload = json.loads((tmp_path / "session_summary.json").read_text(encoding="utf-8"))
    assert payload["summary_status"] == "degraded"
    assert payload["total_trades"] == 0
    assert payload["loop_iterations"] == 124
    assert "warehouse unavailable" in payload["summary_error"]


def test_append_order_event_writes_lifecycle_jsonl(tmp_path: Path) -> None:
    sig = SignalEvent(
        symbol="SPY",
        setup_type="MR",
        side="long",
        entry_price=100.0,
        z_score=-0.5,
        thresholds_used={"mr": 1.0},
        timestamp=datetime.now(timezone.utc),
    )
    state = _minimal_state()

    append_order_event(
        tmp_path / "order_events.jsonl",
        "execution_failed",
        "SPY",
        7,
        signal=sig,
        state=state,
        details={"reason": "risk_reward"},
    )

    line = (tmp_path / "order_events.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["event_type"] == "execution_failed"
    assert payload["symbol"] == "SPY"
    assert payload["loop_count"] == 7
    assert payload["signal"]["setup_type"] == "MR"
    assert payload["trade_plan"]["stop_loss"] == 99.0
    assert payload["details"]["reason"] == "risk_reward"
