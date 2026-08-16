"""Unit tests for intraday live_state publisher."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from src.core.config import OptionsSettings, Settings
from src.execution.trade_tracker import TradeTracker
from src.telemetry import live_state as live_state_mod
from src.telemetry.live_state import (
    build_live_state_row,
    publish_live_state,
    reset_publish_state_for_tests,
    should_publish_this_tick,
)


class FakeOrderManager:
    def __init__(self, positions=None, equity=100_000.0, last_equity=99_500.0):
        self.positions = list(positions or [])
        self._equity = equity
        self._last_equity = last_equity
        self.trading_client = MagicMock()
        account = MagicMock()
        account.equity = equity
        account.last_equity = last_equity
        self.trading_client.get_account.return_value = account

    def get_open_positions(self):
        return list(self.positions)

    def get_account_equity(self):
        return self._equity


def test_build_live_state_row_option_with_meta() -> None:
    symbol = "QQQ260724P00711000"
    om = FakeOrderManager(
        [
            {
                "symbol": symbol,
                "option_symbol": symbol,
                "asset_class": "option",
                "qty": 2,
                "side": "long",
                "entry_price": 8.75,
                "current_price": 10.95,
                "unrealized_pl": 440.0,
                "unrealized_plpc": 0.25143,
            }
        ]
    )
    tracker = TradeTracker(om, options_settings=OptionsSettings())
    tracker.register_open_trade(
        symbol,
        "ord-1",
        1,
        metadata={
            "underlying_symbol": "QQQ",
            "option_plan": {"setup_type": "MR"},
            "opened_at": datetime(2026, 7, 16, 13, 31, tzinfo=timezone.utc),
        },
    )
    tracker.update_positions(now=datetime(2026, 7, 16, 13, 40, tzinfo=timezone.utc))

    settings = Settings(options=OptionsSettings(stop_loss_pct=0.35, profit_target_pct=0.25))
    row = build_live_state_row(
        tracker,
        om,
        settings,
        session_date=date(2026, 7, 16),
        loop_count=12,
        now=datetime(2026, 7, 16, 13, 45, tzinfo=timezone.utc),
    )

    assert row["source"] == "paper"
    assert row["loop_count"] == 12
    assert row["equity"] == 100_000.0
    assert row["daily_pnl"] == 500.0
    assert row["open_position_count"] == 1
    assert row["unrealized_pnl"] == 440.0
    pos = row["open_positions"][0]
    assert pos["symbol"] == symbol
    assert pos["entry_order_id"] == "ord-1"
    assert pos["underlying_symbol"] == "QQQ"
    assert pos["setup_type"] == "MR"
    assert pos["dte"] == 8  # 2026-07-24 - 2026-07-16
    assert row["exit_policy"]["stop_loss_pct"] == 0.35
    assert row["exit_policy"]["profit_target_pct"] == 0.25
    assert row["session_stats"]["fills"] == 0


def test_build_live_state_row_empty_portfolio() -> None:
    om = FakeOrderManager([])
    tracker = TradeTracker(om)
    settings = Settings()
    row = build_live_state_row(
        tracker,
        om,
        settings,
        session_date=date(2026, 7, 16),
        loop_count=1,
    )
    assert row["open_position_count"] == 0
    assert row["open_positions"] == []
    assert row["unrealized_pnl"] == 0.0
    assert row["session_stats"]["wins"] == 0


def test_build_live_state_row_session_closed_stats() -> None:
    om = FakeOrderManager([])
    tracker = TradeTracker(om)
    tracker.session_closed = [
        {"symbol": "A", "pnl": 100.0, "exit_reason": "option_profit_target", "closed_at": "t1"},
        {"symbol": "B", "pnl": -40.0, "exit_reason": "option_stop_loss", "closed_at": "t2"},
    ]
    row = build_live_state_row(
        tracker,
        om,
        Settings(),
        session_date=date(2026, 7, 16),
        loop_count=3,
        session_end_reason="entry_window_closed_flat",
    )
    assert row["session_stats"]["fills"] == 2
    assert row["session_stats"]["wins"] == 1
    assert row["session_stats"]["losses"] == 1
    assert row["session_stats"]["realized_pnl"] == 60.0
    assert row["session_stats"]["session_end_reason"] == "entry_window_closed_flat"


def test_publish_live_state_merges_closed_positions_across_segments() -> None:
    reset_publish_state_for_tests()
    cursor = MagicMock()
    cursor.fetchone.return_value = (
        date(2026, 7, 16),
        {
            "fills": 1,
            "realized_pnl": 100.0,
            "wins": 1,
            "losses": 0,
            "closed": [
                {
                    "position_id": "prior-segment",
                    "symbol": "SPY260724C00700000",
                    "closed_at": "2026-07-16T14:00:00+00:00",
                    "pnl": 100.0,
                }
            ],
            "session_end_reason": "segment_handoff",
        },
    )
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    row = {
        "source": "paper",
        "session_date": "2026-07-16",
        "heartbeat_ts": "2026-07-16T15:00:00+00:00",
        "loop_count": 2,
        "equity": 99_900.0,
        "last_equity": 100_000.0,
        "daily_pnl": -100.0,
        "unrealized_pnl": 0.0,
        "open_position_count": 0,
        "open_positions": [],
        "session_stats": {
            "fills": 1,
            "realized_pnl": -40.0,
            "wins": 0,
            "losses": 1,
            "closed": [
                {
                    "position_id": "current-segment",
                    "symbol": "SPY260724P00700000",
                    "closed_at": "2026-07-16T15:00:00+00:00",
                    "pnl": -40.0,
                }
            ],
        },
        "exit_policy": {},
    }

    with patch.object(live_state_mod, "connect", return_value=conn):
        assert publish_live_state(row) is True

    live_insert = next(
        call
        for call in cursor.execute.call_args_list
        if "INSERT INTO llm_advisor_live_state" in call.args[0]
    )
    stats = json.loads(live_insert.args[1]["session_stats"])
    assert stats["fills"] == 2
    assert stats["realized_pnl"] == 60.0
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert {item["position_id"] for item in stats["closed"]} == {
        "prior-segment",
        "current-segment",
    }
    assert "session_end_reason" not in stats
    reset_publish_state_for_tests()


def test_publish_live_state_never_raises_and_self_disables() -> None:
    reset_publish_state_for_tests()
    with patch.object(live_state_mod, "connect", side_effect=RuntimeError("db down")):
        for _ in range(10):
            assert publish_live_state({"source": "paper"}) is False
        assert live_state_mod._disabled is True
        # Further calls short-circuit without connect
        assert publish_live_state({"source": "paper"}) is False
    reset_publish_state_for_tests()


def test_should_publish_this_tick_respects_interval(monkeypatch) -> None:
    monkeypatch.setenv("LIVE_STATE_PUBLISH_TICKS", "3")
    assert should_publish_this_tick(0) is True
    assert should_publish_this_tick(1) is False
    assert should_publish_this_tick(3) is True
