"""TradeTracker.session_closed accrual on both close paths."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.core.config import OptionsSettings
from src.execution.trade_tracker import TradeTracker


def test_session_closed_on_option_profit_target() -> None:
    storage = MagicMock()
    order_manager = MagicMock()
    symbol = "SPY260116C00500000"
    order_manager.get_open_positions.return_value = [
        {
            "symbol": symbol,
            "option_symbol": symbol,
            "asset_class": "option",
            "qty": 1,
            "entry_price": 2.0,
            "current_price": 2.6,
            "unrealized_pl": 60.0,
            "unrealized_plpc": 0.30,
        }
    ]
    order_manager.close_position.return_value = True

    tracker = TradeTracker(
        order_manager,
        storage=storage,
        options_settings=OptionsSettings(profit_target_pct=0.25),
    )
    tracker.register_open_trade(symbol, "o1", 7, metadata={"asset_class": "option"})
    tracker.update_positions(now=datetime.now(timezone.utc))

    closed = tracker.get_session_closed()
    assert len(closed) == 1
    assert closed[0]["symbol"] == symbol
    assert closed[0]["pnl"] == 60.0
    assert closed[0]["exit_reason"] == "option_profit_target"
    assert "closed_at" in closed[0]


def test_session_closed_on_disappeared_position() -> None:
    storage = MagicMock()
    order_manager = MagicMock()
    order_manager.get_open_positions.side_effect = [
        [
            {
                "symbol": "IWM",
                "qty": -10.0,
                "side": "short",
                "current_price": 200.0,
                "avg_entry_price": 201.0,
                "unrealized_pl": 10.0,
                "asset_class": "us_equity",
            }
        ],
        [],
    ]

    tracker = TradeTracker(order_manager, storage=storage)
    tracker.register_open_trade("IWM", "o2", 8)
    tracker.update_positions()
    tracker.update_positions()

    closed = tracker.get_session_closed()
    assert len(closed) == 1
    assert closed[0]["symbol"] == "IWM"
    assert closed[0]["pnl"] == 10.0
    assert closed[0]["exit_reason"] == "position_closed"
