from __future__ import annotations

from unittest.mock import MagicMock

from src.execution.trade_tracker import TradeTracker


def test_close_persists_current_price_from_alpaca_position() -> None:
    storage = MagicMock()
    order_manager = MagicMock()
    order_manager.get_open_positions.side_effect = [
        [
            {
                "symbol": "IWM",
                "qty": -655.0,
                "side": "short",
                "current_price": 289.81,
                "avg_entry_price": 290.06,
                "unrealized_pl": 160.43,
            }
        ],
        [],
    ]

    tracker = TradeTracker(order_manager, storage=storage)
    tracker.register_open_trade("IWM", "order-1", 99)
    tracker.update_positions()  # open
    tracker.update_positions()  # flat → persist close

    storage.close_trade_by_pk.assert_called_once()
    _, kwargs = storage.close_trade_by_pk.call_args
    assert kwargs.get("exit_price") == 289.81
    assert kwargs.get("pnl") == 160.43


def test_close_derives_exit_price_when_current_price_missing() -> None:
    storage = MagicMock()
    order_manager = MagicMock()
    order_manager.get_open_positions.side_effect = [
        [
            {
                "symbol": "IWM",
                "qty": -655.0,
                "side": "short",
                "avg_entry_price": 290.06,
                "unrealized_pl": 163.75,
            }
        ],
        [],
    ]

    tracker = TradeTracker(order_manager, storage=storage)
    tracker.register_open_trade("IWM", "order-1", 99)
    tracker.update_positions()  # seed open position
    tracker.update_positions()  # flat → derive exit_price

    storage.close_trade_by_pk.assert_called_once()
    _, kwargs = storage.close_trade_by_pk.call_args
    assert kwargs.get("exit_price") is not None
    assert abs(float(kwargs["exit_price"]) - 289.81) < 0.05
