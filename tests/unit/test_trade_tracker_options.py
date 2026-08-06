"""Option position exit management tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.core.config import OptionsSettings
from src.execution.trade_tracker import TradeTracker
from src.execution.eod_position_manager import check_and_close_eod


class FakeStorage:
    def __init__(self) -> None:
        self.closed = []
        self.deleted = []
        self.entry_fills = []
        self.positions = []

    def close_trade_by_pk(self, trade_pk, exit_time, exit_price=None, pnl=None, exit_reason=""):
        self.closed.append(
            {
                "trade_pk": trade_pk,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": exit_reason,
            }
        )

    def delete_position_by_trade_pk(self, trade_pk):
        self.deleted.append(trade_pk)

    def update_trade_entry_fill(self, trade_pk, qty, entry_price):
        self.entry_fills.append(
            {"trade_pk": trade_pk, "qty": qty, "entry_price": entry_price}
        )

    def update_position(self, position):
        self.positions.append(position)


class FakeOrderManager:
    def __init__(self, positions):
        self.positions = list(positions)
        self.closed = []

    def get_open_positions(self):
        return list(self.positions)

    def close_position(self, symbol: str) -> bool:
        self.closed.append(symbol)
        self.positions = [pos for pos in self.positions if pos["symbol"] != symbol]
        return True


class ProtectiveStopFillManager(FakeOrderManager):
    def get_latest_exit_fill(self, symbol: str):
        return {
            "order_id": "stop-1",
            "status": "filled",
            "filled_qty": 1,
            "filled_avg_price": 2.20,
            "is_protective_stop": True,
        }


def _option_position(symbol: str = "SPY260116C00500000", plpc: float = 0.26):
    return {
        "symbol": symbol,
        "option_symbol": symbol,
        "asset_class": "option",
        "qty": 1,
        "entry_price": 2.00,
        "current_price": 2.52,
        "market_value": 252.0,
        "cost_basis": 200.0,
        "unrealized_pl": 52.0,
        "unrealized_plpc": plpc,
    }


def test_trade_tracker_closes_option_at_profit_target() -> None:
    storage = FakeStorage()
    manager = FakeOrderManager([_option_position(plpc=0.26)])
    tracker = TradeTracker(
        manager,
        storage=storage,
        options_settings=OptionsSettings(profit_target_pct=0.25),
    )
    tracker.register_open_trade(
        "SPY260116C00500000",
        "order-1",
        42,
        metadata={"asset_class": "option", "opened_at": datetime.now(timezone.utc)},
    )

    positions = tracker.update_positions(now=datetime.now(timezone.utc))
    assert len(positions) == 1  # close requested; broker disappearance confirms the fill
    positions = tracker.update_positions(now=datetime.now(timezone.utc))

    assert positions == []
    assert manager.closed == ["SPY260116C00500000"]
    assert storage.closed[0]["exit_reason"] == "option_profit_target"
    assert storage.closed[0]["pnl"] == 52.0
    assert storage.deleted == [42]
    events = tracker.pop_exit_events()
    requested = next(
        event for event in events if event["event_type"] == "option_exit_requested"
    )
    assert requested["details"]["reason"] == "option_profit_target"


def test_trade_tracker_closes_option_at_time_stop() -> None:
    storage = FakeStorage()
    manager = FakeOrderManager([_option_position(plpc=0.01)])
    now = datetime.now(timezone.utc)
    tracker = TradeTracker(
        manager,
        storage=storage,
        options_settings=OptionsSettings(max_hold_minutes=30, allow_overnight=False),
    )
    tracker.register_open_trade(
        "SPY260116C00500000",
        "order-1",
        42,
        metadata={"asset_class": "option", "opened_at": now - timedelta(minutes=31)},
    )

    positions = tracker.update_positions(now=now)
    assert len(positions) == 1
    positions = tracker.update_positions(now=now)

    assert positions == []
    assert manager.closed == ["SPY260116C00500000"]
    assert storage.closed[0]["exit_reason"] == "option_time_stop"


def test_trade_tracker_force_closes_option_on_entry_window_end() -> None:
    storage = FakeStorage()
    manager = FakeOrderManager([_option_position(plpc=0.01)])
    tracker = TradeTracker(
        manager,
        storage=storage,
        options_settings=OptionsSettings(),
    )
    tracker.register_open_trade(
        "SPY260116C00500000",
        "order-1",
        42,
        metadata={"asset_class": "option", "opened_at": datetime.now(timezone.utc)},
    )

    positions = tracker.update_positions(
        now=datetime.now(timezone.utc),
        force_close_options=True,
        force_close_reason="option_entry_window_close",
    )
    assert len(positions) == 1
    positions = tracker.update_positions(now=datetime.now(timezone.utc))

    assert positions == []
    assert manager.closed == ["SPY260116C00500000"]
    assert storage.closed[0]["exit_reason"] == "option_entry_window_close"


def test_entry_window_setting_does_not_force_close_by_default() -> None:
    manager = FakeOrderManager([_option_position(plpc=0.01)])
    settings = OptionsSettings(close_at_entry_window_end=False)
    tracker = TradeTracker(manager, options_settings=settings)

    positions = tracker.update_positions(
        now=datetime(2026, 5, 21, 19, 31, tzinfo=timezone.utc),
        force_close_options=settings.close_at_entry_window_end,
        force_close_reason="option_entry_window_close",
    )

    assert len(positions) == 1
    assert manager.closed == []


def test_eod_holds_seven_dte_option_and_closes_zero_dte_option() -> None:
    manager = FakeOrderManager(
        [
            _option_position("SPY260521C00740000", plpc=0.01),
            _option_position("SPY260528C00740000", plpc=0.01),
        ]
    )
    now = datetime(2026, 5, 21, 15, 50, tzinfo=timezone(timedelta(hours=-4)))

    policy_complete = check_and_close_eod(
        manager,
        None,
        now,
        "15:50",
        allow_overnight=True,
        eod_flatten_max_dte=0,
    )

    assert policy_complete
    assert manager.closed == ["SPY260521C00740000"]
    assert [pos["symbol"] for pos in manager.positions] == ["SPY260528C00740000"]


def test_time_stop_is_inert_for_future_expiry_when_overnight_allowed() -> None:
    now = datetime(2026, 5, 21, 19, 0, tzinfo=timezone.utc)
    symbol = "SPY260528C00740000"
    manager = FakeOrderManager([_option_position(symbol, plpc=0.01)])
    tracker = TradeTracker(
        manager,
        options_settings=OptionsSettings(max_hold_minutes=30, allow_overnight=True),
    )
    tracker.register_open_trade(
        symbol,
        "order-1",
        42,
        metadata={"asset_class": "option", "opened_at": now - timedelta(hours=4)},
    )

    positions = tracker.update_positions(now=now)

    assert len(positions) == 1
    assert manager.closed == []


def test_protective_stop_event_activates_actual_entry_fill() -> None:
    storage = FakeStorage()
    symbol = "SPY260116C00500000"
    tracker = TradeTracker(
        FakeOrderManager([]),
        storage=storage,
        options_settings=OptionsSettings(),
    )
    tracker.register_open_trade(
        symbol,
        "entry-1",
        42,
        metadata={
            "asset_class": "option",
            "underlying_symbol": "SPY",
            "option_symbol": symbol,
        },
    )

    tracker._persist_actual_entry_fills(
        [
            {
                "event_type": "option_protective_stop_submitted",
                "symbol": symbol,
                "details": {
                    "actual_filled_qty": 3,
                    "actual_entry_price": 2.15,
                    "position": {
                        "side": "long",
                        "current_price": 2.20,
                        "unrealized_pl": 15.0,
                    },
                },
            }
        ]
    )

    assert storage.entry_fills == [
        {"trade_pk": 42, "qty": 3, "entry_price": 2.15}
    ]
    assert storage.positions[0]["qty"] == 3
    assert storage.positions[0]["entry_price"] == 2.15


def test_profitable_protective_stop_is_labeled_as_anomaly() -> None:
    storage = FakeStorage()
    symbol = "SPY260116C00500000"
    manager = ProtectiveStopFillManager([_option_position(plpc=0.01)])
    tracker = TradeTracker(
        manager,
        storage=storage,
        options_settings=OptionsSettings(stop_loss_pct=0.35),
    )
    tracker.register_open_trade(
        symbol,
        "order-1",
        42,
        metadata={"asset_class": "option", "opened_at": datetime.now(timezone.utc)},
    )

    tracker.update_positions(now=datetime.now(timezone.utc))
    manager.positions = []
    tracker.update_positions(now=datetime.now(timezone.utc))

    assert storage.closed[0]["exit_reason"] == "option_protective_stop_anomaly"
    events = tracker.pop_exit_events()
    anomaly = next(event for event in events if event["event_type"] == "option_protective_stop_anomaly")
    assert anomaly["details"]["realized_pnl"] == 20.0
