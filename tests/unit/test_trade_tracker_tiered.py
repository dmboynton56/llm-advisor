from __future__ import annotations

from datetime import datetime, timezone

from src.core.config import OptionsSettings
from src.execution.trade_tracker import TradeTracker


class TieredFakeManager:
    def __init__(self) -> None:
        self.positions = [
            {
                "symbol": "SPY260116C00500000",
                "option_symbol": "SPY260116C00500000",
                "asset_class": "option",
                "underlying_symbol": "SPY",
                "qty": 4,
                "entry_price": 2.0,
                "current_price": 2.5,
                "unrealized_pl": 200.0,
                "unrealized_plpc": 0.25,
            }
        ]
        self.initial_stop_sent = False
        self.stop_prices = []
        self.orders = []
        self.fills = []
        self.risk_events = []

    def get_open_positions(self):
        return [dict(position) for position in self.positions]

    def ensure_protective_stops(self, positions, order_meta):
        if not self.initial_stop_sent:
            self.initial_stop_sent = True
            pos = positions[0]
            event = {
                    "event_type": "option_protective_stop_submitted",
                    "symbol": pos["symbol"],
                    "details": {
                        "actual_filled_qty": pos["qty"],
                        "actual_entry_price": pos["entry_price"],
                        "position": pos,
                        "stop_price": 1.3,
                    },
                }
            self.risk_events.append(event)
            return [
                event
            ]
        state = order_meta.get(positions[0]["symbol"], {}).get("tiered_exit_state")
        if state:
            self.stop_prices.append(state.active_stop_price)
        return []

    def pop_risk_events(self):
        events, self.risk_events = self.risk_events, []
        return events

    def close_position_quantity(self, symbol, qty, **kwargs):
        self.orders.append({"symbol": symbol, "qty": qty, **kwargs})
        self.positions[0]["qty"] -= int(qty)
        self.positions[0]["current_price"] = {
            "tp1": 2.5,
            "tp2": 3.0,
            "runner_trail": 2.5,
        }[kwargs["stage"]]
        self.fills.append(
            {
                "order_id": f"order-{len(self.fills) + 1}",
                "status": "filled",
                "filled_qty": qty,
                "filled_avg_price": self.positions[0]["current_price"],
            }
        )
        return {
            "success": True,
            "order_id": self.fills[-1]["order_id"],
            "status": "new",
        }

    def get_latest_exit_fill(self, symbol):
        return self.fills[-1] if self.fills else None

    def cancel_exit_order(self, symbol, order_id):
        return {"status": "canceled"}

    def close_position(self, symbol):
        self.positions = []
        return True


def test_new_eligible_lifecycle_uses_tp1_tp2_and_runner_once_each():
    manager = TieredFakeManager()
    tracker = TradeTracker(
        manager,
        options_settings=OptionsSettings(tiered_exit_enabled=True),
    )
    symbol = "SPY260116C00500000"
    tracker.register_open_trade(
        symbol,
        "entry-1",
        101,
        metadata={
            "asset_class": "option",
            "underlying_symbol": "SPY",
            "option_symbol": symbol,
            "tiered_candidate": True,
        },
    )
    now = datetime.now(timezone.utc)

    tracker.update_positions(now=now)
    assert [order["stage"] for order in manager.orders] == ["tp1"]
    tracker.update_positions(now=now)
    assert [order["stage"] for order in manager.orders] == ["tp1"]

    manager.positions[0]["current_price"] = 3.0
    manager.positions[0]["unrealized_plpc"] = 0.50
    tracker.update_positions(now=now)
    assert [order["stage"] for order in manager.orders] == ["tp1", "tp2"]
    tracker.update_positions(now=now)

    manager.positions[0]["current_price"] = 2.5
    manager.positions[0]["unrealized_plpc"] = 0.25
    tracker.update_positions(now=now)
    assert [order["stage"] for order in manager.orders] == ["tp1", "tp2", "runner_trail"]
