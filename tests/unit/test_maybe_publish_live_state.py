"""Loop helper: backtest must never publish live_state."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from src.live.loop import maybe_publish_live_state


def test_maybe_publish_skipped_in_backtest() -> None:
    with patch("src.telemetry.live_state.build_live_state_row") as build:
        with patch("src.telemetry.live_state.publish_live_state") as publish:
            maybe_publish_live_state(
                is_backtest=True,
                trade_tracker=MagicMock(),
                order_manager=MagicMock(),
                settings=MagicMock(),
                trading_date=date(2026, 7, 16),
                loop_count=1,
                force=True,
            )
            build.assert_not_called()
            publish.assert_not_called()


def test_maybe_publish_calls_in_live_mode() -> None:
    tracker = MagicMock()
    om = MagicMock()
    settings = MagicMock()
    with patch("src.telemetry.live_state.should_publish_this_tick", return_value=True):
        with patch(
            "src.telemetry.live_state.build_live_state_row",
            return_value={"source": "paper"},
        ) as build:
            with patch("src.telemetry.live_state.publish_live_state") as publish:
                maybe_publish_live_state(
                    is_backtest=False,
                    trade_tracker=tracker,
                    order_manager=om,
                    settings=settings,
                    trading_date=date(2026, 7, 16),
                    loop_count=5,
                )
                build.assert_called_once()
                publish.assert_called_once_with({"source": "paper"})
