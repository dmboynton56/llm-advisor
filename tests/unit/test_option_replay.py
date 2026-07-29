from datetime import datetime, timedelta, timezone

import pytest

from src.analysis.option_replay import OptionMark, ReplayPolicy, replay_exit


def _marks():
    start = datetime(2026, 7, 29, 14, 0, tzinfo=timezone.utc)
    return [
        OptionMark(start, 2.00, 100.0),
        OptionMark(start + timedelta(minutes=1), 1.80, 99.5),
        OptionMark(start + timedelta(minutes=2), 1.25, 98.5),
        OptionMark(start + timedelta(minutes=3), 2.60, 101.5),
    ]


def test_replay_premium_stop_uses_first_crossing() -> None:
    result = replay_exit(
        _marks(),
        entry_price=2.0,
        qty=2,
        policy=ReplayPolicy(
            name="25_35",
            profit_target_pct=0.25,
            stop_loss_pct=0.35,
        ),
    )

    assert result.exit_reason == "premium_stop"
    assert result.exit_option_price == 1.25
    assert result.pnl == pytest.approx(-150.0)


def test_replay_underlying_plan_uses_contemporaneous_option_mark() -> None:
    result = replay_exit(
        _marks(),
        entry_price=2.0,
        qty=1,
        policy=ReplayPolicy(
            name="underlying",
            underlying_stop=99.0,
            underlying_target=101.0,
            underlying_side="long",
        ),
    )

    assert result.exit_reason == "underlying_stop"
    assert result.exit_option_price == 1.25
