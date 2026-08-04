from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import OptionsSettings
from src.execution.tiered_exit import TieredExitState, allocate_tier_quantities
from src.analysis.option_replay import OptionMark, ReplayPolicy, replay_exit


def test_tier_allocations_match_paper_policy_examples() -> None:
    assert [allocate_tier_quantities(q) for q in (4, 5, 6, 8)] == [
        (2, 1, 1),
        (3, 1, 1),
        (3, 2, 1),
        (4, 2, 2),
    ]


def test_tiered_state_advances_only_after_confirmed_fill() -> None:
    state = TieredExitState.create(
        lifecycle_id="life-1",
        underlying_symbol="SPY",
        option_symbol="SPY260101C00500000",
        initial_qty=4,
        entry_price=2.0,
        original_stop_price=1.30,
    )
    state.begin_pending(stage="tp1", qty=2, client_order_id="client-1")
    state.apply_fill(
        stage="tp1",
        qty=1,
        exit_price=2.50,
        pending_remaining_qty=1,
    )
    assert state.stage == "pre_tp1"
    assert state.pending_qty == 1
    assert state.remaining_qty == 3
    state.apply_fill(stage="tp1", qty=1, exit_price=2.50)
    assert state.stage == "post_tp1"
    assert state.remaining_qty == 2
    assert state.active_stop_price == pytest.approx(1.90)


def test_replay_tiered_gap_and_reversal_use_one_action_per_mark() -> None:
    start = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    marks = [
        OptionMark(start, 2.00),
        OptionMark(start + timedelta(minutes=1), 3.20),  # crosses TP1 and TP2
        OptionMark(start + timedelta(minutes=2), 3.00),  # TP2 is next mark
        OptionMark(start + timedelta(minutes=3), 2.50),  # runner floor/trail
    ]
    result = replay_exit(
        marks,
        entry_price=2.0,
        qty=4,
        policy=ReplayPolicy(name="tiered", tiered=True, stop_loss_pct=0.35),
    )
    assert [fill["stage"] for fill in result.fills] == ["tp1", "tp2", "runner_trail"]
    assert result.fills[0]["qty"] == 2
    assert result.fills[1]["qty"] == 1
    assert result.fills[2]["qty"] == 1
    assert result.runner_contribution == pytest.approx(50.0)
    assert result.mfe_pct == pytest.approx(0.60)


def test_tiered_settings_are_paper_only() -> None:
    with pytest.raises(ValueError, match="PAPER"):
        OptionsSettings(tiered_exit_enabled=True, paper_only=False)


def test_tiered_stop_has_priority_over_same_mark_time_stop() -> None:
    start = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    result = replay_exit(
        [OptionMark(start, 2.0), OptionMark(start + timedelta(minutes=1), 1.2)],
        entry_price=2.0,
        qty=4,
        policy=ReplayPolicy(
            name="tiered",
            tiered=True,
            stop_loss_pct=0.35,
            max_hold_minutes=1,
        ),
    )
    assert result.exit_reason == "tiered_stop"
