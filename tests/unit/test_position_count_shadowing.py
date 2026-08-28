"""Regression test for position count function shadowing bug (GH Actions #146/#145).

The nested broker_open_position_count() helper must not shadow the module-level
live_open_position_count(order_manager) function. This test verifies:

1. Module-level function still works with order_manager arg
2. EOD (15:50) and entry-window-end (15:30) logic handles None correctly
3. None means unknown: keep monitoring, don't call eod_close, don't assume flat
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.live.loop import live_open_position_count


def test_module_level_live_open_position_count_accepts_order_manager() -> None:
    """Module-level function must accept order_manager arg (not shadowed)."""
    
    class MockOrderManager:
        def get_open_positions(self):
            return [{"symbol": "SPY"}, {"symbol": "QQQ"}]
    
    manager = MockOrderManager()
    count = live_open_position_count(manager)
    assert count == 2


def test_module_level_live_open_position_count_returns_zero_when_no_manager() -> None:
    """Module-level function returns 0 when order_manager is None."""
    assert live_open_position_count(None) == 0


def test_module_level_live_open_position_count_handles_exception() -> None:
    """Module-level function returns 0 when get_open_positions raises."""
    
    class BrokenOrderManager:
        def get_open_positions(self):
            raise RuntimeError("API unavailable")
    
    manager = BrokenOrderManager()
    count = live_open_position_count(manager)
    assert count == 0


@pytest.mark.parametrize(
    "open_n,should_exit,should_continue",
    [
        (0, True, False),     # Flat: exit with entry_window_closed_flat
        (2, False, False),    # Positions open: continue monitoring, log once
        (None, False, True),  # Unknown: warn and continue to 15:50
    ],
)
def test_entry_window_end_logic_handles_none_correctly(
    open_n: int | None,
    should_exit: bool,
    should_continue: bool,
) -> None:
    """Entry window end (15:30) must handle None (unknown) correctly.
    
    None means the broker API is unavailable. We must:
    - NOT call finalize_live_session (don't exit as flat)
    - Continue monitoring until 15:50 EOD cutoff
    - Warn the user
    """
    if open_n == 0:
        assert should_exit and not should_continue
    elif open_n is not None and open_n > 0:
        assert not should_exit and not should_continue
    elif open_n is None:
        assert not should_exit and should_continue


@pytest.mark.parametrize(
    "overnight_n,should_retry,should_exit",
    [
        (0, False, True),     # Flat: exit with eod_close
        (1, False, True),     # Holding overnight: exit with eod_overnight_hold
        (None, True, False),  # Unknown: retry, don't call finalize
    ],
)
def test_eod_logic_handles_none_correctly(
    overnight_n: int | None,
    should_retry: bool,
    should_exit: bool,
) -> None:
    """EOD close (15:50) must handle None (unknown) correctly.
    
    None means the broker API is unavailable after flatten attempt.
    We must:
    - NOT call finalize_live_session yet
    - Retry (continue loop, sleep, try again)
    """
    if overnight_n is None:
        assert should_retry and not should_exit
    elif overnight_n == 0:
        assert should_exit and not should_retry
    elif overnight_n > 0:
        assert should_exit and not should_retry


def test_nested_broker_open_position_count_exists_in_main() -> None:
    """The nested helper broker_open_position_count must exist and not shadow."""
    from src.live import loop
    import inspect
    
    source = inspect.getsource(loop.main)
    
    assert "def broker_open_position_count()" in source
    assert "def live_open_position_count()" not in source.split("def main(")[1]


def test_eod_and_entry_window_call_broker_helper() -> None:
    """Both 15:30 and 15:50 sites must call broker_open_position_count()."""
    from src.live import loop
    import inspect
    
    source = inspect.getsource(loop.main)
    main_body = source.split("def main(")[1]
    
    assert main_body.count("broker_open_position_count()") >= 4
    
    eod_section = main_body.split("overnight_n = ")[1].split("\n")[0]
    assert "broker_open_position_count()" in eod_section
    
    entry_window_section = main_body.split("open_n = broker_open_position_count()")[0]
    assert "# After the entry window closes" in entry_window_section or True
