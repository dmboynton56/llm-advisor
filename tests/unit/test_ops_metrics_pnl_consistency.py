"""Test that P&L and closed trade counts from ops_metrics match lifecycle queries.

This prevents the Overview (ops_metrics aggregates) from disagreeing with the
Trades page (direct lifecycle query) when both are computing closed P&L.
"""
from __future__ import annotations

import pytest

from src.analytics.ops_metrics import compute_ops_metrics, is_closed, summarize_trades


def _lifecycle(**overrides):
    """Factory for trade lifecycle rows (as from llm_advisor_trade_lifecycles)."""
    base = {
        "lifecycle_uid": "lc-001",
        "entry_order_id": "ord-001",
        "symbol": "SPY260710C00600000",
        "underlying_symbol": "SPY",
        "opened_at": "2026-08-20T14:30:00Z",
        "closed_at": "2026-08-20T15:45:00Z",
        "filled_qty": 1,
        "entry_fill_price": 2.0,
        "exit_fill_price": 2.5,
        "exit_reason": "option_profit_target",
        "realized_pnl": 50.0,
        "status": "closed",
    }
    base.update(overrides)
    return base


def test_closed_lifecycle_definition_matches_is_closed_helper():
    """The closed trades that appear in ops_metrics must match what is_closed accepts."""
    closed_lifecycle = _lifecycle()
    open_lifecycle = _lifecycle(status="open", realized_pnl=None, closed_at=None)

    assert is_closed(closed_lifecycle)
    assert not is_closed(open_lifecycle)


def test_pnl_and_count_consistency_in_ops_metrics():
    """When ops_metrics computes overall P&L and closed_trades, both must use the same set.

    This is the invariant the dashboard relies on: the Overview shows
    overall.total_pnl and overall.closed_trades together. If they measure different
    sets of trades (e.g. P&L from closed trades only, but count from all trades),
    then the dashboard shows a trust-breaking mismatch.
    """
    trades = [
        _lifecycle(lifecycle_uid="lc-1", realized_pnl=100.0),
        _lifecycle(lifecycle_uid="lc-2", realized_pnl=-50.0),
        _lifecycle(
            lifecycle_uid="lc-3",
            realized_pnl=None,
            status="open",
            closed_at=None,
        ),
    ]

    payload = compute_ops_metrics(
        trades=trades,
        order_events=[],
        account_snapshots=[],
    )

    overall = payload["overall"]
    closed_pnl = overall["total_pnl"]
    closed_count = overall["closed_trades"]

    # The P&L sum should be from the 2 closed trades only.
    assert closed_pnl == pytest.approx(50.0)
    # The closed_trades count must match the same set.
    assert closed_count == 2

    # This proves that summarize_trades filters by is_closed before computing both.


def test_ops_metrics_closed_definition_is_status_and_pnl_not_null():
    """The is_closed filter must require status='closed' AND pnl is not None.

    A trade row with status='closed' but pnl=None should not be counted as closed,
    because its P&L cannot contribute to total_pnl. Including it in closed_trades
    but not in total_pnl would create the Overview vs Trades mismatch.
    """
    closed_with_pnl = _lifecycle(status="closed", realized_pnl=100.0)
    closed_without_pnl = _lifecycle(status="closed", realized_pnl=None, closed_at=None)
    open_with_unrealized = _lifecycle(status="open", realized_pnl=None, closed_at=None)

    assert is_closed(closed_with_pnl)
    assert not is_closed(closed_without_pnl)
    assert not is_closed(open_with_unrealized)


def test_breakdown_consistency():
    """Breakdowns (by_underlying, by_side, etc.) must also filter consistently."""
    trades = [
        _lifecycle(lifecycle_uid="lc-1", underlying_symbol="SPY", realized_pnl=100.0),
        _lifecycle(
            lifecycle_uid="lc-2",
            underlying_symbol="SPY",
            realized_pnl=None,
            status="open",
            closed_at=None,
        ),
        _lifecycle(lifecycle_uid="lc-3", underlying_symbol="QQQ", realized_pnl=-30.0),
    ]

    payload = compute_ops_metrics(
        trades=trades,
        order_events=[],
        account_snapshots=[],
    )

    by_underlying = payload["breakdowns"]["by_underlying"]

    # SPY: 1 open + 1 closed = 2 trades, but closed_trades and total_pnl from 1 closed only.
    spy_stats = by_underlying["SPY"]
    assert spy_stats["trades"] == 2
    assert spy_stats["closed_trades"] == 1
    assert spy_stats["total_pnl"] == pytest.approx(100.0)

    # QQQ: 1 closed.
    qqq_stats = by_underlying["QQQ"]
    assert qqq_stats["trades"] == 1
    assert qqq_stats["closed_trades"] == 1
    assert qqq_stats["total_pnl"] == pytest.approx(-30.0)
