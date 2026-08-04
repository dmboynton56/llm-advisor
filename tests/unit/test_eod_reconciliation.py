from scripts.run_eod_aggregate import (
    AccountSnapshotRow,
    OrderEventRow,
    build_broker_reconciliations,
    build_trade_lifecycles,
)


def _event(event_type, ts, details):
    return OrderEventRow(
        event_uid=f"uid:{event_type}:{ts}",
        run_date="2026-07-29",
        event_ts=ts,
        event_type=event_type,
        symbol="SPY260806C00739000",
        loop_count=1,
        setup_type="MR",
        side="long",
        entry_price=739.0,
        z_score=-0.5,
        order_id=None,
        details=details,
        source_file="events.jsonl",
    )


def test_builds_fill_aware_lifecycle_and_daily_reconciliation() -> None:
    events = [
        _event(
            "option_protective_stop_submitted",
            "2026-07-29T17:00:00+00:00",
            {
                "entry_order_id": "entry-1",
                "stop_order_id": "stop-1",
                "actual_filled_qty": 3,
                "actual_entry_price": 6.06,
                "stop_price": 3.94,
            },
        ),
        _event(
            "option_exit_filled",
            "2026-07-29T18:02:30+00:00",
            {
                "entry_order_id": "entry-1",
                "reason": "option_profit_target",
                "actual_filled_qty": 3,
                "actual_exit_price": 7.67,
                "realized_pnl": 483,
                "exit_order": {
                    "order_id": "exit-1",
                    "status": "filled",
                    "filled_at": "2026-07-29T18:02:29+00:00",
                },
            },
        ),
    ]
    snapshots = [
        AccountSnapshotRow(
            snapshot_date="2026-07-29",
            captured_at="2026-07-29T20:00:00+00:00",
            equity=100483,
            last_equity=100000,
            buying_power=100483,
            daily_pnl=483,
            daily_pnl_pct=0.00483,
            source="alpaca_paper",
        )
    ]

    lifecycle = build_trade_lifecycles(events)[0]
    assert lifecycle.entry_order_id == "entry-1"
    assert lifecycle.exit_order_id == "exit-1"
    assert lifecycle.realized_pnl == 483
    assert lifecycle.status == "closed"

    reconciliation = build_broker_reconciliations(
        ["2026-07-29"], events, snapshots
    )[0]
    assert reconciliation.booked_realized_pnl == 483
    assert reconciliation.pnl_gap == 0
    assert reconciliation.status == "ok"


def test_partial_tier_fills_stay_on_one_lifecycle() -> None:
    events = [
        _event(
            "option_protective_stop_submitted",
            "2026-07-29T17:00:00+00:00",
            {
                "entry_order_id": "entry-tier-1",
                "stop_order_id": "stop-1",
                "actual_filled_qty": 4,
                "actual_entry_price": 2.0,
                "stop_price": 1.3,
            },
        ),
        _event(
            "option_partial_exit_filled",
            "2026-07-29T17:10:00+00:00",
            {
                "entry_order_id": "entry-tier-1",
                "stage": "tp1",
                "filled_qty": 2,
                "filled_avg_price": 2.5,
                "realized_pnl": 100,
                "tiered_exit_state": {"lifecycle_id": "101"},
            },
        ),
        _event(
            "option_exit_filled",
            "2026-07-29T18:00:00+00:00",
            {
                "entry_order_id": "entry-tier-1",
                "reason": "runner_trail",
                "actual_filled_qty": 4,
                "actual_exit_price": 2.75,
                "realized_pnl": 300,
                "exit_order": {"order_id": "exit-final", "status": "filled"},
            },
        ),
    ]
    lifecycles = build_trade_lifecycles(events)
    assert len(lifecycles) == 1
    assert lifecycles[0].realized_pnl == 300
    assert len(lifecycles[0].details["tiered_partial_fills"]) == 1
    reconciliation = build_broker_reconciliations(
        ["2026-07-29"], events, []
    )[0]
    assert reconciliation.booked_realized_pnl == 300
    assert reconciliation.lifecycle_exit_count == 1
