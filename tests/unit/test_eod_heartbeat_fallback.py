from __future__ import annotations

import json
import sys

from scripts import run_eod_aggregate
from scripts.run_eod_aggregate import (
    RunRow,
    TradeRow,
    dedupe_runs,
    dedupe_trades,
    parse_heartbeat,
    parse_order_events,
    run_row_from_heartbeat,
    run_row_from_order_events,
)


def test_heartbeat_can_create_zero_trade_run_row(tmp_path) -> None:
    log_path = tmp_path / "live_loop_log.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "ts": "2026-05-22T16:00:28+00:00",
                "symbols": {"SPY": {}, "QQQ": {}, "IWM": {}},
                "loop_count": 124,
                "backtest": False,
                "shutdown": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    heartbeat = parse_heartbeat("2026-05-22", log_path)
    assert heartbeat is not None

    run = run_row_from_heartbeat("2026-05-22", heartbeat)
    assert run.run_date == "2026-05-22"
    assert run.total_trades == 0
    assert run.closed_trades == 0
    assert run.source_file == str(log_path)


def test_eod_dry_run_accepts_heartbeat_only_artifact(tmp_path) -> None:
    processed = tmp_path / "2026-05-22" / "processed"
    processed.mkdir(parents=True)
    (processed / "live_loop_log.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-05-22T16:00:28+00:00",
                "symbols": {"SPY": {}, "QQQ": {}, "IWM": {}},
                "loop_count": 124,
                "backtest": False,
                "shutdown": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    argv = sys.argv
    try:
        sys.argv = [
            "run_eod_aggregate.py",
            "--data-dir",
            str(tmp_path),
            "--date",
            "2026-05-22",
            "--no-bigquery",
            "--dry-run",
        ]
        run_eod_aggregate.main()
    finally:
        sys.argv = argv


def test_eod_dry_run_accepts_order_event_only_artifact(tmp_path) -> None:
    processed = tmp_path / "2026-05-22" / "processed"
    processed.mkdir(parents=True)
    (processed / "order_events.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-05-22T14:24:40+00:00",
                "event_type": "signal_detected",
                "symbol": "IWM",
                "loop_count": 46,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    argv = sys.argv
    try:
        sys.argv = [
            "run_eod_aggregate.py",
            "--data-dir",
            str(tmp_path),
            "--date",
            "2026-05-22",
            "--no-bigquery",
            "--dry-run",
        ]
        run_eod_aggregate.main()
    finally:
        sys.argv = argv


def test_parse_order_events_extracts_lifecycle_fields(tmp_path) -> None:
    events_path = tmp_path / "order_events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "ts": "2026-05-22T14:24:40+00:00",
                "event_type": "execution_failed",
                "symbol": "IWM",
                "loop_count": 46,
                "signal": {
                    "setup_type": "MR",
                    "side": "short",
                    "entry_price": 284.18,
                    "z_score": 0.15,
                },
                "details": {
                    "attempt": 1,
                    "reason": "risk_reward",
                    "order": {"order_id": "alpaca-1"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_order_events("2026-05-22", events_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.run_date == "2026-05-22"
    assert row.event_type == "execution_failed"
    assert row.symbol == "IWM"
    assert row.setup_type == "MR"
    assert row.side == "short"
    assert row.entry_price == 284.18
    assert row.order_id == "alpaca-1"
    assert row.details["reason"] == "risk_reward"


def test_order_events_can_create_zero_trade_run_row(tmp_path) -> None:
    events_path = tmp_path / "order_events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "ts": "2026-05-22T14:24:40+00:00",
                "event_type": "signal_detected",
                "symbol": "IWM",
                "loop_count": 46,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_order_events("2026-05-22", events_path)
    run = run_row_from_order_events("2026-05-22", rows)

    assert run.run_date == "2026-05-22"
    assert run.total_trades == 0
    assert run.source_file == str(events_path)


def test_dedupe_runs_prefers_artifact_final_equity_over_bq() -> None:
    artifact = RunRow(
        run_date="2026-06-02",
        total_trades=1,
        closed_trades=1,
        winning_trades=1,
        losing_trades=0,
        total_pnl=160.43,
        average_win=160.43,
        average_loss=0.0,
        final_equity=100002.92,
        return_pct=None,
        daily_return_pct=None,
        win_rate=1.0,
        source_file="session_summary.json",
    )
    bq = RunRow(
        run_date="2026-06-02",
        total_trades=1,
        closed_trades=1,
        winning_trades=1,
        losing_trades=0,
        total_pnl=160.43,
        average_win=160.43,
        average_loss=0.0,
        final_equity=None,
        return_pct=None,
        daily_return_pct=None,
        win_rate=1.0,
        source_file="bq://proj.trading_signals.trades",
    )
    merged = dedupe_runs([bq, artifact])
    assert len(merged) == 1
    assert merged[0].final_equity == 100002.92
    assert "session_summary" in merged[0].source_file


def test_dedupe_trades_prefers_row_with_exit_price() -> None:
    artifact = TradeRow(
        trade_uid="2026-06-02:abc",
        run_date="2026-06-02",
        order_id="abc",
        symbol="IWM",
        side="short",
        qty=655,
        entry_price=290.06,
        stop_loss=290.34,
        take_profit=289.64,
        entry_time="2026-06-02T14:05:25+00:00",
        exit_time="2026-06-02T14:10:00+00:00",
        exit_price=289.81,
        exit_reason="position_closed",
        pnl=160.43,
        status="closed",
        source_file="session_summary.json",
    )
    bq = TradeRow(
        trade_uid="2026-06-02:abc",
        run_date="2026-06-02",
        order_id="abc",
        symbol="IWM",
        side="short",
        qty=655,
        entry_price=290.06,
        stop_loss=290.34,
        take_profit=289.64,
        entry_time="2026-06-02T14:05:25+00:00",
        exit_time="2026-06-02T14:10:00+00:00",
        exit_price=None,
        exit_reason="position_closed",
        pnl=160.43,
        status="closed",
        source_file="bq://proj.trading_signals.trades",
    )
    merged = dedupe_trades([bq, artifact])
    assert len(merged) == 1
    assert merged[0].exit_price == 289.81
