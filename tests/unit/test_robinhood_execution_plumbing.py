import json
from datetime import datetime, timezone

from src.execution.execution_result_ingestor import ingest_execution_result
from src.execution.trade_plan_exporter import (
    build_trade_plan_from_payload,
    write_trade_plan,
)


def test_trade_plan_to_mock_fill_round_trip_appends_order_event(tmp_path) -> None:
    payload = {
        "signal": {
            "symbol": "SPY",
            "setup_type": "MR",
            "side": "long",
            "z_score": -2.1,
            "thresholds_used": {"mr_arm": 1.8},
        },
        "trade": {
            "setup": "MR",
            "side": "long",
            "entry_price": 512.34,
            "sl_price": 509.10,
            "tp_price": 517.20,
            "triggered_at": "2026-07-15T14:32:05Z",
            "execution_attempts": 0,
        },
        "state": {"symbol": "SPY", "htf_bias": "neutral"},
    }
    plan = build_trade_plan_from_payload(
        payload,
        account_equity=100_000,
        realized_pnl=0,
    )
    plan_path = write_trade_plan(plan, tmp_path / "trade_plans")
    result = {
        "schema": "llm_advisor_execution_result_v1",
        "trade_plan_id": plan["trade_plan_id"],
        "venue": "robinhood_agentic",
        "status": "filled",
        "symbol": "SPY",
        "side": "buy",
        "order_type": "limit",
        "limit_price": 512.34,
        "fill_price": 512.30,
        "reviewed_at": "2026-07-15T14:34:00Z",
        "submitted_at": "2026-07-15T14:35:00Z",
        "filled_at": "2026-07-15T14:35:10Z",
        "review_warnings": [],
        "approval_message": f"APPROVE {plan['trade_plan_id']}",
        "robinhood_order_id": "rh-mock-123",
    }

    events_path, archived, appended, event = ingest_execution_result(
        result, json.loads(plan_path.read_text(encoding="utf-8")), tmp_path / "processed"
    )

    assert appended is True
    assert archived.is_file()
    assert event["event_type"] == "robinhood_execution_filled"
    assert event["details"]["trade_plan_id"] == plan["trade_plan_id"]
    assert event["details"]["order"]["order_id"] == "rh-mock-123"
    written = json.loads(events_path.read_text(encoding="utf-8"))
    assert written == event

    _, _, appended_again, _ = ingest_execution_result(
        result, plan, tmp_path / "processed"
    )
    assert appended_again is False
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == 1


def test_exporter_fails_closed_without_fresh_account_pnl() -> None:
    payload = {
        "signal": {"symbol": "IWM", "setup_type": "MR", "side": "long"},
        "trade": {
            "setup": "MR",
            "side": "long",
            "entry_price": 220,
            "sl_price": 218,
            "tp_price": 224,
            "triggered_at": datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc),
        },
        "state": {"symbol": "IWM"},
    }

    plan = build_trade_plan_from_payload(payload)

    breaker = plan["risk"]["daily_loss_circuit_breaker"]
    assert breaker["triggered"] is True
    assert breaker["reason"] == "missing_robinhood_daily_pnl"
