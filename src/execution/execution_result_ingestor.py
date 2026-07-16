"""Validate and ingest auditable Robinhood MCP execution results."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from src.execution.trade_plan_exporter import (
    TRADE_PLAN_SCHEMA,
    TradePlanValidationError,
    validate_trade_plan,
)

EXECUTION_RESULT_SCHEMA = "llm_advisor_execution_result_v1"
EXECUTION_STATUSES = frozenset(
    {
        "rejected",
        "approved_not_submitted",
        "submitted",
        "partially_filled",
        "filled",
        "cancelled",
        "failed",
    }
)
ORDER_STATUSES = frozenset({"submitted", "partially_filled", "filled", "cancelled"})

EVENT_TYPE_BY_STATUS = {
    "rejected": "robinhood_execution_rejected",
    "approved_not_submitted": "robinhood_execution_approved",
    "submitted": "robinhood_execution_submitted",
    "partially_filled": "robinhood_execution_partially_filled",
    "filled": "robinhood_execution_filled",
    "cancelled": "robinhood_execution_cancelled",
    "failed": "robinhood_execution_failed",
}


class ExecutionResultValidationError(ValueError):
    """Raised when an execution result is unsafe or malformed."""


def _as_utc(value: Any, field: str, *, required: bool = True) -> Optional[datetime]:
    if value is None or value == "":
        if required:
            raise ExecutionResultValidationError(f"{field} is required")
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ExecutionResultValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise ExecutionResultValidationError(f"{field} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_float(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionResultValidationError(f"{field} must be numeric") from exc


def validate_execution_result(
    result: Mapping[str, Any],
    *,
    trade_plan: Optional[Mapping[str, Any]] = None,
) -> None:
    if result.get("schema") != EXECUTION_RESULT_SCHEMA:
        raise ExecutionResultValidationError(f"schema must be {EXECUTION_RESULT_SCHEMA}")
    plan_id = str(result.get("trade_plan_id") or "").strip()
    if not plan_id:
        raise ExecutionResultValidationError("trade_plan_id is required")
    status = str(result.get("status") or "").strip().lower()
    if status not in EXECUTION_STATUSES:
        raise ExecutionResultValidationError(f"unsupported execution status: {status or '<missing>'}")
    if str(result.get("venue") or "").strip().lower() != "robinhood_agentic":
        raise ExecutionResultValidationError("venue must be robinhood_agentic")

    reviewed_at = _as_utc(result.get("reviewed_at"), "reviewed_at")
    submitted_at = _as_utc(result.get("submitted_at"), "submitted_at", required=False)
    filled_at = _as_utc(result.get("filled_at"), "filled_at", required=False)
    if submitted_at and submitted_at < reviewed_at:
        raise ExecutionResultValidationError("submitted_at cannot precede reviewed_at")
    if filled_at and (submitted_at is None or filled_at < submitted_at):
        raise ExecutionResultValidationError("filled_at requires and cannot precede submitted_at")

    warnings = result.get("review_warnings")
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ExecutionResultValidationError("review_warnings must be a list of strings")
    approval = str(result.get("approval_message") or "").strip()
    order_id = str(result.get("robinhood_order_id") or "").strip()
    if status in ORDER_STATUSES:
        if approval != f"APPROVE {plan_id}":
            raise ExecutionResultValidationError("placed orders require the exact APPROVE <trade_plan_id> message")
        if not order_id:
            raise ExecutionResultValidationError("placed orders require robinhood_order_id")
        if submitted_at is None:
            raise ExecutionResultValidationError("placed orders require submitted_at")
    elif status == "approved_not_submitted" and approval != f"APPROVE {plan_id}":
        raise ExecutionResultValidationError("approved_not_submitted requires the exact approval message")
    elif status == "rejected" and order_id:
        raise ExecutionResultValidationError("rejected results cannot contain a Robinhood order ID")
    if status == "filled":
        fill_price = _optional_float(result.get("fill_price"), "fill_price")
        if fill_price is None or fill_price <= 0 or filled_at is None:
            raise ExecutionResultValidationError("filled results require positive fill_price and filled_at")

    if trade_plan is not None:
        try:
            validate_trade_plan(trade_plan)
        except TradePlanValidationError as exc:
            raise ExecutionResultValidationError(f"invalid linked trade plan: {exc}") from exc
        if trade_plan.get("schema") != TRADE_PLAN_SCHEMA:
            raise ExecutionResultValidationError("linked trade plan schema mismatch")
        if str(trade_plan.get("trade_plan_id")) != plan_id:
            raise ExecutionResultValidationError("execution result does not match the linked trade plan")
        for field in ("symbol", "side", "order_type"):
            if str(result.get(field) or "") != str(trade_plan.get(field) or ""):
                raise ExecutionResultValidationError(f"execution result {field} does not match trade plan")
        result_limit = _optional_float(result.get("limit_price"), "limit_price")
        if result_limit != float(trade_plan.get("limit_price")):
            raise ExecutionResultValidationError("execution result limit_price does not match trade plan")


def build_order_event(result: Mapping[str, Any], trade_plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_result(result, trade_plan=trade_plan)
    status = str(result["status"]).lower()
    event_time = (
        result.get("filled_at")
        or result.get("submitted_at")
        or result.get("reviewed_at")
    )
    order_id = str(result.get("robinhood_order_id") or "").strip() or None
    event_uid = f"{result['trade_plan_id']}:{status}:{order_id or 'none'}"
    return {
        "ts": _as_utc(event_time, "event timestamp").isoformat(),
        "event_type": EVENT_TYPE_BY_STATUS[status],
        "symbol": trade_plan["symbol"],
        "loop_count": None,
        "signal": {
            "setup_type": trade_plan["setup"],
            "side": trade_plan["side"],
            "entry_price": trade_plan["limit_price"],
            "z_score": trade_plan.get("signal", {}).get("z_score"),
            "thresholds_used": trade_plan.get("signal", {}).get("thresholds_used", {}),
        },
        "details": {
            "venue": "robinhood_agentic",
            "event_uid": event_uid,
            "trade_plan_id": result["trade_plan_id"],
            "execution_status": status,
            "approval_message": result.get("approval_message"),
            "review_warnings": list(result.get("review_warnings", [])),
            "order": {
                "order_id": order_id,
                "status": status,
                "limit_price": result.get("limit_price"),
                "fill_price": result.get("fill_price"),
            },
            "execution_result": dict(result),
        },
    }


def _event_already_present(events_path: Path, event_uid: str) -> bool:
    if not events_path.exists():
        return False
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            details = payload.get("details") if isinstance(payload, dict) else None
            if isinstance(details, dict) and details.get("event_uid") == event_uid:
                return True
    return False


def ingest_execution_result(
    result: Mapping[str, Any],
    trade_plan: Mapping[str, Any],
    processed_dir: Path,
) -> tuple[Path, Path, bool, dict[str, Any]]:
    """Archive a result and append one idempotent order event.

    Returns ``(events_path, archived_result_path, appended, event)``.
    """
    event = build_order_event(result, trade_plan)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    result_dir = processed / "execution_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    archived = result_dir / f"{result['trade_plan_id']}.json"
    encoded = json.dumps(dict(result), indent=2, sort_keys=True) + "\n"
    if archived.exists() and archived.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"conflicting execution result already exists: {archived}")
    if not archived.exists():
        temporary = archived.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(archived)

    events_path = processed / "order_events.jsonl"
    event_uid = event["details"]["event_uid"]
    appended = not _event_already_present(events_path, event_uid)
    if appended:
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    return events_path, archived, appended, event


def copy_result_to_processed(source: Path, archived: Path) -> None:
    """Compatibility helper for callers that want to retain source metadata."""
    if source.resolve() != archived.resolve() and not archived.exists():
        shutil.copy2(source, archived)
