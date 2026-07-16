"""Build, validate, and persist Robinhood-reviewed equity trade plans.

This module deliberately has no Robinhood dependency.  The deterministic live
loop emits a file contract; a separately authenticated MCP client reviews and
executes it under human approval.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from src.execution.risk_calculator import max_shares_for_buying_power

TRADE_PLAN_SCHEMA = "llm_advisor_trade_plan_v1"
ALLOWED_SYMBOLS = frozenset({"SPY", "QQQ", "IWM"})
MAX_NOTIONAL_PCT = 0.02
MAX_TOTAL_EXPOSURE_PCT = 0.10
DEFAULT_EXPIRY_MINUTES = 30
DEFAULT_DAILY_LOSS_LIMIT_PCT = 0.01


class TradePlanValidationError(ValueError):
    """Raised when a candidate violates the Robinhood handoff contract."""


def _as_utc(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise TradePlanValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise TradePlanValidationError(f"{field} is required")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _positive_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TradePlanValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise TradePlanValidationError(f"{field} must be greater than zero")
    return result


def _setup_name(value: Any) -> tuple[str, str]:
    raw = str(value or "").strip().lower()
    mapping = {
        "mr": ("mean_reversion", "mr"),
        "mean_reversion": ("mean_reversion", "mr"),
        "mean-reversion": ("mean_reversion", "mr"),
        "tc": ("trend_continuation", "tc"),
        "trend_continuation": ("trend_continuation", "tc"),
        "trend-continuation": ("trend_continuation", "tc"),
    }
    if raw not in mapping:
        raise TradePlanValidationError(f"unsupported setup: {value!r}")
    return mapping[raw]


def _buy_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side not in {"buy", "long"}:
        raise TradePlanValidationError("Robinhood export supports long-equity buys only")
    return "buy"


def compute_daily_loss_circuit_breaker(
    account_equity: Optional[float],
    realized_pnl: Optional[float],
    limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT,
) -> dict[str, Any]:
    """Return a fail-closed daily-loss decision.

    ``limit_pct`` is a positive loss magnitude (``0.01`` means one percent).
    Missing account data is not treated as zero loss because this contract may
    ultimately authorize real-money execution.
    """
    try:
        limit = float(limit_pct)
    except (TypeError, ValueError) as exc:
        raise TradePlanValidationError("daily loss limit must be numeric") from exc
    if not 0 < limit <= 1:
        raise TradePlanValidationError("daily loss limit must be in (0, 1]")

    if account_equity is None or realized_pnl is None:
        return {
            "data_available": False,
            "triggered": True,
            "account_equity": None,
            "realized_pnl": None,
            "limit_pct": limit,
            "limit_amount": None,
            "reason": "missing_robinhood_daily_pnl",
        }

    equity = _positive_number(account_equity, "account_equity")
    try:
        pnl = float(realized_pnl)
    except (TypeError, ValueError) as exc:
        raise TradePlanValidationError("realized_pnl must be numeric") from exc
    if not math.isfinite(pnl):
        raise TradePlanValidationError("realized_pnl must be finite")
    limit_amount = equity * limit
    triggered = pnl <= -limit_amount
    return {
        "data_available": True,
        "triggered": triggered,
        "account_equity": equity,
        "realized_pnl": pnl,
        "limit_pct": limit,
        "limit_amount": limit_amount,
        "reason": "daily_loss_limit_reached" if triggered else None,
    }


def _gemini_multiplier(state: Any, setup_code: str) -> float:
    multiplier = getattr(state, "threshold_multiplier", None)
    if multiplier is None:
        return 1.0
    attr = "mr_arm_multiplier" if setup_code == "mr" else "tc_arm_multiplier"
    try:
        return float(getattr(multiplier, attr, 1.0))
    except (TypeError, ValueError):
        return 1.0


def make_trade_plan_id(
    triggered_at: Any,
    symbol: str,
    setup: str,
    side: str,
) -> str:
    """Create a deterministic ID that remains stable across live-loop retries."""
    triggered = _as_utc(triggered_at, "triggered_at")
    normalized_symbol = str(symbol or "").strip().upper()
    _, setup_code = _setup_name(setup)
    _buy_side(side)
    timestamp = triggered.strftime("%H%M%S%f")
    return f"{triggered.date().isoformat()}-{normalized_symbol}-{setup_code}-long-{timestamp}"


def build_trade_plan(
    signal: Any,
    state: Any,
    *,
    account_equity: Optional[float] = None,
    realized_pnl: Optional[float] = None,
    daily_loss_limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT,
    expires_in_minutes: int = DEFAULT_EXPIRY_MINUTES,
    max_notional_pct: float = MAX_NOTIONAL_PCT,
) -> dict[str, Any]:
    """Build ``llm_advisor_trade_plan_v1`` from a signal and symbol state."""
    trade = getattr(state, "trade", None)
    if trade is None:
        raise TradePlanValidationError("state.trade is required")

    symbol = str(getattr(signal, "symbol", "") or "").strip().upper()
    if symbol not in ALLOWED_SYMBOLS:
        raise TradePlanValidationError(f"symbol {symbol or '<missing>'} is not allowlisted")
    state_symbol = str(getattr(state, "symbol", symbol) or "").strip().upper()
    if state_symbol != symbol:
        raise TradePlanValidationError("signal symbol does not match state symbol")

    setup, setup_code = _setup_name(getattr(signal, "setup_type", None) or trade.setup)
    side = _buy_side(getattr(signal, "side", None) or trade.side)
    generated_at = _as_utc(trade.triggered_at, "state.trade.triggered_at")
    try:
        expiry_minutes = int(expires_in_minutes)
    except (TypeError, ValueError) as exc:
        raise TradePlanValidationError("expires_in_minutes must be an integer") from exc
    if expiry_minutes <= 0:
        raise TradePlanValidationError("expires_in_minutes must be positive")

    try:
        notional_pct = float(max_notional_pct)
    except (TypeError, ValueError) as exc:
        raise TradePlanValidationError("max_notional_pct must be numeric") from exc
    if not 0 < notional_pct <= MAX_NOTIONAL_PCT:
        raise TradePlanValidationError(f"max_notional_pct cannot exceed {MAX_NOTIONAL_PCT}")

    entry = _positive_number(trade.entry_price, "limit_price")
    stop = _positive_number(trade.sl_price, "stop_loss")
    take_profit = _positive_number(trade.tp_price, "take_profit")
    if not stop < entry < take_profit:
        raise TradePlanValidationError("long-equity prices must satisfy stop_loss < limit_price < take_profit")

    breaker = compute_daily_loss_circuit_breaker(
        account_equity,
        realized_pnl,
        limit_pct=daily_loss_limit_pct,
    )
    max_notional_amount = account_equity * notional_pct if account_equity is not None else None
    max_whole_shares = (
        max_shares_for_buying_power(float(account_equity), entry, notional_pct)
        if account_equity is not None
        else None
    )
    thresholds = getattr(signal, "thresholds_used", {})
    if not isinstance(thresholds, Mapping):
        raise TradePlanValidationError("signal.thresholds_used must be an object")

    plan = {
        "schema": TRADE_PLAN_SCHEMA,
        "trade_plan_id": make_trade_plan_id(generated_at, symbol, setup, side),
        "generated_at": _iso_z(generated_at),
        "source": "stdev_live_loop",
        "symbol": symbol,
        "setup": setup,
        "side": side,
        "instrument": "equity_long",
        "order_type": "limit",
        "limit_price": entry,
        "stop_loss": stop,
        "take_profit": take_profit,
        "max_notional_pct_of_equity": notional_pct,
        "max_total_exposure_pct_of_equity": MAX_TOTAL_EXPOSURE_PCT,
        "signal": {
            "z_score": float(getattr(signal, "z_score", 0.0)),
            "thresholds_used": dict(thresholds),
            "gemini_multiplier": _gemini_multiplier(state, setup_code),
            "premarket_bias": str(getattr(state, "htf_bias", "neutral") or "neutral"),
        },
        "risk": {
            "daily_loss_circuit_breaker": breaker,
            "max_notional_amount": max_notional_amount,
            "max_whole_shares": max_whole_shares,
        },
        "constraints": {
            "expires_at": _iso_z(generated_at + timedelta(minutes=expiry_minutes)),
            "market_session": "regular",
            "one_new_position_per_day": True,
            "human_approval_required": True,
            "approval_format": "APPROVE <trade_plan_id>",
            "execution_attempts": int(getattr(trade, "execution_attempts", 0) or 0),
        },
        "status": "pending_approval",
    }
    validate_trade_plan(plan)
    return plan


def build_trade_plan_from_payload(
    payload: Mapping[str, Any],
    **risk_inputs: Any,
) -> dict[str, Any]:
    """Build a plan from the standalone CLI's plain-JSON payload."""
    signal_data = payload.get("signal")
    trade_data = payload.get("trade")
    state_data = payload.get("state", {})
    if not isinstance(signal_data, Mapping) or not isinstance(trade_data, Mapping):
        raise TradePlanValidationError("input requires signal and trade objects")
    if not isinstance(state_data, Mapping):
        raise TradePlanValidationError("state must be an object")

    class Record:
        def __init__(self, values: Mapping[str, Any]):
            self.__dict__.update(values)

    state_values = dict(state_data)
    state_values["trade"] = Record(trade_data)
    return build_trade_plan(Record(signal_data), Record(state_values), **risk_inputs)


def validate_trade_plan(plan: Mapping[str, Any], *, now: Optional[datetime] = None) -> None:
    """Validate structural and safety invariants for a trade plan."""
    if plan.get("schema") != TRADE_PLAN_SCHEMA:
        raise TradePlanValidationError(f"schema must be {TRADE_PLAN_SCHEMA}")
    symbol = str(plan.get("symbol") or "").upper()
    if symbol not in ALLOWED_SYMBOLS:
        raise TradePlanValidationError(f"symbol {symbol or '<missing>'} is not allowlisted")
    if plan.get("side") != "buy" or plan.get("instrument") != "equity_long":
        raise TradePlanValidationError("only long-equity buy plans are valid")
    if plan.get("order_type") != "limit":
        raise TradePlanValidationError("only limit orders are valid")
    if plan.get("status") != "pending_approval":
        raise TradePlanValidationError("new plans must be pending_approval")
    if not str(plan.get("trade_plan_id") or "").startswith(f"{_as_utc(plan.get('generated_at'), 'generated_at').date()}-{symbol}-"):
        raise TradePlanValidationError("trade_plan_id does not match date and symbol")

    entry = _positive_number(plan.get("limit_price"), "limit_price")
    stop = _positive_number(plan.get("stop_loss"), "stop_loss")
    take_profit = _positive_number(plan.get("take_profit"), "take_profit")
    if not stop < entry < take_profit:
        raise TradePlanValidationError("long-equity prices must satisfy stop_loss < limit_price < take_profit")
    notional_pct = float(plan.get("max_notional_pct_of_equity", 0))
    if not 0 < notional_pct <= MAX_NOTIONAL_PCT:
        raise TradePlanValidationError("notional cap exceeds the contract maximum")
    if float(plan.get("max_total_exposure_pct_of_equity", 0)) > MAX_TOTAL_EXPOSURE_PCT:
        raise TradePlanValidationError("total exposure cap exceeds the contract maximum")

    constraints = plan.get("constraints")
    if not isinstance(constraints, Mapping):
        raise TradePlanValidationError("constraints must be an object")
    generated = _as_utc(plan.get("generated_at"), "generated_at")
    expires = _as_utc(constraints.get("expires_at"), "constraints.expires_at")
    if expires <= generated:
        raise TradePlanValidationError("expires_at must be after generated_at")
    if constraints.get("human_approval_required") is not True:
        raise TradePlanValidationError("human approval must be required")
    if now is not None and expires <= _as_utc(now, "now"):
        raise TradePlanValidationError("trade plan has expired")

    risk = plan.get("risk")
    breaker = risk.get("daily_loss_circuit_breaker") if isinstance(risk, Mapping) else None
    if not isinstance(breaker, Mapping) or not isinstance(breaker.get("triggered"), bool):
        raise TradePlanValidationError("daily-loss circuit breaker decision is required")


def write_trade_plan(
    plan: Mapping[str, Any],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one plan as ``<trade_plan_id>.json``.

    An existing identical plan is idempotent.  A conflicting payload for the
    same deterministic ID is rejected unless an operator explicitly requests
    overwrite from the standalone CLI.
    """
    validate_trade_plan(plan)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{plan['trade_plan_id']}.json"
    encoded = json.dumps(dict(plan), indent=2, sort_keys=True) + "\n"
    if destination.exists() and not overwrite:
        if destination.read_text(encoding="utf-8") == encoded:
            return destination
        raise FileExistsError(f"conflicting trade plan already exists: {destination}")
    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return destination
