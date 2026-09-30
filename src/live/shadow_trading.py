"""Full decision previews for stock experiments; this module never places orders."""
from dataclasses import asdict
from typing import Any, Dict

from src.analysis.trade_validator import validate_trade_with_llm
from src.analysis.llm_client import LLMClient
from src.core.config import Settings
from src.execution.options_order_manager import OptionsOrderManager
from src.live.state_manager import SymbolState
from src.live.threshold_evaluator import SignalEvent
from src.premarket.bias_gatherer import PremarketContext


def evaluate_shadow_trade(
    signal: SignalEvent, state: SymbolState, premarket_context: PremarketContext | None,
    llm_client: LLMClient, order_manager: OptionsOrderManager | None, settings: Settings,
) -> Dict[str, Any]:
    """Record whether this signal would clear validation and broker entry checks.

    A preview uses actual quotes and account state. It is not a simulated fill
    and doesn't reserve capital for other hypothetical trades.
    """
    result: Dict[str, Any] = {
        "signal_uid": signal.signal_uid,
        "action": "error",
        "would_execute": False,
        "paper_start_date": (
            settings.trading.experimental_paper_start.isoformat()
            if settings.trading.experimental_paper_start else None
        ),
    }
    if not settings.llm.enable_trade_validation or not premarket_context:
        return {**result, "reason": "validation_unavailable"}
    if not order_manager or not hasattr(order_manager, "preview_signal_trade"):
        return {**result, "reason": "option_preview_unavailable"}
    try:
        validation = validate_trade_with_llm(
            signal, state, premarket_context, llm_client,
            require_ml_bias=settings.trading.requires_ml_bias(signal.symbol),
        )
        result["validation"] = {**asdict(validation), "verdict": validation.verdict}
        if not validation.should_execute:
            action = "error" if validation.risk_assessment == "validation_error" else "skipped"
            return {**result, "action": action, "reason": validation.reasoning}
        preview = order_manager.preview_signal_trade(signal, state)
        would_execute = preview.get("would_execute") is True
        entry_error = preview.get("error")
        action = "would_buy" if would_execute else "skipped"
        if entry_error in {"option_plan_failed", "broker_position_query_failed", "broker_order_query_failed"}:
            action = "error"
        return {
            **result,
            "action": action,
            "would_execute": would_execute,
            "reason": validation.reasoning if would_execute else preview.get("error", "entry_check_failed"),
            "option_plan": preview.get("option_plan"),
            "entry_checks": preview,
        }
    except Exception as exc:
        return {**result, "reason": "shadow_evaluation_failed", "error": str(exc)}
