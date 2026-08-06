"""Optional LLM validation for trades before execution."""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.analysis.llm_client import LLMClient, normalize_structured_content
from src.live.threshold_evaluator import SignalEvent
from src.live.state_manager import SymbolState
from src.premarket.bias_gatherer import PremarketContext


@dataclass
class TradeValidation:
    """Trade validation result from LLM."""
    should_execute: bool
    confidence: int  # 0-100
    reasoning: str
    risk_assessment: str
    veto_flags: List[str] = field(default_factory=list)
    gate_results: List[Dict[str, Any]] = field(default_factory=list)
    signal_uid: str = ""

    @property
    def verdict(self) -> str:
        """Stable, display-friendly decision label for downstream evidence."""
        return "approved" if self.should_execute else "rejected"


def _gate(
    code: str,
    status: str,
    observed_value: Any = None,
    required_value: Any = None,
    evidence: str = "",
) -> Dict[str, Any]:
    return {
        "code": code,
        "status": status,
        "observed_value": observed_value,
        "required_value": required_value,
        "evidence": evidence,
    }


def _hard_gates(signal: SignalEvent, state: SymbolState, symbol_bias: Any) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []
    entry = float(state.trade.entry_price)
    stop = float(state.trade.sl_price)
    target = float(state.trade.tp_price)
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk else 0.0
    gates.append(_gate("underlying_risk_reward", "pass" if rr >= 1.5 else "fail", rr, 1.5, "Trade-plan target divided by trade-plan stop distance."))

    if signal.setup_type.upper() == "TC":
        expected = "bullish" if signal.side == "long" else "bearish"
        htf = str(getattr(state, "htf_bias", "") or "").lower()
        gates.append(_gate("htf_alignment", "pass" if htf in (expected, "mixed", "") else "fail", htf, expected, "Trend-continuation entries must not oppose the higher-timeframe bias."))

    if symbol_bias is not None:
        error = getattr(symbol_bias, "bias_error", None)
        model_output = getattr(symbol_bias, "model_output", {})
        if isinstance(model_output, dict):
            error = error or model_output.get("error")
        available = bool(getattr(symbol_bias, "bias_available", True)) and not error
        gates.append(_gate("premarket_data_quality", "pass" if available else "fail", "available" if available else str(error or "unavailable"), "available", "A daily-bias error is not allowed to masquerade as a normal reading."))

        if available and signal.setup_type.upper() == "TC":
            ml_bias = str(getattr(symbol_bias, "daily_bias", "") or "").lower()
            expected = "bullish" if signal.side == "long" else "bearish"
            status = "pass" if ml_bias in (expected, "choppy", "") else "fail"
            gates.append(_gate("daily_bias_alignment", status, ml_bias, expected, "TC direction must agree with a directional ML daily bias; choppy is neutral."))
    return gates


def validate_trade_with_llm(
    signal: SignalEvent,
    state: SymbolState,
    premarket_context: PremarketContext,
    llm_client: LLMClient
) -> TradeValidation:
    """
    Validate trade with LLM before execution.
    
    Args:
        signal: Signal event
        state: Symbol state
        premarket_context: Premarket context
        llm_client: LLM client
        
    Returns:
        TradeValidation result
    """
    if not state.trade:
        return TradeValidation(
            should_execute=False,
            confidence=0,
            reasoning="No trade plan in state",
            risk_assessment="Unknown",
            signal_uid=getattr(signal, "signal_uid", ""),
        )
    
    # Get symbol's premarket bias
    symbol_bias = premarket_context.symbols.get(signal.symbol)
    gate_results = _hard_gates(signal, state, symbol_bias)
    failed_gates = [gate for gate in gate_results if gate.get("status") == "fail"]
    if failed_gates:
        return TradeValidation(
            should_execute=False,
            confidence=0,
            reasoning="Hard execution gate failed: " + "; ".join(str(gate.get("code")) for gate in failed_gates),
            risk_assessment="hard_veto",
            veto_flags=[str(gate.get("code")) for gate in failed_gates],
            gate_results=gate_results,
            signal_uid=getattr(signal, "signal_uid", ""),
        )
    
    if symbol_bias and isinstance(symbol_bias.model_output, dict) and symbol_bias.model_output.get("error"):
        err = symbol_bias.model_output.get("error")
        premarket_text = (
            f"ML daily bias model failed to load or run: {err}. "
            "Do not treat ML bias as authoritative; rely on news summary and technical context below.\n"
            f"News Summary: {symbol_bias.news_summary or 'None'}"
        )
    elif symbol_bias:
        ml_bias = symbol_bias.daily_bias
        ml_conf = symbol_bias.confidence
        
        # Check if LLM validation exists
        llm_validation = symbol_bias.model_output.get("llm_validation")
        if llm_validation:
            llm_bias = llm_validation.get("llm_bias", ml_bias)
            llm_conf = llm_validation.get("llm_confidence", ml_conf)
            agreement = llm_validation.get("agreement", "agree")
            reasoning = llm_validation.get("reasoning", "")
            
            premarket_text = f"""ML Model Prediction: {ml_bias} ({ml_conf}% confidence)
LLM Validation: {llm_bias} ({llm_conf}% confidence) - {agreement.upper()}
LLM Reasoning: {reasoning}

News Summary: {symbol_bias.news_summary or 'None'}"""
        else:
            premarket_text = f"""ML Model Prediction: {ml_bias} ({ml_conf}% confidence)
News Summary: {symbol_bias.news_summary or 'None'}"""
    else:
        premarket_text = "No premarket data available"
    
    prompt = f"""A trade signal has been triggered:

Symbol: {signal.symbol}
Setup: {signal.setup_type} ({signal.side})
Entry: {state.trade.entry_price}
Stop Loss: {state.trade.sl_price}
Take Profit: {state.trade.tp_price}

Technical Context:
- z-score: {state.last_z:.2f}
- ATR percentile: {state.atr_percentile:.1f}%
- HTF bias: {state.htf_bias}
- Status: {state.status}

Premarket Context:
{premarket_text}

Should we execute this trade? Analyze risk/reward and return JSON with:
- signal_uid: {getattr(signal, 'signal_uid', '')}
- planned underlying RR: {abs(state.trade.tp_price - state.trade.entry_price) / abs(state.trade.entry_price - state.trade.sl_price) if state.trade.entry_price != state.trade.sl_price else 0.0:.2f}
- hard gate evidence: {json.dumps(gate_results, sort_keys=True)}
- should_execute: boolean
- confidence: integer (0-100)
- reasoning: string explanation
- risk_assessment: string (low/medium/high)
- veto_flags: array of hard-veto codes selected only from:
  weak_trigger, htf_conflict, low_volatility, poor_risk_reward,
  event_risk, liquidity_risk, data_quality
"""
    
    schema = {
        "type": "object",
        "properties": {
            "should_execute": {"type": "boolean"},
            "confidence": {"type": "integer"},
            "reasoning": {"type": "string"},
            "risk_assessment": {"type": "string"},
            "veto_flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "weak_trigger",
                        "htf_conflict",
                        "low_volatility",
                        "poor_risk_reward",
                        "event_risk",
                        "liquidity_risk",
                        "data_quality",
                    ],
                },
            },
        },
        "required": [
            "should_execute",
            "confidence",
            "reasoning",
            "risk_assessment",
            "veto_flags",
        ],
    }
    
    try:
        response = llm_client.call_structured(prompt, schema)
        content = normalize_structured_content(response.content)
        return TradeValidation(
            should_execute=bool(content.get("should_execute", False)),
            confidence=int(content.get("confidence", 0)),
            reasoning=str(content.get("reasoning", "")),
            risk_assessment=str(content.get("risk_assessment", "unknown")),
            veto_flags=[
                str(flag)
                for flag in (content.get("veto_flags") or [])
                if isinstance(flag, str)
            ],
            gate_results=gate_results,
            signal_uid=getattr(signal, "signal_uid", ""),
        )
    except Exception as e:
        # Validation failures should not become implicit approvals.
        print(f"LLM trade validation failed: {e}")
        return TradeValidation(
            should_execute=False,
            confidence=0,
            reasoning=f"LLM validation failed: {str(e)}",
            risk_assessment="validation_error",
            gate_results=gate_results,
            signal_uid=getattr(signal, "signal_uid", ""),
        )
