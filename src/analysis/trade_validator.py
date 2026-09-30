"""Optional LLM validation for trades before execution."""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.analysis.llm_client import LLMClient, normalize_structured_content
from src.execution.risk_calculator import validate_risk_reward
from src.live.threshold_evaluator import SignalEvent
from src.live.state_manager import SymbolState
from src.premarket.bias_gatherer import PremarketContext

_VETO_FLAGS = (
    "weak_trigger", "htf_conflict", "low_volatility", "poor_risk_reward",
    "event_risk", "liquidity_risk", "data_quality",
)


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


def _ml_bias_error(symbol_bias: Any) -> Optional[str]:
    if symbol_bias is None:
        return "missing_premarket_symbol"
    model_output = getattr(symbol_bias, "model_output", {})
    error = getattr(symbol_bias, "bias_error", None)
    if isinstance(model_output, dict):
        error = error or model_output.get("error")
    if error or not getattr(symbol_bias, "bias_available", True):
        return str(error or "unavailable")
    return None


def _hard_gates(
    signal: SignalEvent, state: SymbolState, symbol_bias: Any, require_ml_bias: bool = True
) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []
    entry = float(state.trade.entry_price)
    stop = float(state.trade.sl_price)
    target = float(state.trade.tp_price)
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / risk if risk else 0.0
    rr_ok = validate_risk_reward(entry, stop, target, 1.5) if risk else False
    gates.append(_gate("underlying_risk_reward", "pass" if rr_ok else "fail", rr, 1.5, "Trade-plan target divided by trade-plan stop distance."))

    if signal.setup_type.upper() == "TC":
        expected = "bullish" if signal.side == "long" else "bearish"
        htf = str(getattr(state, "htf_bias", "") or "").lower()
        gates.append(_gate("htf_alignment", "pass" if htf in (expected, "mixed", "") else "fail", htf, expected, "Trend-continuation entries must not oppose the higher-timeframe bias."))

    error = _ml_bias_error(symbol_bias)
    available = error is None
    # A missing symbol record is still a data failure; only an explicitly
    # unavailable model in a recorded stock experiment is a warning.
    status = "pass" if available else "fail" if require_ml_bias or symbol_bias is None else "warn"
    gates.append(_gate(
        "premarket_data_quality", status, "available" if available else error,
        "available" if require_ml_bias else "recorded news/technical experiment",
        "Daily-bias errors remain visible. Stock experiments can use news and technical context without a trained ML model.",
    ))

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
    llm_client: LLMClient,
    require_ml_bias: bool = True,
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
    gate_results = _hard_gates(signal, state, symbol_bias, require_ml_bias)
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
    
    bias_error = _ml_bias_error(symbol_bias)
    if symbol_bias and bias_error:
        premarket_text = (
            f"Experimental stock policy: ML daily bias unavailable ({bias_error}). "
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
                    "enum": list(_VETO_FLAGS),
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
        if type(content.get("should_execute")) is not bool:
            raise ValueError("should_execute must be a JSON boolean")
        confidence = content.get("confidence")
        flags = content.get("veto_flags")
        if type(confidence) is not int or not 0 <= confidence <= 100:
            raise ValueError("confidence must be an integer from 0 to 100")
        if not isinstance(content.get("reasoning"), str) or not isinstance(content.get("risk_assessment"), str):
            raise ValueError("reasoning and risk_assessment must be strings")
        if not isinstance(flags, list) or any(flag not in _VETO_FLAGS for flag in flags):
            raise ValueError("veto_flags must be a list of known veto codes")
        return TradeValidation(
            should_execute=content["should_execute"] and not flags,
            confidence=confidence,
            reasoning=content["reasoning"],
            risk_assessment=content["risk_assessment"],
            veto_flags=flags,
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
