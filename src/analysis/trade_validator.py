"""Optional LLM validation for trades before execution."""
from dataclasses import dataclass, field
from typing import List, Optional

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
        )
    
    # Get symbol's premarket bias
    symbol_bias = premarket_context.symbols.get(signal.symbol)
    
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
        )
    except Exception as e:
        # Validation failures should not become implicit approvals.
        print(f"LLM trade validation failed: {e}")
        return TradeValidation(
            should_execute=False,
            confidence=0,
            reasoning=f"LLM validation failed: {str(e)}",
            risk_assessment="validation_error",
        )
