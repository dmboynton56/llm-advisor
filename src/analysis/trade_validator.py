"""Optional LLM validation for trades before execution."""
from dataclasses import dataclass
from typing import Optional

from src.analysis.llm_client import LLMClient
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
    
    # Build premarket context string with ML and LLM opinions
    premarket_text = "No premarket data available"
    if symbol_bias:
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
"""
    
    schema = {
        "type": "object",
        "properties": {
            "should_execute": {"type": "boolean"},
            "confidence": {"type": "integer"},
            "reasoning": {"type": "string"},
            "risk_assessment": {"type": "string"},
        },
        "required": ["should_execute", "confidence", "reasoning", "risk_assessment"],
    }
    
    try:
        response = llm_client.call_structured(prompt, schema)
        content = response.content
        return TradeValidation(
            should_execute=bool(content.get("should_execute", False)),
            confidence=int(content.get("confidence", 0)),
            reasoning=content.get("reasoning", ""),
            risk_assessment=content.get("risk_assessment", "unknown"),
        )
    except Exception as e:
        # Graceful degradation: allow trade if LLM fails
        print(f"LLM trade validation failed: {e}")
        return TradeValidation(
            should_execute=True,  # Default to allowing trade
            confidence=50,
            reasoning=f"LLM validation failed: {str(e)}",
            risk_assessment="unknown",
        )

