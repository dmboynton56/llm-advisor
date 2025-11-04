"""LLM bias validation - reviews ML predictions and provides independent opinions."""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from src.analysis.llm_client import LLMClient, create_llm_client
from src.core.config import Settings
from src.premarket.bias_gatherer import PremarketContext, PremarketBias


@dataclass
class LLMBiasValidation:
    """LLM's independent bias assessment."""
    symbol: str
    llm_bias: str  # "bullish", "bearish", "choppy"
    llm_confidence: int  # 0-100
    ml_bias: str  # Original ML model prediction
    ml_confidence: int  # Original ML model confidence
    agreement: str  # "agree", "disagree", "partial"
    reasoning: str  # LLM's explanation for agreement/disagreement
    delta_confidence: float  # Difference between LLM and ML confidence


def validate_biases_with_llm(
    premarket_context: PremarketContext,
    llm_client: Optional[LLMClient] = None,
    settings: Optional[Settings] = None
) -> Dict[str, LLMBiasValidation]:
    """
    Ask LLM to review ML bias predictions and provide independent opinions.
    
    Args:
        premarket_context: Premarket context with ML predictions and news
        llm_client: Optional LLM client (creates one if not provided)
        settings: Optional settings (loads defaults if not provided)
        
    Returns:
        Dict mapping symbol -> LLMBiasValidation
    """
    if llm_client is None:
        if settings is None:
            settings = Settings.load()
        llm_client = create_llm_client(settings.llm.provider, settings.llm.model)
    
    # Build prompt with ML predictions, features, and news
    prompt = _build_validation_prompt(premarket_context)
    
    schema = {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "final_bias": {"type": "string", "enum": ["bullish", "bearish", "choppy"]},
                        "final_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                        "reasoning": {"type": "string"},
                        "delta_confidence": {"type": "number"}
                    },
                    "required": ["final_bias", "final_confidence", "reasoning", "delta_confidence"]
                }
            }
        },
        "required": ["symbols"]
    }
    
    try:
        response = llm_client.call_structured(prompt, schema)
        return _parse_validation_response(response.content, premarket_context)
    except Exception as e:
        print(f"LLM bias validation failed: {e}")
        # Return neutral validations (agree with ML)
        return _create_neutral_validations(premarket_context)


def _build_validation_prompt(premarket_context: PremarketContext) -> str:
    """Build prompt asking LLM to review ML predictions."""
    # Build ML predictions summary
    ml_summary = []
    for sym, bias in premarket_context.symbols.items():
        ml_summary.append({
            "symbol": sym,
            "ml_bias": bias.daily_bias,
            "ml_confidence": bias.confidence,
            "features": bias.model_output.get("features_snapshot", {}),
            "probabilities": bias.model_output.get("probabilities", {})
        })
    
    # Build news headlines
    macro_news = premarket_context.market_context.get("macro_news", [])
    news_headlines = []
    for article in macro_news[:10]:  # Limit to top 10
        if isinstance(article, dict):
            headline = article.get("headline", "")
            if headline:
                news_headlines.append(f"- {headline}")
        elif isinstance(article, str):
            news_headlines.append(f"- {article}")
    
    # Build per-symbol news summaries
    symbol_news = {}
    for sym, bias in premarket_context.symbols.items():
        if bias.news_summary:
            symbol_news[sym] = bias.news_summary
    
    prompt = f"""## Pre-Market Analysis Review Request

**Task:** You are a Senior Trading Analyst. Your task is to act as a qualitative check on the quantitative daily bias model. The ML biases are **priors**. You must produce an independent **posterior** using price microstructure, cross-market context, and breadth. Never copy priors. Apply explicit, quantitative adjustments (delta) and explain them briefly.

**1. Raw ML Model Predictions & Feature Data:**
This JSON contains the model's output ('bias', 'confidence') and the exact 'features_snapshot' it used for its prediction for each symbol.
```json
{json.dumps(ml_summary, indent=2)}
```

**2. Raw News Headlines:**
```text
{chr(10).join(news_headlines)}
```

**3. Symbol-Specific News Context:**
{chr(10).join(f"{sym}: {news}" for sym, news in symbol_news.items()) if symbol_news else "None"}

**Your Response:**
Analyze the data and provide a single JSON object. For each symbol, you must:

1. Acknowledge the ML model's prediction (ml_bias, ml_confidence).
2. Review the identified key values in its `features_snapshot` (e.g., a large `overnight_gap_pct`, strong `h4_mom_3bars_pct`, or a sweep of a previous low).
3. Consider the sentiment from the news headlines and symbol-specific news.
4. **Formulate your own `final_bias` and `final_confidence`**. You are explicitly allowed to disagree with the ML model if the features and news suggest a different narrative. For example:
   - If the model says 'Bullish' but there's a large overnight gap down and negative news, you might override it to 'Bearish' or 'Choppy'.
   - If the model says 'Bearish' with low confidence but features show strong momentum and positive news, you might upgrade to 'Bullish'.
5. Calculate `delta_confidence` as the difference between your final_confidence and the ML's ml_confidence (can be positive or negative).
6. Write a `reasoning` that justifies your final decision, explaining WHY you agree or disagree with the ML model by referencing both the news and the key technical features you observed.

Return JSON with structure:
{{
  "symbols": {{
    "SPY": {{
      "final_bias": "bullish|bearish|choppy",
      "final_confidence": 0-100,
      "delta_confidence": -20.5,
      "reasoning": "I agree/disagree because..."
    }},
    ...
  }}
}}
"""
    return prompt


def _parse_validation_response(
    content: Dict[str, Any],
    premarket_context: PremarketContext
) -> Dict[str, LLMBiasValidation]:
    """Parse LLM response into LLMBiasValidation objects."""
    validations = {}
    
    symbols_data = content.get("symbols", {})
    
    for symbol, bias_data in premarket_context.symbols.items():
        llm_data = symbols_data.get(symbol, {})
        
        if not llm_data:
            # No LLM response for this symbol - default to agreeing with ML
            validations[symbol] = LLMBiasValidation(
                symbol=symbol,
                llm_bias=bias_data.daily_bias,
                llm_confidence=bias_data.confidence,
                ml_bias=bias_data.daily_bias,
                ml_confidence=bias_data.confidence,
                agreement="agree",
                reasoning="No LLM response - defaulting to ML prediction",
                delta_confidence=0.0
            )
            continue
        
        llm_bias = llm_data.get("final_bias", bias_data.daily_bias).lower()
        llm_confidence = int(llm_data.get("final_confidence", bias_data.confidence))
        ml_bias = bias_data.daily_bias.lower()
        ml_confidence = bias_data.confidence
        delta_conf = float(llm_data.get("delta_confidence", llm_confidence - ml_confidence))
        
        # Determine agreement
        if llm_bias == ml_bias:
            if abs(llm_confidence - ml_confidence) <= 10:
                agreement = "agree"
            else:
                agreement = "partial"
        else:
            agreement = "disagree"
        
        validations[symbol] = LLMBiasValidation(
            symbol=symbol,
            llm_bias=llm_bias,
            llm_confidence=llm_confidence,
            ml_bias=ml_bias,
            ml_confidence=ml_confidence,
            agreement=agreement,
            reasoning=llm_data.get("reasoning", "No reasoning provided"),
            delta_confidence=delta_conf
        )
    
    return validations


def _create_neutral_validations(premarket_context: PremarketContext) -> Dict[str, LLMBiasValidation]:
    """Create neutral validations that agree with ML when LLM call fails."""
    validations = {}
    for symbol, bias_data in premarket_context.symbols.items():
        validations[symbol] = LLMBiasValidation(
            symbol=symbol,
            llm_bias=bias_data.daily_bias,
            llm_confidence=bias_data.confidence,
            ml_bias=bias_data.daily_bias,
            ml_confidence=bias_data.confidence,
            agreement="agree",
            reasoning="LLM validation failed - defaulting to ML prediction",
            delta_confidence=0.0
        )
    return validations


def enhance_premarket_context_with_llm_validation(
    premarket_context: PremarketContext,
    llm_validations: Dict[str, LLMBiasValidation]
) -> PremarketContext:
    """
    Enhance PremarketContext with LLM validation results.
    
    Updates the premarket_context field in PremarketBias objects with LLM reasoning,
    and adds validation metadata.
    """
    # Create enhanced symbol biases
    enhanced_symbols = {}
    
    for symbol, bias_data in premarket_context.symbols.items():
        validation = llm_validations.get(symbol)
        
        if validation:
            # Update premarket_context field with LLM reasoning
            enhanced_bias = PremarketBias(
                symbol=bias_data.symbol,
                daily_bias=bias_data.daily_bias,  # Keep ML bias as primary
                confidence=bias_data.confidence,  # Keep ML confidence as primary
                model_output={
                    **bias_data.model_output,
                    "llm_validation": {
                        "llm_bias": validation.llm_bias,
                        "llm_confidence": validation.llm_confidence,
                        "agreement": validation.agreement,
                        "delta_confidence": validation.delta_confidence,
                        "reasoning": validation.reasoning
                    }
                },
                news_summary=bias_data.news_summary,
                premarket_price=bias_data.premarket_price,
                premarket_context=f"LLM Opinion: {validation.llm_bias} ({validation.llm_confidence}% confidence). {validation.reasoning}"
            )
        else:
            enhanced_bias = bias_data
        
        enhanced_symbols[symbol] = enhanced_bias
    
    # Return new PremarketContext with enhanced data
    return PremarketContext(
        date=premarket_context.date,
        symbols=enhanced_symbols,
        market_context=premarket_context.market_context
    )

