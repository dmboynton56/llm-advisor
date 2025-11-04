"""Periodic market analysis using LLM (every 15 minutes)."""
from typing import Dict, Any
from datetime import datetime

from src.analysis.llm_client import LLMClient, create_llm_client
from src.live.state_manager import SymbolState
from src.premarket.bias_gatherer import PremarketContext
from config.thresholds import ThresholdMultiplier


class MarketAnalyzer:
    """Market analyzer that calls LLM periodically to adjust thresholds."""
    
    def __init__(self, llm_client: LLMClient, interval_minutes: int = 15):
        self.llm_client = llm_client
        self.interval_minutes = interval_minutes
    
    def analyze_market(
        self,
        states: Dict[str, SymbolState],
        premarket_context: PremarketContext,
        recent_price_action: Dict[str, Any]
    ) -> ThresholdMultiplier:
        """
        Analyze market and return threshold multipliers.
        
        Args:
            states: Current symbol states
            premarket_context: Premarket bias and news context
            recent_price_action: Recent price action data
            
        Returns:
            ThresholdMultiplier with adjustments
        """
        prompt = self._build_market_analysis_prompt(
            states=states,
            premarket_context=premarket_context,
            recent_price_action=recent_price_action
        )
        
        schema = {
            "type": "object",
            "properties": {
                "market_assessment": {"type": "string"},
                "opportunities": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "integer"},
                "threshold_multipliers": {
                    "type": "object",
                    "properties": {
                        "mr_arm_multiplier": {"type": "number"},
                        "mr_trigger_multiplier": {"type": "number"},
                        "tc_arm_multiplier": {"type": "number"},
                        "tc_trigger_multiplier": {"type": "number"},
                    },
                    "required": ["mr_arm_multiplier", "mr_trigger_multiplier", "tc_arm_multiplier", "tc_trigger_multiplier"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["market_assessment", "opportunities", "confidence", "threshold_multipliers", "reasoning"],
        }
        
        try:
            response = self.llm_client.call_structured(prompt, schema)
            return self._parse_multiplier_response(response.content)
        except Exception as e:
            # Graceful degradation: return neutral multipliers
            print(f"LLM market analysis failed: {e}")
            return ThresholdMultiplier(
                mr_arm_multiplier=1.0,
                mr_trigger_multiplier=1.0,
                tc_arm_multiplier=1.0,
                tc_trigger_multiplier=1.0,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
            )
    
    def _build_market_analysis_prompt(
        self,
        states: Dict[str, SymbolState],
        premarket_context: PremarketContext,
        recent_price_action: Dict[str, Any]
    ) -> str:
        """Build prompt for market analysis."""
        # Build technical state summary
        tech_summary = []
        for sym, state in states.items():
            tech_summary.append(
                f"- {sym}: z-score={state.last_z:.2f}, ATR percentile={state.atr_percentile:.1f}%, "
                f"HTF bias={state.htf_bias}, status={state.status}"
            )
        
        # Build premarket context summary with ML and LLM opinions
        bias_summary = []
        for sym, bias in premarket_context.symbols.items():
            ml_bias = bias.daily_bias
            ml_conf = bias.confidence
            
            # Check if LLM validation exists
            llm_validation = bias.model_output.get("llm_validation")
            if llm_validation:
                llm_bias = llm_validation.get("llm_bias", ml_bias)
                llm_conf = llm_validation.get("llm_confidence", ml_conf)
                agreement = llm_validation.get("agreement", "agree")
                
                if agreement == "disagree":
                    bias_summary.append(
                        f"- {sym}: ML={ml_bias} ({ml_conf}%) | LLM DISAGREES={llm_bias} ({llm_conf}%) | {llm_validation.get('reasoning', '')[:60]}"
                    )
                elif agreement == "partial":
                    bias_summary.append(
                        f"- {sym}: ML={ml_bias} ({ml_conf}%) | LLM PARTIAL={llm_bias} ({llm_conf}%)"
                    )
                else:
                    bias_summary.append(
                        f"- {sym}: {ml_bias} ({ml_conf}%) | LLM agrees ({llm_conf}%)"
                    )
            else:
                bias_summary.append(f"- {sym}: {ml_bias} ({ml_conf}% confidence)")
        
        # Build news summary
        macro_news = premarket_context.market_context.get('macro_news', [])[:3]
        news_summary = []
        for article in macro_news:
            if isinstance(article, dict):
                news_summary.append(article.get("headline", ""))
            elif isinstance(article, str):
                news_summary.append(article)
        
        prompt = f"""You are analyzing the current market state for a STDEV trading system.

Current Technical State:
{chr(10).join(tech_summary)}

Premarket Context (from this morning):
Daily biases (ML model predictions + LLM validation):
{chr(10).join(bias_summary)}
News: {news_summary if news_summary else 'None'}

Recent Price Action (last 15 minutes):
{recent_price_action}

Questions:
1. What's your assessment of current market conditions?
2. Do you see any good trading opportunities coming up in the next 15-30 minutes?
3. How confident are you in these opportunities? (0-100)
4. Should we adjust our trading thresholds? If yes, suggest multipliers:
   - If highly confident: lower thresholds (multiplier < 1.0, e.g., 0.8)
   - If less confident: raise thresholds (multiplier > 1.0, e.g., 1.2)
   - If neutral: keep thresholds as-is (multiplier = 1.0)

Return JSON with market_assessment, opportunities, confidence, threshold_multipliers, and reasoning.
"""
        return prompt
    
    def _parse_multiplier_response(self, content: Dict) -> ThresholdMultiplier:
        """Parse LLM response into ThresholdMultiplier."""
        multipliers = content.get("threshold_multipliers", {})
        return ThresholdMultiplier(
            mr_arm_multiplier=float(multipliers.get("mr_arm_multiplier", 1.0)),
            mr_trigger_multiplier=float(multipliers.get("mr_trigger_multiplier", 1.0)),
            tc_arm_multiplier=float(multipliers.get("tc_arm_multiplier", 1.0)),
            tc_trigger_multiplier=float(multipliers.get("tc_trigger_multiplier", 1.0)),
            confidence=float(content.get("confidence", 0.0)),
            reasoning=content.get("reasoning", ""),
        )

