"""STDEV threshold configuration with multiplier support."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdMultiplier:
    """Multipliers for adjusting thresholds based on LLM market analysis."""
    mr_arm_multiplier: float = 1.0    # Multiply k1 by this
    mr_trigger_multiplier: float = 1.0 # Multiply k3 by this
    tc_arm_multiplier: float = 1.0     # Multiply k2 by this
    tc_trigger_multiplier: float = 1.0 # Multiply k3 by this
    confidence: float = 0.0            # LLM confidence in opportunity (0-100)
    reasoning: str = ""                # LLM reasoning


@dataclass
class STDEVThresholds:
    """Base STDEV thresholds for trading signals."""
    # Mean reversion
    mr_arm_z: float = 1.2      # Base threshold: arm when |z| >= this
    mr_trigger_z: float = 0.6   # Base threshold: trigger when z returns to this
    
    # Trend continuation
    tc_arm_z: float = 1.8       # Base threshold: arm when |z| >= this with trend
    tc_trigger_z: float = 0.6   # Base threshold: trigger when z continues
    
    # Risk management
    atr_multiplier_sl: float = 1.4  # Stop loss = entry ± ATR * this
    min_rr_ratio: float = 1.5        # Minimum risk:reward
    max_risk_per_trade: float = 1.0 # % of account
    
    # Filtering
    atr_percentile_cap: float = 85.0 # Only trade if ATR percentile <= this
    
    def apply_multiplier(self, multiplier: ThresholdMultiplier) -> "STDEVThresholds":
        """Return new thresholds with multiplier applied."""
        return STDEVThresholds(
            mr_arm_z=self.mr_arm_z * multiplier.mr_arm_multiplier,
            mr_trigger_z=self.mr_trigger_z * multiplier.mr_trigger_multiplier,
            tc_arm_z=self.tc_arm_z * multiplier.tc_arm_multiplier,
            tc_trigger_z=self.tc_trigger_z * multiplier.tc_trigger_multiplier,
            atr_multiplier_sl=self.atr_multiplier_sl,
            min_rr_ratio=self.min_rr_ratio,
            max_risk_per_trade=self.max_risk_per_trade,
            atr_percentile_cap=self.atr_percentile_cap
        )

