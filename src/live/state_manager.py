"""Symbol state management for live trading loop."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.features.stdev_features import RollingStats
from config.thresholds import STDEVThresholds, ThresholdMultiplier


@dataclass
class TradePlan:
    """Trade plan when signal triggers."""
    setup: str  # "MR" or "TC"
    side: str   # "long" or "short"
    entry_price: float
    sl_price: float
    tp_price: float
    triggered_at: datetime
    execution_attempts: int = 0  # Track how many times we've tried to execute
    first_execution_attempt: Optional[datetime] = None  # When we first tried to execute
    signal_uid: str = ""
    first_capacity_blocked_at: Optional[datetime] = None
    capacity_skip_count: int = 0
    capacity_revalidation_pending: bool = False
    revalidation_count: int = 0
    selected_option_symbols: List[str] = field(default_factory=list)
    excluded_option_symbols: List[str] = field(default_factory=list)
    alternate_contract_attempted: bool = False
    guard_failure_reasons: List[str] = field(default_factory=list)


@dataclass
class SymbolState:
    """State for a single symbol in the live loop."""
    symbol: str
    rolling: RollingStats
    htf_bias: str
    ema_slope_hourly: float
    atr_percentile: float
    atr_5m: float
    thresholds: STDEVThresholds
    threshold_multiplier: Optional[ThresholdMultiplier] = None
    status: str = "idle"  # idle | mr_armed | mr_triggered | tc_armed | tc_triggered
    side: Optional[str] = None
    last_mu: float = 0.0
    last_sigma: float = 0.0
    last_z: float = 0.0
    # Entry-time underlying levels are observational during the tiered paper
    # trial; they never trigger an option exit.
    shadow_levels: Dict[str, Any] = field(default_factory=dict)
    armed_z: Optional[float] = None
    z_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    trade: Optional[TradePlan] = None
    last_update_utc: Optional[str] = None

    def should_gate(self) -> bool:
        """Check if trading should be gated based on ATR percentile."""
        return self.atr_percentile <= self.thresholds.atr_percentile_cap
    
    def get_active_thresholds(self) -> STDEVThresholds:
        """Get thresholds with multiplier applied."""
        if self.threshold_multiplier:
            return self.thresholds.apply_multiplier(self.threshold_multiplier)
        return self.thresholds
    
    def update_features(self, mu: float, sigma: float, z: float, timestamp: datetime) -> None:
        """Update rolling stats and feature values."""
        self.last_mu = float(mu)
        self.last_sigma = float(sigma)
        self.last_z = float(z)
        self.last_update_utc = timestamp.isoformat()
        if self.status.endswith("_armed") or self.status.endswith("_triggered"):
            self.z_trajectory.append({"ts": self.last_update_utc, "z": self.last_z})
            self.z_trajectory = self.z_trajectory[-120:]

    def arm(self, status: str, side: str, z: float) -> None:
        """Start a new auditable signal trajectory."""
        self.status = status
        self.side = side
        self.armed_z = float(z)
        self.z_trajectory = [{"ts": self.last_update_utc, "z": float(z)}]
    
    def reset_to_idle(self) -> None:
        """Reset state to idle."""
        self.status = "idle"
        self.side = None
        self.armed_z = None
        self.z_trajectory = []
        self.trade = None
