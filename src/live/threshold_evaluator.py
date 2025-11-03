"""Threshold evaluation with multiplier support."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.live.state_manager import SymbolState, TradePlan
from config.thresholds import ThresholdMultiplier, STDEVThresholds


@dataclass
class SignalEvent:
    """Signal event when threshold is crossed."""
    symbol: str
    setup_type: str  # "MR" or "TC"
    side: str        # "long" or "short"
    entry_price: float
    z_score: float
    thresholds_used: dict  # Threshold values used
    timestamp: datetime


def evaluate_thresholds(
    state: SymbolState,
    current_price: float,
    multiplier: Optional[ThresholdMultiplier] = None
) -> Optional[SignalEvent]:
    """
    Evaluate thresholds and return signal if triggered.
    
    Args:
        state: Symbol state with current features
        current_price: Current price
        multiplier: Optional threshold multiplier from LLM analysis
        
    Returns:
        SignalEvent if threshold crossed, None otherwise
    """
    # Get active thresholds (with multiplier applied)
    if multiplier:
        state.threshold_multiplier = multiplier
    thresholds = state.get_active_thresholds()
    
    # Gate check
    if not state.should_gate():
        state.reset_to_idle()
        return None
    
    z = state.last_z
    
    # Mean reversion arming
    if state.status == "idle":
        if abs(z) >= thresholds.mr_arm_z:
            state.status = "mr_armed"
            state.side = "long" if z < 0 else "short"
            state.armed_z = z
            return None
        
        # Trend continuation arming (requires slope alignment)
        if state.ema_slope_hourly > 0 and z >= thresholds.tc_arm_z:
            state.status = "tc_armed"
            state.side = "long"
            state.armed_z = z
            return None
        if state.ema_slope_hourly < 0 and z <= -thresholds.tc_arm_z:
            state.status = "tc_armed"
            state.side = "short"
            state.armed_z = z
            return None
    
    # MR trigger
    if state.status == "mr_armed":
        if state.side == "long" and z >= -thresholds.mr_trigger_z:
            return _create_signal(state, "MR", current_price, thresholds)
        if state.side == "short" and z <= thresholds.mr_trigger_z:
            return _create_signal(state, "MR", current_price, thresholds)
        if abs(z) < thresholds.mr_arm_z / 2:
            state.reset_to_idle()
    
    # TC trigger
    if state.status == "tc_armed":
        if state.side == "long" and z >= thresholds.tc_trigger_z:
            return _create_signal(state, "TC", current_price, thresholds)
        if state.side == "short" and z <= -thresholds.tc_trigger_z:
            return _create_signal(state, "TC", current_price, thresholds)
        if abs(z) < thresholds.mr_arm_z:
            state.reset_to_idle()
    
    return None


def _create_signal(
    state: SymbolState,
    setup_type: str,
    price: float,
    thresholds: "STDEVThresholds"
) -> SignalEvent:
    """Create a signal event and trade plan."""
    atr_offset = thresholds.atr_multiplier_sl * state.atr_5m
    
    if state.side == "long":
        sl = price - atr_offset
        tp = price + thresholds.min_rr_ratio * atr_offset
    else:
        sl = price + atr_offset
        tp = price - thresholds.min_rr_ratio * atr_offset
    
    state.trade = TradePlan(
        setup=setup_type,
        side=state.side or "long",
        entry_price=price,
        sl_price=sl,
        tp_price=tp,
        triggered_at=datetime.now(),
    )
    state.status = f"{setup_type.lower()}_triggered"
    
    return SignalEvent(
        symbol=state.symbol,
        setup_type=setup_type,
        side=state.side or "long",
        entry_price=price,
        z_score=state.last_z,
        thresholds_used={
            "mr_arm_z": thresholds.mr_arm_z,
            "mr_trigger_z": thresholds.mr_trigger_z,
            "tc_arm_z": thresholds.tc_arm_z,
            "tc_trigger_z": thresholds.tc_trigger_z,
        },
        timestamp=datetime.now(),
    )

