"""Unit tests for state manager."""
import pytest
from datetime import datetime
from src.live.state_manager import SymbolState, TradePlan
from src.features.stdev_features import RollingStats
from config.thresholds import STDEVThresholds, ThresholdMultiplier


def test_symbol_state_initialization():
    """Test symbol state initialization."""
    thresholds = STDEVThresholds()
    rolling = RollingStats.from_seed([100.0] * 120, window=120)
    
    state = SymbolState(
        symbol="SPY",
        rolling=rolling,
        htf_bias="bullish",
        ema_slope_hourly=0.5,
        atr_percentile=65.0,
        atr_5m=0.5,
        thresholds=thresholds,
    )
    
    assert state.symbol == "SPY"
    assert state.status == "idle"
    assert state.side is None
    assert state.trade is None


def test_should_gate():
    """Test gating based on ATR percentile."""
    thresholds = STDEVThresholds(atr_percentile_cap=85.0)
    rolling = RollingStats.from_seed([100.0] * 120, window=120)
    
    state = SymbolState(
        symbol="SPY",
        rolling=rolling,
        htf_bias="bullish",
        ema_slope_hourly=0.5,
        atr_percentile=65.0,  # Below cap
        atr_5m=0.5,
        thresholds=thresholds,
    )
    
    assert state.should_gate() == True  # Should allow trading
    
    state.atr_percentile = 90.0  # Above cap
    assert state.should_gate() == False  # Should gate


def test_get_active_thresholds_with_multiplier():
    """Test getting thresholds with multiplier applied."""
    thresholds = STDEVThresholds(mr_arm_z=1.2)
    rolling = RollingStats.from_seed([100.0] * 120, window=120)
    
    state = SymbolState(
        symbol="SPY",
        rolling=rolling,
        htf_bias="bullish",
        ema_slope_hourly=0.5,
        atr_percentile=65.0,
        atr_5m=0.5,
        thresholds=thresholds,
    )
    
    multiplier = ThresholdMultiplier(mr_arm_multiplier=0.8)
    state.threshold_multiplier = multiplier
    
    active = state.get_active_thresholds()
    assert active.mr_arm_z == pytest.approx(1.2 * 0.8)


def test_update_features():
    """Test updating features."""
    thresholds = STDEVThresholds()
    rolling = RollingStats.from_seed([100.0] * 120, window=120)
    state = SymbolState(
        symbol="SPY",
        rolling=rolling,
        htf_bias="bullish",
        ema_slope_hourly=0.5,
        atr_percentile=65.0,
        atr_5m=0.5,
        thresholds=thresholds,
    )
    
    timestamp = datetime.now()
    state.update_features(100.0, 1.0, 0.5, timestamp)
    
    assert state.last_mu == 100.0
    assert state.last_sigma == 1.0
    assert state.last_z == 0.5
    assert state.last_update_utc == timestamp.isoformat()


def test_reset_to_idle():
    """Test resetting to idle state."""
    thresholds = STDEVThresholds()
    rolling = RollingStats.from_seed([100.0] * 120, window=120)
    state = SymbolState(
        symbol="SPY",
        rolling=rolling,
        htf_bias="bullish",
        ema_slope_hourly=0.5,
        atr_percentile=65.0,
        atr_5m=0.5,
        thresholds=thresholds,
    )
    
    # Set to armed state
    state.status = "mr_armed"
    state.side = "long"
    state.armed_z = 1.5
    state.trade = TradePlan(
        setup="MR",
        side="long",
        entry_price=100.0,
        sl_price=95.0,
        tp_price=110.0,
        triggered_at=datetime.now(),
    )
    
    state.reset_to_idle()
    
    assert state.status == "idle"
    assert state.side is None
    assert state.armed_z is None
    assert state.trade is None

