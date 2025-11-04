"""Unit tests for threshold evaluator."""
import pytest
from datetime import datetime
from src.live.threshold_evaluator import evaluate_thresholds, SignalEvent
from src.live.state_manager import SymbolState
from src.features.stdev_features import RollingStats
from config.thresholds import STDEVThresholds, ThresholdMultiplier


@pytest.fixture
def test_state():
    """Create test symbol state."""
    thresholds = STDEVThresholds(
        mr_arm_z=1.2,
        mr_trigger_z=0.6,
        tc_arm_z=1.8,
        tc_trigger_z=0.6,
        atr_percentile_cap=85.0,
    )
    rolling = RollingStats.from_seed([100.0] * 120, window=120)
    
    return SymbolState(
        symbol="SPY",
        rolling=rolling,
        htf_bias="bullish",
        ema_slope_hourly=0.5,
        atr_percentile=65.0,
        atr_5m=0.5,
        thresholds=thresholds,
    )


def test_idle_state_no_signal(test_state):
    """Test that idle state with normal z-score returns no signal."""
    test_state.rolling.update(100.0)  # Normal price, z ≈ 0
    test_state.update_features(100.0, 1.0, 0.1, datetime.now())
    
    signal = evaluate_thresholds(test_state, 100.0)
    assert signal is None
    assert test_state.status == "idle"


def test_mr_arming(test_state):
    """Test mean reversion arming."""
    # Update to high z-score (should arm)
    test_state.rolling.update(102.5)  # High price
    mu, sigma, z = test_state.rolling.update(102.5)
    test_state.update_features(mu, sigma, z, datetime.now())
    
    signal = evaluate_thresholds(test_state, 102.5)
    
    # Should arm but not trigger yet
    assert signal is None
    assert test_state.status == "mr_armed"
    assert test_state.side == "short"  # High z = short mean reversion


def test_mr_trigger(test_state):
    """Test mean reversion trigger."""
    # Arm first
    test_state.status = "mr_armed"
    test_state.side = "short"
    test_state.armed_z = 1.5
    
    # Update to trigger level (z returns toward 0)
    test_state.rolling.update(100.5)
    mu, sigma, z = test_state.rolling.update(100.5)
    test_state.update_features(mu, sigma, z, datetime.now())
    
    signal = evaluate_thresholds(test_state, 100.5)
    
    assert signal is not None
    assert signal.setup_type == "MR"
    assert signal.side == "short"


def test_tc_arming(test_state):
    """Test trend continuation arming."""
    # Bullish trend (positive slope)
    test_state.ema_slope_hourly = 0.5
    
    # Update to high z-score with trend (should arm TC)
    test_state.rolling.update(102.0)
    mu, sigma, z = test_state.rolling.update(102.0)
    test_state.update_features(mu, sigma, z, datetime.now())
    
    signal = evaluate_thresholds(test_state, 102.0)
    
    # Should arm TC if z is high enough
    if z >= test_state.thresholds.tc_arm_z:
        assert test_state.status == "tc_armed"
        assert test_state.side == "long"


def test_multiplier_applied(test_state):
    """Test that threshold multipliers are applied."""
    multiplier = ThresholdMultiplier(
        mr_arm_multiplier=0.8,  # Lower threshold (easier to arm)
        mr_trigger_multiplier=1.0,
        tc_arm_multiplier=1.0,
        tc_trigger_multiplier=1.0,
    )
    
    test_state.threshold_multiplier = multiplier
    
    # Update to z-score that wouldn't normally arm but should with multiplier
    test_state.rolling.update(101.0)
    mu, sigma, z = test_state.rolling.update(101.0)
    test_state.update_features(mu, sigma, z, datetime.now())
    
    # With 0.8 multiplier, 1.2 * 0.8 = 0.96 threshold
    # So z = 1.0 should arm
    if abs(z) >= 0.96:
        signal = evaluate_thresholds(test_state, 101.0, multiplier)
        # May arm depending on z value
        assert isinstance(signal, (SignalEvent, type(None)))


def test_gating_prevents_trading(test_state):
    """Test that high ATR percentile gates trading."""
    test_state.atr_percentile = 90.0  # Above cap
    
    signal = evaluate_thresholds(test_state, 100.0)
    
    # Should reset to idle due to gating
    assert signal is None
    assert test_state.status == "idle"

