"""Unit tests for risk calculator."""
import pytest
from src.execution.risk_calculator import (
    calculate_position_size,
    validate_risk_reward,
    calculate_risk_reward_ratio
)


def test_calculate_position_size_basic():
    """Test basic position sizing."""
    equity = 100000.0
    entry = 100.0
    stop_loss = 95.0
    max_risk = 1.0  # 1%
    
    shares = calculate_position_size(equity, entry, stop_loss, max_risk)
    
    # Risk per share = $5
    # Total risk = $1000 (1% of $100k)
    # Shares = $1000 / $5 = 200
    assert shares == 200


def test_calculate_position_size_zero_risk():
    """Test position sizing with zero risk (invalid)."""
    shares = calculate_position_size(100000.0, 100.0, 100.0, 1.0)
    assert shares == 0


def test_calculate_position_size_atr_adjustment():
    """Test ATR-based position size reduction."""
    equity = 100000.0
    entry = 100.0
    stop_loss = 95.0
    max_risk = 1.0
    atr_5m = 3.0  # 3% of price
    
    shares_without_atr = calculate_position_size(equity, entry, stop_loss, max_risk, None)
    shares_with_atr = calculate_position_size(equity, entry, stop_loss, max_risk, atr_5m)
    
    # Should be reduced due to high ATR
    assert shares_with_atr <= shares_without_atr


def test_calculate_position_size_minimum_one_share():
    """Test that position size is at least 1 share."""
    equity = 1000.0
    entry = 100.0
    stop_loss = 99.9  # Very small risk
    max_risk = 0.1  # Very small %
    
    shares = calculate_position_size(equity, entry, stop_loss, max_risk)
    assert shares >= 1


def test_calculate_risk_reward_ratio():
    """Test R:R calculation."""
    # Good R:R (2:1)
    rr = calculate_risk_reward_ratio(100.0, 95.0, 110.0)
    assert rr == 2.0
    
    # Poor R:R (1:1)
    rr = calculate_risk_reward_ratio(100.0, 95.0, 100.0)
    assert rr == 1.0
    
    # Zero risk (invalid)
    rr = calculate_risk_reward_ratio(100.0, 100.0, 110.0)
    assert rr == 0.0


def test_validate_risk_reward():
    """Test R:R validation."""
    # Good R:R (2:1) - should pass with min 1.5
    assert validate_risk_reward(100.0, 95.0, 110.0, 1.5) == True
    
    # Poor R:R (1:1) - should fail with min 1.5
    assert validate_risk_reward(100.0, 95.0, 100.0, 1.5) == False
    
    # Exactly at minimum
    assert validate_risk_reward(100.0, 95.0, 107.5, 1.5) == True  # R:R = 1.5

