"""Pytest configuration and fixtures."""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.stdev_features import RollingStats
from src.live.state_manager import SymbolState, TradePlan
from config.thresholds import STDEVThresholds, ThresholdMultiplier


@pytest.fixture
def sample_thresholds():
    """Sample STDEV thresholds for testing."""
    return STDEVThresholds(
        mr_arm_z=1.2,
        mr_trigger_z=0.6,
        tc_arm_z=1.8,
        tc_trigger_z=0.6,
        atr_multiplier_sl=1.4,
        min_rr_ratio=1.5,
        max_risk_per_trade=1.0,
        atr_percentile_cap=85.0,
    )


@pytest.fixture
def sample_multiplier():
    """Sample threshold multiplier."""
    return ThresholdMultiplier(
        mr_arm_multiplier=1.0,
        mr_trigger_multiplier=1.0,
        tc_arm_multiplier=1.0,
        tc_trigger_multiplier=1.0,
        confidence=75.0,
        reasoning="Test reasoning",
    )


@pytest.fixture
def sample_rolling_stats():
    """Sample rolling stats initialized with seed data."""
    seed_data = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0]
    return RollingStats.from_seed(seed_data, window=120)


@pytest.fixture
def sample_symbol_state(sample_thresholds, sample_rolling_stats):
    """Sample symbol state for testing."""
    return SymbolState(
        symbol="SPY",
        rolling=sample_rolling_stats,
        htf_bias="bullish",
        ema_slope_hourly=0.5,
        atr_percentile=65.0,
        atr_5m=0.5,
        thresholds=sample_thresholds,
    )


@pytest.fixture
def sample_bars():
    """Sample 1-minute bar data."""
    base_time = datetime(2025, 1, 7, 9, 30)
    bars = []
    for i in range(30):
        bar_time = base_time + timedelta(minutes=i)
        bars.append({
            "t": bar_time.isoformat(),
            "o": 450.0 + i * 0.1,
            "h": 450.5 + i * 0.1,
            "l": 449.5 + i * 0.1,
            "c": 450.2 + i * 0.1,
            "v": 1000.0,
        })
    return bars


@pytest.fixture
def sample_price_series():
    """Sample price series for testing."""
    return [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0]

