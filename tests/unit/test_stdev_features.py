"""Unit tests for STDEV features (RollingStats)."""
import pytest
from src.features.stdev_features import RollingStats, compute_z_series


def test_rolling_stats_from_seed():
    """Test initializing RollingStats from seed data."""
    seed = [100.0, 101.0, 99.0, 102.0, 98.0]
    stats = RollingStats.from_seed(seed, window=5)
    
    assert stats.count == 5
    assert stats.window == 5
    assert stats.mean == pytest.approx(100.0, abs=0.1)
    assert stats.std > 0


def test_rolling_stats_update():
    """Test updating rolling stats with new value."""
    seed = [100.0, 101.0, 99.0, 102.0, 98.0]
    stats = RollingStats.from_seed(seed, window=5)
    
    old_mean = stats.mean
    
    mu, sigma, z = stats.update(103.0)
    
    assert stats.count == 5  # Window size maintained
    assert mu == stats.mean
    assert mu != old_mean  # Mean changed
    assert sigma == stats.std
    assert z != 0


def test_rolling_stats_window_size():
    """Test that window size is maintained."""
    stats = RollingStats.from_seed([100.0], window=3)
    
    # Add more than window size
    for i in range(10):
        stats.update(100.0 + i)
    
    assert stats.count == 3  # Should not exceed window


def test_rolling_stats_std_calculation():
    """Test standard deviation calculation."""
    # Constant values should have std = 0
    stats = RollingStats.from_seed([100.0, 100.0, 100.0], window=3)
    assert stats.std == 0.0
    
    # Varied values should have std > 0
    stats2 = RollingStats.from_seed([100.0, 110.0, 90.0], window=3)
    assert stats2.std > 0


def test_rolling_stats_z_score():
    """Test z-score calculation."""
    seed = [100.0, 100.0, 100.0, 100.0, 100.0]
    stats = RollingStats.from_seed(seed, window=5)
    
    mu, sigma, z = stats.update(110.0)
    
    # With constant seed, sigma should be 0, so z should be 0
    # But once we update, sigma may be > 0
    assert isinstance(z, float)


def test_compute_z_series():
    """Test computing z-score series."""
    prices = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]
    z_scores = compute_z_series(prices, window=3)
    
    assert len(z_scores) == len(prices)
    assert all(isinstance(z, float) for z in z_scores)
    # First few should be 0 (window not filled)
    # Later ones should have valid z-scores


def test_rolling_stats_empty():
    """Test RollingStats with empty seed."""
    stats = RollingStats.from_seed([], window=5)
    
    assert stats.count == 0
    assert stats.mean == 0.0
    assert stats.std == 0.0
    
    mu, sigma, z = stats.update(100.0)
    assert stats.count == 1
    assert mu == 100.0
    assert sigma == 0.0  # Need at least 2 values for std
    assert z == 0.0

