"""Technical feature computation for live trading loop."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

from src.features.stdev_features import RollingStats
from src.live.state_manager import SymbolState


@dataclass
class SymbolFeatures:
    """Computed features for a symbol."""
    symbol: str
    z_score: float
    mu: float
    sigma: float
    atr_5m: float
    ema_slope_hourly: float
    atr_percentile: float
    timestamp: datetime


def compute_features(
    symbol: str,
    current_price: float,
    state: SymbolState,
    bars_1m: List[Dict[str, Any]]
) -> SymbolFeatures:
    """
    Compute technical features for a symbol.
    
    Args:
        symbol: Symbol name
        current_price: Current price
        state: Current symbol state with rolling stats
        bars_1m: Recent 1-minute bars (for additional indicators if needed)
        
    Returns:
        SymbolFeatures with computed values
    """
    # Update rolling stats
    mu, sigma, z = state.rolling.update(current_price)
    
    # Update state
    state.update_features(mu, sigma, z, datetime.now())
    
    return SymbolFeatures(
        symbol=symbol,
        z_score=z,
        mu=mu,
        sigma=sigma,
        atr_5m=state.atr_5m,
        ema_slope_hourly=state.ema_slope_hourly,
        atr_percentile=state.atr_percentile,
        timestamp=datetime.now(),
    )

