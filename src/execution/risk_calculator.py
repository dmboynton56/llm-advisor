"""Position sizing logic based on risk parameters."""
from typing import Optional


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    max_risk_percent: float,
    atr_5m: Optional[float] = None
) -> int:
    """
    Calculate position size in shares based on risk parameters.
    
    Args:
        account_equity: Total account equity
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
        max_risk_percent: Maximum risk per trade as percentage (e.g., 1.0 for 1%)
        atr_5m: Optional 5-minute ATR for ATR-based position sizing
        
    Returns:
        Number of shares to trade (rounded down to integer)
    """
    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share == 0:
        return 0
    
    # Calculate total risk amount
    total_risk = account_equity * (max_risk_percent / 100.0)
    
    # Calculate base position size
    shares = int(total_risk / risk_per_share)
    
    # Optionally adjust based on ATR
    # Larger ATR = smaller position size (more volatile = less shares)
    if atr_5m and entry_price > 0:
        atr_percent = (atr_5m / entry_price) * 100.0
        
        # If ATR is high (> 2% of price), reduce position size
        if atr_percent > 2.0:
            # Reduce by up to 50% for very high volatility
            reduction_factor = min(0.5, 1.0 - ((atr_percent - 2.0) / 10.0))
            shares = int(shares * reduction_factor)
    
    return max(1, shares)  # At least 1 share


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float
) -> float:
    """
    Calculate risk:reward ratio for a trade.
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price
        
    Returns:
        Risk:reward ratio (reward / risk)
    """
    risk = abs(entry_price - stop_loss_price)
    reward = abs(take_profit_price - entry_price)
    
    if risk == 0:
        return 0.0
    
    return reward / risk


def validate_risk_reward(
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
    min_rr_ratio: float
) -> bool:
    """
    Validate that trade meets minimum risk:reward ratio.
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price
        min_rr_ratio: Minimum required risk:reward ratio
        
    Returns:
        True if trade meets minimum R:R, False otherwise
    """
    rr_ratio = calculate_risk_reward_ratio(entry_price, stop_loss_price, take_profit_price)
    return rr_ratio >= min_rr_ratio

