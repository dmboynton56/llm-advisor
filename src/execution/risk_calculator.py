"""Position sizing logic based on risk parameters."""
from typing import Optional

_RR_EPSILON = 1e-9


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    max_risk_percent: float,
    atr_5m: Optional[float] = None,
    max_shares: Optional[int] = None,
) -> int:
    """
    Calculate position size in shares based on risk parameters.

    Args:
        account_equity: Total account equity
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
        max_risk_percent: Maximum risk per trade as percentage (e.g., 1.0 for 1%)
        atr_5m: Optional 5-minute ATR for ATR-based position sizing
        max_shares: Optional cap from buying power / notional limits

    Returns:
        Number of shares to trade (rounded down to integer), or 0 if untradeable
    """
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share == 0:
        return 0

    total_risk = account_equity * (max_risk_percent / 100.0)
    shares = int(total_risk / risk_per_share)

    if atr_5m and entry_price > 0:
        atr_percent = (atr_5m / entry_price) * 100.0
        if atr_percent > 2.0:
            reduction_factor = min(0.5, 1.0 - ((atr_percent - 2.0) / 10.0))
            shares = int(shares * reduction_factor)

    if max_shares is not None:
        shares = min(shares, max(0, max_shares))

    if shares <= 0:
        return 0

    return max(1, shares)


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float
) -> float:
    """Calculate risk:reward ratio (reward / risk)."""
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
    """Validate that trade meets minimum risk:reward ratio."""
    rr_ratio = calculate_risk_reward_ratio(entry_price, stop_loss_price, take_profit_price)
    return rr_ratio + _RR_EPSILON >= min_rr_ratio


def max_shares_for_buying_power(
    buying_power: float,
    entry_price: float,
    notional_pct: float = 0.95,
) -> int:
    """Max whole shares affordable within a notional fraction of buying power."""
    if buying_power <= 0 or entry_price <= 0 or notional_pct <= 0:
        return 0
    return int((buying_power * notional_pct) / entry_price)
