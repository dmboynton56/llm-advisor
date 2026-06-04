"""Map stock STDEV signals into paper option trade plans."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Any, List, Optional

from src.core.config import OptionsSettings, RiskSettings
from src.data.alpaca_options_client import AlpacaOptionsClient, OptionSnapshot


@dataclass(frozen=True)
class OptionTradePlan:
    underlying_symbol: str
    option_symbol: str
    strategy_type: str
    contract_type: str
    side: str
    position_intent: str
    qty: int
    limit_price: float
    estimated_premium: float
    max_loss: float
    expiration_date: str
    dte: int
    strike_price: float
    delta: Optional[float]
    implied_volatility: Optional[float]
    bid_price: float
    ask_price: float
    mid_price: float
    bid_ask_spread_pct: float
    open_interest: int
    setup_type: str
    signal_side: str
    z_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OptionsStrategyMapper:
    """Selects an option contract for a stock signal using conservative filters."""

    def __init__(self, options: OptionsSettings, risk: RiskSettings):
        self.options = options
        self.risk = risk

    def build_trade_plan(
        self,
        signal: Any,
        state: Any,
        options_client: AlpacaOptionsClient,
        account_equity: float,
    ) -> Optional[OptionTradePlan]:
        if self.options.strategy_type != "single_long":
            raise ValueError(f"Unsupported options strategy type: {self.options.strategy_type}")

        contract_type = "call" if str(signal.side).lower() == "long" else "put"
        today = date.today()
        min_exp = today + timedelta(days=self.options.min_dte)
        max_exp = today + timedelta(days=self.options.max_dte)

        candidates = options_client.find_candidates(
            underlying_symbol=signal.symbol,
            contract_type=contract_type,
            underlying_price=float(signal.entry_price),
            expiration_date_gte=min_exp,
            expiration_date_lte=max_exp,
            strike_window_pct=self.options.strike_window_pct,
        )
        filtered = self._filter_candidates(candidates)
        if not filtered:
            return None

        best = self._rank_candidates(filtered)[0]
        mid = best.quote.mid_price
        limit_price = self._round_price(mid * (1.0 + self.options.order_price_buffer_pct))
        risk_budget = min(
            float(self.options.max_premium_per_trade),
            float(account_equity) * (float(self.risk.max_risk_per_trade_percent) / 100.0),
        )
        contract_cost = limit_price * 100.0
        qty = int(risk_budget // contract_cost)
        if qty <= 0:
            return None

        return OptionTradePlan(
            underlying_symbol=signal.symbol,
            option_symbol=best.contract.symbol,
            strategy_type=self.options.strategy_type,
            contract_type=contract_type,
            side="buy",
            position_intent="buy_to_open",
            qty=qty,
            limit_price=limit_price,
            estimated_premium=contract_cost * qty,
            max_loss=contract_cost * qty,
            expiration_date=best.contract.expiration_date.isoformat(),
            dte=best.dte,
            strike_price=best.contract.strike_price,
            delta=best.greeks.delta,
            implied_volatility=best.implied_volatility,
            bid_price=best.quote.bid_price,
            ask_price=best.quote.ask_price,
            mid_price=mid,
            bid_ask_spread_pct=best.quote.spread_pct,
            open_interest=best.contract.open_interest,
            setup_type=signal.setup_type,
            signal_side=signal.side,
            z_score=float(signal.z_score),
        )

    def _filter_candidates(self, candidates: List[OptionSnapshot]) -> List[OptionSnapshot]:
        filtered = []
        for candidate in candidates:
            delta = candidate.greeks.delta
            if delta is None:
                continue
            abs_delta = abs(delta)
            if not (self.options.min_delta <= abs_delta <= self.options.max_delta):
                continue
            if candidate.quote.spread_pct > self.options.max_bid_ask_spread_pct:
                continue
            if candidate.contract.open_interest < self.options.min_open_interest:
                continue
            if candidate.quote.mid_price * 100.0 > self.options.max_premium_per_trade:
                continue
            filtered.append(candidate)
        return filtered

    def _rank_candidates(self, candidates: List[OptionSnapshot]) -> List[OptionSnapshot]:
        target_delta = (self.options.min_delta + self.options.max_delta) / 2.0

        def score(candidate: OptionSnapshot) -> tuple[float, int, float]:
            delta = abs(candidate.greeks.delta or 0.0)
            return (
                abs(delta - target_delta),
                -candidate.contract.open_interest,
                candidate.quote.spread_pct,
            )

        return sorted(candidates, key=score)

    @staticmethod
    def _round_price(price: float) -> float:
        return round(float(price), 2)
