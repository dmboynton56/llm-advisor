"""Map stock STDEV signals into paper option trade plans."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

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
    selection_tier: str = "primary"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptionSelectionProfile:
    name: str
    min_dte: int
    max_dte: int
    min_delta: float
    max_delta: float
    max_premium_per_trade: float
    max_bid_ask_spread_pct: float
    min_open_interest: int
    strike_window_pct: float
    contract_limit: int


class OptionsStrategyMapper:
    """Selects an option contract for a stock signal using conservative filters."""

    def __init__(self, options: OptionsSettings, risk: RiskSettings):
        self.options = options
        self.risk = risk
        self.last_rejection: Optional[Dict[str, Any]] = None

    def build_trade_plan(
        self,
        signal: Any,
        state: Any,
        options_client: AlpacaOptionsClient,
        account_equity: float,
    ) -> Optional[OptionTradePlan]:
        self.last_rejection = None
        if self.options.strategy_type != "single_long":
            raise ValueError(f"Unsupported options strategy type: {self.options.strategy_type}")

        contract_type = "call" if str(signal.side).lower() == "long" else "put"
        today = date.today()
        attempted_rejections: List[Dict[str, Any]] = []

        for profile in self._selection_profiles():
            try:
                plan = self._build_trade_plan_for_profile(
                    signal=signal,
                    options_client=options_client,
                    account_equity=account_equity,
                    contract_type=contract_type,
                    today=today,
                    profile=profile,
                )
            except Exception:
                if attempted_rejections and self.last_rejection:
                    self.last_rejection = {
                        "reason": "candidate_fetch_error_after_prior_attempts",
                        "attempted_profiles": [*attempted_rejections, self.last_rejection],
                    }
                raise

            if plan is not None:
                self.last_rejection = None
                return plan

            if self.last_rejection is not None:
                attempted_rejections.append(self.last_rejection)

        if len(attempted_rejections) > 1:
            self.last_rejection = {
                "reason": "no_option_candidate_after_fallback",
                "attempted_profiles": attempted_rejections,
            }
        elif attempted_rejections:
            self.last_rejection = attempted_rejections[0]
        return None

    def _build_trade_plan_for_profile(
        self,
        signal: Any,
        options_client: AlpacaOptionsClient,
        account_equity: float,
        contract_type: str,
        today: date,
        profile: OptionSelectionProfile,
    ) -> Optional[OptionTradePlan]:
        min_exp = today + timedelta(days=profile.min_dte)
        max_exp = today + timedelta(days=profile.max_dte)

        try:
            candidates = options_client.find_candidates(
                underlying_symbol=signal.symbol,
                contract_type=contract_type,
                underlying_price=float(signal.entry_price),
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                strike_window_pct=profile.strike_window_pct,
                limit=profile.contract_limit,
            )
        except Exception as exc:
            self.last_rejection = {
                "reason": "candidate_fetch_error",
                "selection_tier": profile.name,
                "error": str(exc),
                "search": self._search_context(
                    signal=signal,
                    contract_type=contract_type,
                    min_exp=min_exp,
                    max_exp=max_exp,
                    profile=profile,
                ),
                "candidate_source": self._candidate_source(options_client),
            }
            raise

        filtered, filter_rejections = self._filter_candidates_with_diagnostics(candidates, profile)
        if not filtered:
            self.last_rejection = {
                "reason": "no_candidate_snapshots" if not candidates else "all_candidates_filtered",
                "selection_tier": profile.name,
                "candidate_count": len(candidates),
                "filtered_count": 0,
                "filter_rejections": filter_rejections,
                "search": self._search_context(
                    signal=signal,
                    contract_type=contract_type,
                    min_exp=min_exp,
                    max_exp=max_exp,
                    profile=profile,
                ),
                "candidate_source": self._candidate_source(options_client),
            }
            return None

        best = self._rank_candidates(filtered, profile)[0]
        mid = best.quote.mid_price
        limit_price = self._round_price(mid * (1.0 + self.options.order_price_buffer_pct))
        risk_budget = min(
            float(profile.max_premium_per_trade),
            float(account_equity) * (float(self.risk.max_risk_per_trade_percent) / 100.0),
        )
        contract_cost = limit_price * 100.0
        qty = int(risk_budget // contract_cost)
        if qty <= 0:
            self.last_rejection = {
                "reason": "premium_exceeds_risk_budget",
                "selection_tier": profile.name,
                "candidate_count": len(candidates),
                "filtered_count": len(filtered),
                "risk_budget": risk_budget,
                "limit_price": limit_price,
                "contract_cost": contract_cost,
                "max_premium_per_trade": profile.max_premium_per_trade,
                "best_candidate": self._candidate_summary(best),
                "search": self._search_context(
                    signal=signal,
                    contract_type=contract_type,
                    min_exp=min_exp,
                    max_exp=max_exp,
                    profile=profile,
                ),
                "candidate_source": self._candidate_source(options_client),
            }
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
            selection_tier=profile.name,
        )

    def _filter_candidates(self, candidates: List[OptionSnapshot]) -> List[OptionSnapshot]:
        filtered, _ = self._filter_candidates_with_diagnostics(candidates, self._primary_profile())
        return filtered

    def _filter_candidates_with_diagnostics(
        self,
        candidates: List[OptionSnapshot],
        profile: OptionSelectionProfile,
    ) -> Tuple[List[OptionSnapshot], Dict[str, int]]:
        rejections = {
            "missing_delta": 0,
            "delta_out_of_range": 0,
            "spread_too_wide": 0,
            "open_interest_too_low": 0,
            "premium_too_high": 0,
        }
        filtered = []
        for candidate in candidates:
            delta = candidate.greeks.delta
            if delta is None:
                rejections["missing_delta"] += 1
                continue
            abs_delta = abs(delta)
            if not (profile.min_delta <= abs_delta <= profile.max_delta):
                rejections["delta_out_of_range"] += 1
                continue
            if candidate.quote.spread_pct > profile.max_bid_ask_spread_pct:
                rejections["spread_too_wide"] += 1
                continue
            if candidate.contract.open_interest < profile.min_open_interest:
                rejections["open_interest_too_low"] += 1
                continue
            if candidate.quote.mid_price * 100.0 > profile.max_premium_per_trade:
                rejections["premium_too_high"] += 1
                continue
            filtered.append(candidate)
        return filtered, rejections

    def _rank_candidates(
        self,
        candidates: List[OptionSnapshot],
        profile: OptionSelectionProfile,
    ) -> List[OptionSnapshot]:
        target_delta = (profile.min_delta + profile.max_delta) / 2.0

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

    def _selection_profiles(self) -> List[OptionSelectionProfile]:
        profiles = [self._primary_profile()]
        if self.options.fallback_enabled:
            profiles.append(self._fallback_profile())
        return profiles

    def _primary_profile(self) -> OptionSelectionProfile:
        return OptionSelectionProfile(
            name="primary",
            min_dte=self.options.min_dte,
            max_dte=self.options.max_dte,
            min_delta=self.options.min_delta,
            max_delta=self.options.max_delta,
            max_premium_per_trade=self.options.max_premium_per_trade,
            max_bid_ask_spread_pct=self.options.max_bid_ask_spread_pct,
            min_open_interest=self.options.min_open_interest,
            strike_window_pct=self.options.strike_window_pct,
            contract_limit=self.options.contract_limit,
        )

    def _fallback_profile(self) -> OptionSelectionProfile:
        return OptionSelectionProfile(
            name="fallback",
            min_dte=self.options.fallback_min_dte,
            max_dte=self.options.fallback_max_dte,
            min_delta=self.options.fallback_min_delta,
            max_delta=self.options.fallback_max_delta,
            max_premium_per_trade=self.options.fallback_max_premium_per_trade,
            max_bid_ask_spread_pct=self.options.fallback_max_bid_ask_spread_pct,
            min_open_interest=self.options.fallback_min_open_interest,
            strike_window_pct=self.options.fallback_strike_window_pct,
            contract_limit=self.options.fallback_contract_limit,
        )

    def _search_context(
        self,
        signal: Any,
        contract_type: str,
        min_exp: date,
        max_exp: date,
        profile: OptionSelectionProfile,
    ) -> Dict[str, Any]:
        return {
            "selection_tier": profile.name,
            "underlying_symbol": signal.symbol,
            "contract_type": contract_type,
            "underlying_price": float(signal.entry_price),
            "expiration_date_gte": min_exp.isoformat(),
            "expiration_date_lte": max_exp.isoformat(),
            "strike_window_pct": profile.strike_window_pct,
            "contract_limit": profile.contract_limit,
            "filters": {
                "min_delta": profile.min_delta,
                "max_delta": profile.max_delta,
                "max_bid_ask_spread_pct": profile.max_bid_ask_spread_pct,
                "min_open_interest": profile.min_open_interest,
                "max_premium_per_trade": profile.max_premium_per_trade,
                "order_price_buffer_pct": self.options.order_price_buffer_pct,
            },
        }

    @staticmethod
    def _candidate_source(options_client: AlpacaOptionsClient) -> Dict[str, Any]:
        return dict(getattr(options_client, "last_candidate_diagnostics", {}) or {})

    @staticmethod
    def _candidate_summary(candidate: OptionSnapshot) -> Dict[str, Any]:
        return {
            "option_symbol": candidate.contract.symbol,
            "expiration_date": candidate.contract.expiration_date.isoformat(),
            "dte": candidate.dte,
            "strike_price": candidate.contract.strike_price,
            "delta": candidate.greeks.delta,
            "bid_price": candidate.quote.bid_price,
            "ask_price": candidate.quote.ask_price,
            "mid_price": candidate.quote.mid_price,
            "bid_ask_spread_pct": candidate.quote.spread_pct,
            "open_interest": candidate.contract.open_interest,
        }
