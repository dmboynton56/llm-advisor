"""Alpaca option contract and market-data access."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

from alpaca.data.enums import OptionsFeed
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest

from src.utils.env_sanitize import getenv_strip

load_dotenv()


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying_symbol: str
    contract_type: str
    expiration_date: date
    strike_price: float
    tradable: bool
    open_interest: int = 0
    close_price: Optional[float] = None


@dataclass(frozen=True)
class OptionQuote:
    bid_price: float
    ask_price: float
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    @property
    def mid_price(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2.0
        return max(self.bid_price, self.ask_price)

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return 1.0
        return max(0.0, self.ask_price - self.bid_price) / mid


@dataclass(frozen=True)
class OptionGreeks:
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None


@dataclass(frozen=True)
class OptionSnapshot:
    contract: OptionContract
    quote: OptionQuote
    implied_volatility: Optional[float] = None
    greeks: OptionGreeks = field(default_factory=OptionGreeks)

    @property
    def dte(self) -> int:
        return max(0, (self.contract.expiration_date - date.today()).days)


class AlpacaOptionsClient:
    """Small wrapper around Alpaca option contracts and option snapshots."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
        feed: str = "indicative",
    ):
        api_key = api_key or getenv_strip("ALPACA_API_KEY")
        api_secret = api_secret or getenv_strip("ALPACA_SECRET_KEY")

        if not api_key or not api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")

        self.trading_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper)
        self.data_client = OptionHistoricalDataClient(api_key, api_secret)
        self.feed = OptionsFeed.OPRA if str(feed).lower() == "opra" else OptionsFeed.INDICATIVE

    def fetch_contracts(
        self,
        underlying_symbol: str,
        contract_type: str,
        expiration_date_gte: date,
        expiration_date_lte: date,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        limit: int = 100,
    ) -> List[OptionContract]:
        """Fetch active option contracts for one underlying and normalize them."""
        ctype = ContractType.CALL if contract_type.lower() == "call" else ContractType.PUT
        request = GetOptionContractsRequest(
            underlying_symbols=[underlying_symbol],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            type=ctype,
            strike_price_gte=str(strike_price_gte) if strike_price_gte is not None else None,
            strike_price_lte=str(strike_price_lte) if strike_price_lte is not None else None,
            limit=limit,
        )
        response = self.trading_client.get_option_contracts(request)
        return [self._normalize_contract(c) for c in self._contracts_from_response(response)]

    def fetch_snapshots(self, contracts: Iterable[OptionContract]) -> List[OptionSnapshot]:
        """Fetch option snapshots for contracts and pair them with contract metadata."""
        contract_list = list(contracts)
        if not contract_list:
            return []

        by_symbol = {contract.symbol: contract for contract in contract_list}
        request = OptionSnapshotRequest(
            symbol_or_symbols=list(by_symbol.keys()),
            feed=self.feed,
        )
        response = self.data_client.get_option_snapshot(request)
        snapshots = []
        for symbol, raw_snapshot in self._items(response):
            contract = by_symbol.get(symbol)
            if contract is None:
                continue
            normalized = self._normalize_snapshot(contract, raw_snapshot)
            if normalized is not None:
                snapshots.append(normalized)
        return snapshots

    def find_candidates(
        self,
        underlying_symbol: str,
        contract_type: str,
        underlying_price: float,
        expiration_date_gte: date,
        expiration_date_lte: date,
        strike_window_pct: float,
        limit: int = 100,
    ) -> List[OptionSnapshot]:
        """Fetch contracts near the current underlying price and return usable snapshots."""
        strike_min = underlying_price * (1.0 - strike_window_pct)
        strike_max = underlying_price * (1.0 + strike_window_pct)
        contracts = self.fetch_contracts(
            underlying_symbol=underlying_symbol,
            contract_type=contract_type,
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
            limit=limit,
        )
        return self.fetch_snapshots([contract for contract in contracts if contract.tradable])

    @staticmethod
    def _contracts_from_response(response: Any) -> List[Any]:
        if response is None:
            return []
        if isinstance(response, list):
            return response
        for attr in ("option_contracts", "contracts"):
            value = getattr(response, attr, None)
            if value is not None:
                return list(value)
        if isinstance(response, dict):
            return list(response.get("option_contracts") or response.get("contracts") or [])
        return []

    @staticmethod
    def _items(response: Any) -> Iterable[tuple[str, Any]]:
        if response is None:
            return []
        if isinstance(response, dict):
            return response.items()
        if hasattr(response, "items"):
            return response.items()
        return []

    @staticmethod
    def _get(raw: Any, name: str, default: Any = None) -> Any:
        if isinstance(raw, dict):
            return raw.get(name, default)
        return getattr(raw, name, default)

    @classmethod
    def _float_or_none(cls, raw: Any, name: str) -> Optional[float]:
        value = cls._get(raw, name)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_contract(cls, raw: Any) -> OptionContract:
        expiration = cls._get(raw, "expiration_date")
        if isinstance(expiration, datetime):
            expiration_date = expiration.date()
        elif isinstance(expiration, date):
            expiration_date = expiration
        else:
            expiration_date = datetime.strptime(str(expiration), "%Y-%m-%d").date()

        return OptionContract(
            symbol=str(cls._get(raw, "symbol")),
            underlying_symbol=str(cls._get(raw, "underlying_symbol") or cls._get(raw, "root_symbol")),
            contract_type=str(cls._get(raw, "type")),
            expiration_date=expiration_date,
            strike_price=float(cls._get(raw, "strike_price")),
            tradable=bool(cls._get(raw, "tradable", True)),
            open_interest=int(float(cls._get(raw, "open_interest", 0) or 0)),
            close_price=cls._float_or_none(raw, "close_price"),
        )

    @classmethod
    def _normalize_snapshot(cls, contract: OptionContract, raw: Any) -> Optional[OptionSnapshot]:
        quote = cls._get(raw, "latest_quote")
        if quote is None:
            return None

        bid = cls._float_or_none(quote, "bid_price") or 0.0
        ask = cls._float_or_none(quote, "ask_price") or 0.0
        normalized_quote = OptionQuote(
            bid_price=bid,
            ask_price=ask,
            bid_size=cls._float_or_none(quote, "bid_size"),
            ask_size=cls._float_or_none(quote, "ask_size"),
        )
        if normalized_quote.mid_price <= 0:
            return None

        greeks_raw = cls._get(raw, "greeks")
        greeks = OptionGreeks(
            delta=cls._float_or_none(greeks_raw, "delta") if greeks_raw is not None else None,
            gamma=cls._float_or_none(greeks_raw, "gamma") if greeks_raw is not None else None,
            theta=cls._float_or_none(greeks_raw, "theta") if greeks_raw is not None else None,
            vega=cls._float_or_none(greeks_raw, "vega") if greeks_raw is not None else None,
            rho=cls._float_or_none(greeks_raw, "rho") if greeks_raw is not None else None,
        )

        return OptionSnapshot(
            contract=contract,
            quote=normalized_quote,
            implied_volatility=cls._float_or_none(raw, "implied_volatility"),
            greeks=greeks,
        )
