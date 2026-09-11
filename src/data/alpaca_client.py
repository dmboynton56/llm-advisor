"""Alpaca API wrapper for data and trading."""
import inspect
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
import pandas as pd
import requests

from src.utils.env_sanitize import getenv_strip

load_dotenv()
logger = logging.getLogger(__name__)


class AlpacaDataUnavailable(RuntimeError):
    """Raised when Alpaca market data remains unavailable after bounded retries."""


# Live-loop ticks can skip an iteration on AlpacaDataUnavailable. Process start
# cannot: RollingStats need a historical seed. Keep retrying until this deadline,
# then fail closed. Override with ALPACA_SEED_FETCH_DEADLINE_SECONDS.
DEFAULT_SEED_FETCH_DEADLINE_SECONDS = 900.0
SEED_RETRY_INITIAL_WAIT_SECONDS = 2.0
SEED_RETRY_MAX_WAIT_SECONDS = 32.0


class AlpacaDataClient:
    """Alpaca data client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, feed: Optional[str] = None):
        api_key = api_key or getenv_strip("ALPACA_API_KEY")
        api_secret = api_secret or getenv_strip("ALPACA_SECRET_KEY")
        
        if not api_key or not api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")

        self.timeout_seconds = self._env_float("ALPACA_DATA_TIMEOUT_SECONDS", 20.0)
        self.max_attempts = max(1, self._env_int("ALPACA_DATA_MAX_ATTEMPTS", 2))
        self.retry_wait_seconds = max(0.0, self._env_float("ALPACA_DATA_RETRY_WAIT_SECONDS", 2.0))
        client_kwargs = {
            "api_key": api_key,
            "secret_key": api_secret,
        }
        sdk_params = inspect.signature(StockHistoricalDataClient.__init__).parameters
        if "retry_attempts" in sdk_params:
            client_kwargs["retry_attempts"] = self._env_int("ALPACA_SDK_RETRY_ATTEMPTS", 2)
        if "retry_wait_seconds" in sdk_params:
            client_kwargs["retry_wait_seconds"] = self._env_int("ALPACA_SDK_RETRY_WAIT_SECONDS", 2)
        self.client = StockHistoricalDataClient(**client_kwargs)
        self._install_request_timeout()
        feed_env = (feed or os.getenv("ALPACA_DATA_FEED") or "iex").lower()
        self.feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _install_request_timeout(self) -> None:
        """Alpaca's SDK does not expose a timeout, so default one at the session edge."""
        session = getattr(self.client, "_session", None)
        if session is None or getattr(session, "_llm_advisor_timeout_installed", False):
            return

        original_request = session.request
        timeout_seconds = self.timeout_seconds

        def request_with_timeout(method: str, url: str, **kwargs: Any):
            kwargs.setdefault("timeout", timeout_seconds)
            return original_request(method, url, **kwargs)

        session.request = request_with_timeout
        setattr(session, "_llm_advisor_timeout_installed", True)

    @staticmethod
    def _retryable_api_error(exc: APIError) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in ("429", "504", "timeout", "timed out"))
    
    def fetch_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame,
        adjustment: Adjustment = Adjustment.SPLIT
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols."""
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=self.feed,
            adjustment=adjustment
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                df = self.client.get_stock_bars(req).df
                break
            except requests.exceptions.RequestException as exc:
                last_exc = exc
            except APIError as exc:
                if not self._retryable_api_error(exc):
                    raise
                last_exc = exc

            if attempt < self.max_attempts:
                logger.warning(
                    "Alpaca bar fetch failed for %s %s (%s/%s): %s",
                    ",".join(symbols),
                    timeframe,
                    attempt,
                    self.max_attempts,
                    last_exc,
                )
                time.sleep(self.retry_wait_seconds)
        else:
            raise AlpacaDataUnavailable(
                "Alpaca stock bars unavailable after "
                f"{self.max_attempts} attempts for {','.join(symbols)} {timeframe}: {last_exc}"
            ) from last_exc
        
        if df is None or df.empty:
            return {sym: pd.DataFrame() for sym in symbols}
        
        # Extract per-symbol DataFrames
        result = {}
        for sym in symbols:
            try:
                sdf = df.xs(sym, level="symbol").sort_index()
            except (KeyError, AttributeError):
                sdf = pd.DataFrame()
            result[sym] = sdf
        
        return result
    
    def fetch_window_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        *,
        include_1m: bool = True,
        include_5m: bool = True,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Fetch 1m and/or 5m bars, returning as records.

        Live seeding only needs one timeframe per window (prev-day 1m,
        premarket 5m). Fetching the unused timeframe doubles Alpaca calls
        and can 504 after the needed bars already succeeded.
        """
        if not include_1m and not include_5m:
            raise ValueError("fetch_window_bars requires include_1m or include_5m")

        df_1m: Dict[str, pd.DataFrame] = {}
        df_5m: Dict[str, pd.DataFrame] = {}
        if include_1m:
            df_1m = self.fetch_bars(symbols, start, end, TimeFrame(1, TimeFrameUnit.Minute))
        if include_5m:
            df_5m = self.fetch_bars(symbols, start, end, TimeFrame(5, TimeFrameUnit.Minute))

        return {
            sym: {
                "bars_1m": self._df_to_records(df_1m.get(sym, pd.DataFrame())) if include_1m else [],
                "bars_5m": self._df_to_records(df_5m.get(sym, pd.DataFrame())) if include_5m else [],
            }
            for sym in symbols
        }
    
    def _df_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of records."""
        if df.empty:
            return []
        
        records = []
        for idx, row in df.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else pd.to_datetime(idx)
            records.append({
                "t": timestamp.isoformat(),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": float(row.get("volume", 0.0)),
            })
        return records


def seed_fetch_deadline_seconds() -> float:
    raw = os.getenv("ALPACA_SEED_FETCH_DEADLINE_SECONDS")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_SEED_FETCH_DEADLINE_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SEED_FETCH_DEADLINE_SECONDS


def _invoke_fetch_window_bars(
    client: Any,
    symbols: List[str],
    start: datetime,
    end: datetime,
    include_1m: bool,
    include_5m: bool,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    kwargs: Dict[str, Any] = {}
    try:
        params = inspect.signature(client.fetch_window_bars).parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        if "include_1m" in params or accepts_kwargs:
            kwargs["include_1m"] = include_1m
            kwargs["include_5m"] = include_5m
    except (TypeError, ValueError):
        pass
    return client.fetch_window_bars(symbols, start, end, **kwargs)


def fetch_seed_window_bars(
    client: Any,
    symbols: List[str],
    start: datetime,
    end: datetime,
    *,
    include_1m: bool = True,
    include_5m: bool = True,
    deadline_seconds: Optional[float] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Fetch historical bars required to seed a live segment.

    In-loop ticks skip on AlpacaDataUnavailable. Startup cannot: fail closed
    only after deadline_seconds of exponential backoff. After a multi-symbol
    batch failure, try each symbol once before waiting to retry the round.
    """
    if deadline_seconds is None:
        deadline_seconds = seed_fetch_deadline_seconds()

    def try_fetch() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        try:
            return _invoke_fetch_window_bars(
                client, symbols, start, end, include_1m, include_5m
            )
        except AlpacaDataUnavailable as batch_exc:
            if len(symbols) <= 1:
                raise
            logger.warning(
                "Seed batch bar fetch failed for %s; retrying per symbol: %s",
                ",".join(symbols),
                batch_exc,
            )
            merged: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for sym in symbols:
                piece = _invoke_fetch_window_bars(
                    client, [sym], start, end, include_1m, include_5m
                )
                merged[sym] = piece[sym]
            return merged

    started = monotonic()
    delay = SEED_RETRY_INITIAL_WAIT_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            return try_fetch()
        except AlpacaDataUnavailable as exc:
            elapsed = monotonic() - started
            remaining = deadline_seconds - elapsed
            if remaining <= 0:
                raise AlpacaDataUnavailable(
                    "Alpaca seed bars unavailable after "
                    f"{attempt} rounds / {elapsed:.0f}s for {','.join(symbols)}: {exc}"
                ) from exc
            wait = min(delay, remaining)
            logger.warning(
                "Alpaca seed bar fetch failed (round %s, %.0fs elapsed, %.0fs left): "
                "%s; retrying in %.1fs",
                attempt,
                elapsed,
                remaining,
                exc,
                wait,
            )
            sleep(wait)
            delay = min(delay * 2, SEED_RETRY_MAX_WAIT_SECONDS)
