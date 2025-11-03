"""Alpaca API wrapper for data and trading."""
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
import pandas as pd

load_dotenv()


class AlpacaDataClient:
    """Alpaca data client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, feed: Optional[str] = None):
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not api_secret:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")
        
        self.client = StockHistoricalDataClient(api_key, api_secret)
        feed_env = (feed or os.getenv("ALPACA_DATA_FEED") or "iex").lower()
        self.feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX
    
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
        df = self.client.get_stock_bars(req).df
        
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
        end: datetime
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Fetch 1m and 5m bars, returning as records."""
        df_1m = self.fetch_bars(symbols, start, end, TimeFrame(1, TimeFrameUnit.Minute))
        df_5m = self.fetch_bars(symbols, start, end, TimeFrame(5, TimeFrameUnit.Minute))
        
        return {
            sym: {
                "bars_1m": self._df_to_records(df_1m.get(sym, pd.DataFrame())),
                "bars_5m": self._df_to_records(df_5m.get(sym, pd.DataFrame())),
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

