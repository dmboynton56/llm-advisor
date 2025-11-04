"""Mock data client for backtesting with historical data."""
from datetime import datetime
from typing import Dict, List, Any


class MockDataClient:
    """Simulates AlpacaDataClient but uses prefetched historical data."""
    
    def __init__(self, bars_cache: Dict[str, Dict[str, List[Dict[str, Any]]]]):
        """
        Initialize mock data client with prefetched bars.
        
        Args:
            bars_cache: Prefetched bars dict {symbol: {"bars_1m": [...], "bars_5m": [...]}}
        """
        self.bars_cache = bars_cache
        self.current_time = None  # Simulated current time
    
    def set_current_time(self, dt: datetime):
        """Set simulated current time."""
        self.current_time = dt
    
    def fetch_window_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Slice historical bars based on time window.
        
        Args:
            symbols: List of symbols to fetch
            start: Start datetime (UTC)
            end: End datetime (UTC)
            
        Returns:
            Dict of {symbol: {"bars_1m": [...], "bars_5m": [...]}}
        """
        result = {}
        for sym in symbols:
            if sym not in self.bars_cache:
                result[sym] = {"bars_1m": [], "bars_5m": []}
                continue
            
            bars_1m = self._filter_bars(
                self.bars_cache[sym]["bars_1m"], start, end
            )
            bars_5m = self._filter_bars(
                self.bars_cache[sym]["bars_5m"], start, end
            )
            result[sym] = {"bars_1m": bars_1m, "bars_5m": bars_5m}
        
        return result
    
    def _filter_bars(self, bars: List[Dict[str, Any]], start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Filter bars by timestamp."""
        filtered = []
        for bar in bars:
            bar_time_str = bar.get("t", "")
            if not bar_time_str:
                continue
            
            try:
                # Handle ISO format timestamps
                if "T" in bar_time_str:
                    bar_time = datetime.fromisoformat(bar_time_str.replace("Z", "+00:00"))
                else:
                    bar_time = datetime.fromisoformat(bar_time_str)
                
                # Handle timezone-naive times
                if bar_time.tzinfo is None:
                    # Assume UTC if no timezone
                    import pytz
                    bar_time = pytz.UTC.localize(bar_time)
                
                if start <= bar_time <= end:
                    filtered.append(bar)
            except (ValueError, AttributeError):
                continue
        
        return filtered

