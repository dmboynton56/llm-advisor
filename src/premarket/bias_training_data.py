"""Read-only Alpaca history acquisition for reproducible bias training."""
from datetime import datetime
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

TIMEFRAMES = {
    "daily": TimeFrame.Day,
    "20min": TimeFrame(20, TimeFrameUnit.Minute),
}


def with_higher_timeframes(frames: dict) -> dict:
    """Derive UTC-aligned higher bars identically in training and inference.

    Alpaca bars are start-timestamped. Partial final aggregates are kept here;
    the shared feature extractor rejects them until their full duration ends.
    """
    minute = frames["20min"].tz_convert("UTC")
    aggregation = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for name, interval in (("hourly", "1h"), ("4hour", "4h")):
        frames[name] = minute.resample(interval, origin="epoch").agg(aggregation).dropna(subset=["close"]).tz_convert("America/New_York")
    return frames


def fetch_bias_inference_frames(client, symbol: str, metadata: dict, now=None) -> dict:
    """Use the artifact's feed and cutoff, with free delayed-SIP compatibility.

    Feature history ends just before 09:00 ET. At the scheduled 09:20 premarket
    run it is safely outside SIP's 15-minute restricted recent-history window.
    This does not change the trading/quote feed or buy a data subscription.
    """
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="America/New_York")
    cutoff = pd.Timestamp(f"{now.tz_convert('America/New_York').date()} {metadata['feature_cutoff_et']}", tz="America/New_York")
    if now < cutoff + pd.Timedelta(minutes=16 if metadata["data_feed"] == "sip" else 0):
        raise ValueError("Daily bias requested before historical feed is available")
    frames = {}
    for name, timeframe in TIMEFRAMES.items():
        request = StockBarsRequest(
            symbol_or_symbols=[symbol], timeframe=timeframe,
            # ATR needs daily history; higher-timeframe features use just 20
            # completed bars. Avoid downloading months of intraday data daily.
            start=(cutoff - pd.Timedelta(days=120 if name == "daily" else 14)).to_pydatetime(),
            end=(cutoff - pd.Timedelta(microseconds=1)).to_pydatetime(),
            feed=DataFeed(metadata["data_feed"]), adjustment=Adjustment.SPLIT,
        )
        frame = client.get_stock_bars(request).df
        if frame.empty:
            raise ValueError(f"Missing daily-bias {name} history for {symbol}")
        frames[name] = frame.xs(symbol, level=0).sort_index().tz_convert("America/New_York")
    return with_higher_timeframes(frames)


def fetch_history(client, symbol: str, start: datetime, end: datetime,
                  cache_dir: Path, feed: str = "sip") -> dict[str, pd.DataFrame]:
    """Cache exact split-adjusted input frames; SDK paginates historical bars.

    This client is a market-data client, never an order/execution client.
    The cache key includes dates, feed, adjustment and timeframe.
    """
    frames = {}
    for name, timeframe in TIMEFRAMES.items():
        path = cache_dir / f"{symbol}_{feed}_split_{start.date()}_{end.date()}_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, index_col="timestamp", parse_dates=True)
            frame.index = pd.to_datetime(frame.index, utc=True)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol], timeframe=timeframe,
                start=start, end=end, feed=DataFeed(feed), adjustment=Adjustment.SPLIT,
            )
            frame = client.get_stock_bars(request).df
            if frame.empty:
                raise ValueError(f"No {feed} {name} history for {symbol}")
            frame = frame.xs(symbol, level=0).sort_index()
            if frame.index.has_duplicates:
                raise ValueError(f"Duplicate {name} timestamps for {symbol}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index_label="timestamp")
            temporary.replace(path)
        if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Invalid cached {name} timestamps for {symbol}")
        frames[name] = frame.tz_convert("America/New_York")
        print(f"{symbol} {name}: {len(frame)} bars", flush=True)
    return with_higher_timeframes(frames)
