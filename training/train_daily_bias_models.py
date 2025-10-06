import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta, timezone
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from pathlib import Path

# ---- NEW: modern Alpaca SDK imports ----
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import watchlist from central config ---
from config.settings import WATCHLIST

# ---- Load .env (project root first) ----
try:
    dotenv_path = Path(__file__).resolve().parents[1] / '.env'
    print(f"Attempting to load .env from: {dotenv_path}")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(".env file loaded successfully.")
    else:
        print("Warning: .env file not found at the expected project root location.")
        load_dotenv()  # Fallback
except Exception as e:
    print(f"Error loading .env file: {e}")
    load_dotenv()

warnings.filterwarnings('ignore')

# ---- Initialize Alpaca clients (modern) ----
trading = None
data = None
try:
    # Keep your existing env var names
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    paper = (os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true")

    if not api_key or not api_secret:
        raise ValueError("Alpaca API keys not found in .env file.")

    # Trading client (toggle paper vs live)
    trading = TradingClient(api_key, api_secret, paper=paper)

    # Data client (keys optional for some feeds; we pass keys for full access)
    data = StockHistoricalDataClient(api_key, api_secret)

    # Choose feed via env, default to SIP if available; otherwise IEX
    # ALPACA_DATA_FEED=sip|iex
    feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    FEED = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX

    print("✅ Alpaca clients initialized (alpaca-py). Data feed:", FEED.value)
except Exception as e:
    print(f"❌ Error initializing Alpaca clients: {e}")
    trading, data = None, None

# --- Helper: safe divide ---
def _safe_divide(numerator, denominator, default=0):
    return np.where(denominator != 0, numerator / denominator, default)

# --- Helper: normalize bars DF to a simple DatetimeIndex for a single symbol ---
def _as_single_symbol_df(bars_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    alpaca-py returns a MultiIndex (symbol, timestamp). This flattens to a DatetimeIndex for the given symbol.
    """
    if bars_df is None or bars_df.empty:
        return pd.DataFrame()
    if bars_df.index.nlevels == 2:
        try:
            df = bars_df.xs(symbol, level=0).copy()
        except KeyError:
            return pd.DataFrame()
    else:
        df = bars_df.copy()
    # Ensure datetime is tz-aware (usually UTC from alpaca-py)
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    return df.sort_index()

# --- NEW: Upgraded Data Fetching for Multi-Timeframe Analysis ---
def download_data(symbol):
    """Downloads daily, 4-hour, and 1-hour historical data for a given symbol from Alpaca (alpaca-py)."""
    if data is None:
        print("  Alpaca data client not available. Skipping data download.")
        return None, None, None

    print(f"  Downloading multi-timeframe data for {symbol}...")
    try:
        # Use timezone-aware UTC datetimes for requests
        end_dt = datetime.now(timezone.utc)
        # 5y daily, 2y 4H, ~700d 1H (roughly 2y)
        start_1d = end_dt - timedelta(days=5 * 365)
        start_4h = end_dt - timedelta(days=2 * 365)
        start_1h = end_dt - timedelta(days=700)

        # Daily bars
        req_1d = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_1d,
            end=end_dt,
            feed=FEED,
            adjustment='raw',
            limit=None,
        )
        bars_1d = data.get_stock_bars(req_1d).df
        df_1d = _as_single_symbol_df(bars_1d, symbol)

        # 4-Hour bars (TimeFrame with unit/multiplier)
        req_4h = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(4, TimeFrameUnit.Hour),
            start=start_4h,
            end=end_dt,
            feed=FEED,
            adjustment='raw',
            limit=None,
        )
        bars_4h = data.get_stock_bars(req_4h).df
        df_4h = _as_single_symbol_df(bars_4h, symbol)

        # 1-Hour bars
        req_1h = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start_1h,
            end=end_dt,
            feed=FEED,
            adjustment='raw',
            limit=None,
        )
        bars_1h = data.get_stock_bars(req_1h).df
        df_1h = _as_single_symbol_df(bars_1h, symbol)

        if df_1d.empty or df_4h.empty or df_1h.empty:
            print(f"  Warning: Could not fetch complete data for {symbol}.")
            return None, None, None

        print(f"  Downloaded: {len(df_1d)} days, {len(df_4h)} 4H bars, {len(df_1h)} 1H bars.")
        return df_1d, df_4h, df_1h

    except Exception as e:
        if "is not a valid symbol" in str(e).lower():
            print(f"  Error: '{symbol}' is not a valid stock/ETF symbol for Alpaca. Skipping.")
        else:
            print(f"  Error downloading data for {symbol}: {e}")
        return None, None, None

# --- Merge timeframes and create HTF features ---
def merge_and_enrich_data(df_1d, df_4h, df_1h):
    """Merges H4 and H1 data into the daily dataframe to create richer features."""
    print("  Merging timeframes and creating HTF features...")
    df_1d = df_1d.copy()

    # Pre-calc HTF indicators on their native frames
    df_4h = df_4h.copy()
    df_1h = df_1h.copy()
    df_4h['sma_20'] = df_4h['close'].rolling(20).mean()
    df_1h['volume_sma_10'] = df_1h['volume'].rolling(10).mean()

    # Create target columns on daily frame
    df_1d['h4_close_vs_sma20'] = np.nan
    df_1d['h1_eod_volume_ratio'] = np.nan

    # Timezone handling: Alpaca returns UTC. Convert for the “3 PM ET” logic.
    h1_eastern = df_1h.tz_convert("US/Eastern")

    # Iterate through each day and map HTF context
    for date, _ in df_1d.iterrows():
        # 'date' is a UTC timestamp for the end of day bar
        # Use prior day (US/Eastern) to find the 3pm ET hour bar
        prev_day_utc = (pd.Timestamp(date).tz_convert("US/Eastern") - pd.Timedelta(days=1)).date()

        # Last 4H bar strictly before the daily bar time
        last_4h_bar = df_4h[df_4h.index < date].last("1D")
        if not last_4h_bar.empty:
            last_close = last_4h_bar.iloc[-1]['close']
            last_sma = last_4h_bar.iloc[-1]['sma_20']
            df_1d.loc[date, 'h4_close_vs_sma20'] = float(_safe_divide(last_close - last_sma, last_sma) * 100)

        # The 3 PM ET 1H bar of the previous calendar day in ET
        mask_prev_day_3pm = (h1_eastern.index.date == prev_day_utc) & (h1_eastern.index.hour == 15)
        last_1h_bars = h1_eastern[mask_prev_day_3pm]
        if not last_1h_bars.empty:
            last_volume = last_1h_bars.iloc[-1]['volume']
            avg_volume = last_1h_bars.iloc[-1]['volume_sma_10']
            df_1d.loc[date, 'h1_eod_volume_ratio'] = float(_safe_divide(last_volume, avg_volume, default=np.nan))

    return df_1d

# --- Daily feature engineering ---
def engineer_features(df):
    df = df.copy()
    # Ensure expected lower-case columns
    df.columns = [x.lower() for x in df.columns]

    # Previous day reference
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)

    # Basic features
    df['gap_pct'] = _safe_divide((df['open'] - df['prev_close']), df['prev_close']) * 100
    df['gap_size_abs'] = abs(df['gap_pct'])

    # ICT-ish / technical features
    df['prev_day_range_pct'] = _safe_divide((df['prev_high'] - df['prev_low']), df['prev_close']) * 100
    df['current_range_pct'] = _safe_divide((df['high'] - df['low']), df['open']) * 100
    vol_5d = df['volume'].rolling(5, min_periods=1).mean()
    df['volume_ratio_5d'] = _safe_divide(df['volume'], vol_5d, 1.0)
    df['swept_prev_high'] = (df['high'] > df['prev_high']).astype(int)
    df['swept_prev_low']  = (df['low']  < df['prev_low']).astype(int)
    df['body_size'] = (df['close'] - df['open']).abs()
    df['total_range'] = df['high'] - df['low']
    df['body_pct'] = _safe_divide(df['body_size'], df['total_range']) * 100
    df['upper_wick_pct'] = _safe_divide(df['high'] - df[['open', 'close']].max(axis=1), df['total_range']) * 100
    df['lower_wick_pct'] = _safe_divide(df[['open', 'close']].min(axis=1) - df['low'], df['total_range']) * 100
    range_nonzero = np.maximum(df['high'] - df['low'], 1e-10)
    df['close_vs_range'] = _safe_divide((df['close'] - df['low']), range_nonzero)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# --- Labeling ---
def define_bias_label(df, threshold=0.5):
    daily_move_pct = _safe_divide(df['close'] - df['open'], df['open']) * 100
    conditions = [daily_move_pct > threshold, daily_move_pct < -threshold]
    choices = ['bullish', 'bearish']
    df = df.copy()
    df['bias_label'] = np.select(conditions, choices, default='choppy')
    return df

# --- Train & save ---
def train_and_save_model(symbol, data_with_features):
    feature_names = [
        'gap_pct', 'gap_size_abs', 'prev_day_range_pct', 'current_range_pct',
        'volume_ratio_5d', 'swept_prev_high', 'swept_prev_low', 'body_pct',
        'upper_wick_pct', 'lower_wick_pct', 'close_vs_range',
        'h4_close_vs_sma20', 'h1_eod_volume_ratio'
    ]
    for f in feature_names:
        if f not in data_with_features.columns:
            data_with_features[f] = 0

    X = data_with_features[feature_names]
    y = data_with_features['bias_label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, class_weight='balanced'
    )
    print("  Training model on enriched data...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # --- Built-in (Gini) feature importances
    importances = model.feature_importances_
    feat_ranking = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
    )

    print("\nTop feature importances (model intrinsic):")
    print(feat_ranking.to_string(float_format=lambda v: f"{v:.4f}"))

    # Save to disk per symbol
    reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, f"{symbol}_rf_feature_importances.csv")
    feat_ranking.to_csv(csv_path)

    # PNG (top 20)
    import matplotlib
    matplotlib.use("Agg")  # safe for headless envs
    import matplotlib.pyplot as plt

    plt.figure()
    feat_ranking.head(20).plot(kind="bar")
    plt.title(f"{symbol} – RF feature importances (top 20)")
    plt.ylabel("Importance")
    plt.tight_layout()
    png_path = os.path.join(reports_dir, f"{symbol}_rf_feature_importances_top20.png")
    plt.savefig(png_path)
    plt.close()

    print(f"  Model for {symbol} trained with Test Accuracy: {acc:.2%}")

    

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, f'{symbol}_daily_bias.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, f'{symbol}_label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    print(f"  Models for {symbol} saved successfully to '{models_dir}'.")

# --- Pipeline ---
def run_training_pipeline():
    print("="*60)
    print("Starting Multi-Timeframe Daily Bias Model Training Pipeline")
    print(f"Symbols to process: {', '.join(WATCHLIST)}")
    print("="*60)

    for symbol in WATCHLIST:
        print(f"\n--- Processing Symbol: {symbol} ---")
        df_1d, df_4h, df_1h = download_data(symbol)
        if df_1d is None:
            continue

        enriched = merge_and_enrich_data(df_1d, df_4h, df_1h)
        feats = engineer_features(enriched)
        labeled = define_bias_label(feats)

        if not labeled.empty:
            train_and_save_model(symbol, labeled)
        else:
            print("  Not enough data to train model after feature engineering.")

    print("\n--- Pipeline Complete: All models have been trained and saved. ---")

if __name__ == "__main__":
    if data is not None:
        run_training_pipeline()
    else:
        print("\nExiting: Alpaca data client could not be initialized.")
        sys.exit(1)
