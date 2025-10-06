import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- NEW: Import watchlist and API keys from central config ---
from config.settings import WATCHLIST
try:
    # Go up two directories from this script (training/ -> llm-advisor/) to find the root
    dotenv_path = Path(__file__).resolve().parents[1] / '.env'
    print(f"Attempting to load .env from: {dotenv_path}")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(".env file loaded successfully.")
    else:
        print("Warning: .env file not found at the expected project root location.")
        load_dotenv() # Fallback to default behavior
except Exception as e:
    print(f"Error loading .env file: {e}")
    load_dotenv()

warnings.filterwarnings('ignore')

# --- NEW: Initialize Alpaca API Client ---
try:
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
    alpaca_paper_trading = os.getenv('ALPACA_PAPER_TRADING', 'true').lower() == 'true'
    
    if not alpaca_api_key or not alpaca_secret_key:
        raise ValueError("Alpaca API keys not found in .env file.")
        
    api = tradeapi.REST(
        key_id=alpaca_api_key,
        secret_key=alpaca_secret_key,
        base_url='https://paper-api.alpaca.markets' if alpaca_paper_trading else 'https://api.alpaca.markets',
        api_version='v2'
    )
    print("✅ Alpaca API client initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Alpaca API: {e}")
    api = None

# --- REUSED CODE: Helper function ---
def _safe_divide(numerator, denominator, default=0):
    """Safe division that handles zero denominators."""
    return np.where(denominator != 0, numerator / denominator, default)

# --- NEW: Upgraded Data Fetching for Multi-Timeframe Analysis ---
def download_data(symbol):
    """Downloads daily, 4-hour, and 1-hour historical data for a given symbol from Alpaca."""
    if not api:
        print("  Alpaca API not available. Skipping data download.")
        return None, None, None
        
    print(f"  Downloading multi-timeframe data for {symbol}...")
    try:
        end_date = datetime.now().isoformat()
        
        # Daily data: 5 years
        start_1d = (datetime.now() - timedelta(days=5*365)).isoformat()
        df_1d = api.get_bars(symbol, '1Day', start=start_1d, end=end_date).df
        
        # 4-Hour data: 2 years
        start_4h = (datetime.now() - timedelta(days=2*365)).isoformat()
        df_4h = api.get_bars(symbol, '4Hour', start=start_4h, end=end_date).df
        
        # 1-Hour data: ~2 years (700 days is a safe limit for many APIs)
        start_1h = (datetime.now() - timedelta(days=700)).isoformat()
        df_1h = api.get_bars(symbol, '1Hour', start=start_1h, end=end_date).df
        
        if df_1d.empty or df_4h.empty or df_1h.empty:
            print(f"  Warning: Could not fetch complete data for {symbol}.")
            return None, None, None
            
        print(f"  Downloaded: {len(df_1d)} days, {len(df_4h)} 4H bars, {len(df_1h)} 1H bars.")
        return df_1d, df_4h, df_1h
        
    except Exception as e:
        print(f"  Error downloading data for {symbol}: {e}")
        return None, None, None

# --- NEW: Function to merge timeframes and create HTF features ---
def merge_and_enrich_data(df_1d, df_4h, df_1h):
    """Merges H4 and H1 data into the daily dataframe to create richer features."""
    print("  Merging timeframes and creating HTF features...")
    df_1d = df_1d.copy()
    
    # Pre-calculate some HTF moving averages
    df_4h['sma_20'] = df_4h['close'].rolling(20).mean()
    df_1h['volume_sma_10'] = df_1h['volume'].rolling(10).mean()

    # Create empty columns in the daily dataframe
    df_1d['h4_close_vs_sma20'] = np.nan
    df_1d['h1_eod_volume_ratio'] = np.nan

    # Iterate through each day to map HTF context
    for date, row in df_1d.iterrows():
        prev_day = date - timedelta(days=1)
        
        # Get the last 4H bar of the previous day
        last_4h_bar = df_4h[df_4h.index < date].last('1D')
        if not last_4h_bar.empty:
            last_close = last_4h_bar.iloc[-1]['close']
            last_sma = last_4h_bar.iloc[-1]['sma_20']
            df_1d.loc[date, 'h4_close_vs_sma20'] = _safe_divide(last_close - last_sma, last_sma) * 100

        # Get the last 1H bar of the previous day for volume analysis
        last_1h_bars = df_1h[(df_1h.index.date == prev_day.date()) & (df_1h.index.hour == 15)] # 3 PM EST bar
        if not last_1h_bars.empty:
            last_volume = last_1h_bars.iloc[-1]['volume']
            avg_volume = last_1h_bars.iloc[-1]['volume_sma_10']
            df_1d.loc[date, 'h1_eod_volume_ratio'] = _safe_divide(last_volume, avg_volume)
            
    return df_1d

# --- UPDATED: Main feature engineering function ---
def engineer_features(df):
    """
    Creates a comprehensive set of ICT-based and technical features.
    """
    df = df.copy()
    df.columns = [x.lower() for x in df.columns]

    # --- Shifted data for previous day's metrics ---
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)

    # --- Basic Features ---
    df['gap_pct'] = _safe_divide((df['open'] - df['prev_close']), df['prev_close']) * 100
    df['gap_size_abs'] = abs(df['gap_pct'])
    
    # --- ICT & Technical Features ---
    df['prev_day_range_pct'] = _safe_divide((df['prev_high'] - df['prev_low']), df['prev_close']) * 100
    df['current_range_pct'] = _safe_divide((df['high'] - df['low']), df['open']) * 100
    vol_5d = df['volume'].rolling(5, min_periods=1).mean()
    df['volume_ratio_5d'] = _safe_divide(df['volume'], vol_5d, 1.0)
    df['swept_prev_high'] = np.where(df['high'] > df['prev_high'], 1, 0)
    df['swept_prev_low'] = np.where(df['low'] < df['prev_low'], 1, 0)
    df['body_size'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['body_pct'] = _safe_divide(df['body_size'], df['total_range']) * 100
    df['upper_wick_pct'] = _safe_divide(df['high'] - df[['open', 'close']].max(axis=1), df['total_range']) * 100
    df['lower_wick_pct'] = _safe_divide(df[['open', 'close']].min(axis=1) - df['low'], df['total_range']) * 100
    range_nonzero = np.maximum(df['high'] - df['low'], 1e-10)
    df['close_vs_range'] = _safe_divide((df['close'] - df['low']), range_nonzero)
    
    # Clean and drop NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# --- NEW: Function to define the target variable (Bias) ---
def define_bias_label(df, threshold=0.5):
    """Determines the actual bias (Bullish/Bearish/Choppy) for a given day."""
    daily_move_pct = _safe_divide(df['close'] - df['open'], df['open']) * 100
    
    conditions = [
        daily_move_pct > threshold,
        daily_move_pct < -threshold
    ]
    choices = ['bullish', 'bearish']
    
    df['bias_label'] = np.select(conditions, choices, default='choppy')
    return df

# --- UPDATED: Model training function with new features ---
def train_and_save_model(symbol, data_with_features):
    """Trains a Random Forest model and saves it to a file."""
    
    # --- UPDATED: Feature list now includes the new HTF features ---
    feature_names = [
        # Original features
        'gap_pct', 'gap_size_abs', 'prev_day_range_pct', 'current_range_pct',
        'volume_ratio_5d', 'swept_prev_high', 'swept_prev_low', 'body_pct',
        'upper_wick_pct', 'lower_wick_pct', 'close_vs_range',
        # New multi-timeframe features
        'h4_close_vs_sma20', 'h1_eod_volume_ratio'
    ]
    
    # Ensure all feature columns exist, fill missing with 0
    for feature in feature_names:
        if feature not in data_with_features.columns:
            data_with_features[feature] = 0
            
    X = data_with_features[feature_names]
    y = data_with_features['bias_label']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, class_weight='balanced'
    )
    
    print("  Training model on enriched data...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Model for {symbol} trained with Test Accuracy: {accuracy:.2%}")
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, f'{symbol}_daily_bias.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, f'{symbol}_label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print(f"  Models for {symbol} saved successfully to '{models_dir}'.")

# --- UPDATED: Main execution block with new workflow ---
def run_training_pipeline():
    """Main function to run the entire training pipeline for all symbols."""
    
    print("="*60)
    print("Starting Multi-Timeframe Daily Bias Model Training Pipeline")
    print(f"Symbols to process: {', '.join(WATCHLIST)}")
    print("="*60)

    for symbol in WATCHLIST:
        print(f"\n--- Processing Symbol: {symbol} ---")
        
        # 1. Download multi-timeframe data
        df_1d, df_4h, df_1h = download_data(symbol)
        if df_1d is None:
            continue
            
        # 2. Merge timeframes and create HTF features
        enriched_data = merge_and_enrich_data(df_1d, df_4h, df_1h)
        
        # 3. Engineer original daily features
        data_with_features = engineer_features(enriched_data)
        
        # 4. Define bias labels
        labeled_data = define_bias_label(data_with_features)
        
        # 5. Train and save model
        if not labeled_data.empty:
            train_and_save_model(symbol, labeled_data)
        else:
            print("  Not enough data to train model after feature engineering.")

    print("\n--- Pipeline Complete: All models have been trained and saved. ---")

if __name__ == "__main__":
    if api:
        run_training_pipeline()
    else:
        print("\nExiting: Alpaca API client could not be initialized.")
        sys.exit(1)

