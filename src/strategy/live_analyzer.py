#!/usr/bin/env python3
"""
Liquidity Flow Agent — Live Analyzer (V2)
Continuously runs during market hours:
1) loads pre-market context
2) fetches multi-timeframe price data (1m/5m/1h/4h)
3) builds the analysis prompt from templates + context
4) queries multiple LLMs (OpenAI, Gemini) for JSON verdicts
5) logs the best high-confidence signal and triggers execution module
"""

import os
import sys
import json
import time
from datetime import datetime, time as dt_time, timedelta
import pytz
from dotenv import load_dotenv

# --- LLM Clients ---
from openai import OpenAI
import google.generativeai as genai

# Add project root to path for local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Settings
from config.settings import (
    WATCHLIST,
    ANALYSIS_INTERVAL_SECONDS,
    CONFIDENCE_THRESHOLD,
    TRADING_WINDOW_START,
    TRADING_WINDOW_END,
    MAX_TRADES_PER_DAY,
)

# Alpaca SDK (alpaca-py)
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# Execution bridge (Phase 3)
from src.execution.order_manager import execute_trade_signal

ET_TZ = pytz.timezone("US/Eastern")


# ----------------------------
# Clients & Environment
# ----------------------------
def _load_env_and_clients():
    """Load .env and initialize market data + trading + LLM clients."""
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    # Alpaca creds
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("Alpaca API credentials not found.")

    # Market data client
    data_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)

    # Trading client (paper)
    # You can also set APCA_API_BASE_URL=https://paper-api.alpaca.markets
    trading_client = TradingClient(alpaca_key, alpaca_secret, paper=True)

    # Data feed (IEX default, SIP if enabled on your account)
    feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX

    # OpenAI
    clients = {"alpaca": data_client, "trading": trading_client, "feed": feed}
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        clients["openai"] = OpenAI(api_key=openai_key)

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        clients["gemini"] = genai.GenerativeModel("gemini-1.5-flash")

    return clients


def _get_todays_context_path():
    """Return absolute path to today's condensed session context."""
    today_str = datetime.now(ET_TZ).strftime("%Y-%m-%d")
    return os.path.join(
        PROJECT_ROOT, "data", "daily_news", today_str, "processed", "session_context.json"
    )


# ----------------------------
# Data helpers (bars → records)
# ----------------------------
def _bars_to_records(df, symbol):
    """
    Convert MultiIndex bars DF to list[dict] with ET timestamps (ISO).
    Handles both MultiIndex (symbol, timestamp) and single DatetimeIndex.
    """
    import pandas as pd

    if df is None or df.empty:
        return []
    sdf = df.xs(symbol) if isinstance(df.index, pd.MultiIndex) else df
    sdf = sdf.tz_convert(ET_TZ).reset_index().rename(columns={"timestamp": "timestamp_et"})
    sdf["timestamp_et"] = sdf["timestamp_et"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return sdf.to_dict(orient="records")


def _fetch_live_data(symbol, data_client, feed):
    """Fetch rolling windows for 1m/5m/1h/4h."""
    end_utc = datetime.now(pytz.utc)
    data = {}

    # 1m / 5m — last 60 minutes
    for label, tf in (("ltf_1m", TimeFrame.Minute), ("ltf_5m", TimeFrame(5, TimeFrameUnit.Minute))):
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=end_utc - timedelta(minutes=60),
            end=end_utc,
            feed=feed,
            adjustment=Adjustment.RAW,
        )
        bars = data_client.get_stock_bars(req).df
        data[label] = _bars_to_records(bars, symbol)

    # 1h — last 48 hours
    req_1h = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Hour,
        start=end_utc - timedelta(days=2),
        end=end_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    data["htf_1h"] = _bars_to_records(data_client.get_stock_bars(req_1h).df, symbol)

    # 4h — last 7 days
    req_4h = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(4, TimeFrameUnit.Hour),
        start=end_utc - timedelta(days=7),
        end=end_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    data["htf_4h"] = _bars_to_records(data_client.get_stock_bars(req_4h).df, symbol)

    return data


# ----------------------------
# Prompting
# ----------------------------
def _construct_prompt(daily_context, realtime_data, prompt_template_path):
    """
    Build the main analysis prompt.
    prompts/prompts.json must include:
      { "main_analysis_prompt": { "content": "... {{daily_context}} ... {{realtime_price_data}} ..." } }
    """
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = json.load(f)["main_analysis_prompt"]["content"]

    context_str = json.dumps(daily_context, separators=(",", ":"))
    realtime_str = json.dumps(realtime_data, separators=(",", ":"))

    prompt = template.replace("{{daily_context}}", context_str)
    prompt = prompt.replace("{{realtime_price_data}}", realtime_str)
    return prompt


def get_all_llm_opinions(prompt, clients):
    """Query all configured LLMs; expect strict JSON verdicts."""
    opinions = {}

    # OpenAI (JSON mode)
    if "openai" in clients:
        try:
            response = clients["openai"].chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            opinions["openai"] = json.loads(response.choices[0].message.content)
        except Exception as e:
            opinions["openai"] = {"error": str(e)}

    # Gemini
    if "gemini" in clients:
        try:
            resp = clients["gemini"].generate_content(prompt)
            cleaned = resp.text.strip().replace("```json", "").replace("```", "")
            opinions["gemini"] = json.loads(cleaned)
        except Exception as e:
            opinions["gemini"] = {"error": str(e)}

    return opinions


# ----------------------------
# Main loop
# ----------------------------
def run_analysis_loop():
    print("Starting Liquidity Flow Agent - Live Analysis Engine (V2)...")
    clients = _load_env_and_clients()

    # Inputs
    context_path = _get_todays_context_path()
    prompt_path = os.path.join(PROJECT_ROOT, "prompts", "prompts.json")

    try:
        with open(context_path, "r", encoding="utf-8") as f:
            daily_context = json.load(f)
        print("Successfully loaded pre-market context.")
    except FileNotFoundError:
        print(
            f"ERROR: Pre-market context file not found at {context_path}.\n"
            "Please run the pre-market pipeline first."
        )
        return

    # Trading window
    start_time = dt_time.fromisoformat(TRADING_WINDOW_START)  # e.g., "09:30"
    end_time = dt_time.fromisoformat(TRADING_WINDOW_END)      # e.g., "12:00"
    trades_today = 0

    while True:
        current_time_et = datetime.now(ET_TZ).time()

        # Outside the window? idle a minute
        if not (start_time <= current_time_et < end_time):
            print(
                f"Outside trading window ({TRADING_WINDOW_START}-{TRADING_WINDOW_END} ET). Pausing...",
                end="\r",
            )
            time.sleep(60)
            continue

        # Hit daily trade cap?
        if trades_today >= MAX_TRADES_PER_DAY:
            print(
                f"Daily trade limit of {MAX_TRADES_PER_DAY} reached. Halting new trade analysis for today."
            )
            time.sleep(300)
            continue

        print(f"\n--- New Analysis Cycle @ {current_time_et.strftime('%H:%M:%S')} ET ---")

        for symbol in WATCHLIST:
            print(f"  Analyzing {symbol}...")

            realtime_data = _fetch_live_data(symbol, clients["alpaca"], clients["feed"])
            prompt = _construct_prompt(daily_context, realtime_data, prompt_path)

            all_opinions = get_all_llm_opinions(prompt, clients)

            best_signal = None
            highest_conf = 0

            for model, response in all_opinions.items():
                if not isinstance(response, dict) or "error" in response:
                    err = response.get("error", "unknown") if isinstance(response, dict) else "invalid"
                    print(f"  > {model.capitalize()} Verdict for {symbol}: ERROR - {err[:120]}...")
                    continue

                signal = response.get("trade_signal", "none")
                conf = int(response.get("confidence", 0))
                print(f"  > {model.capitalize()} Verdict for {symbol}: Signal='{signal}', Confidence={conf}%")

                if signal != "none" and conf > highest_conf:
                    highest_conf = conf
                    best_signal = response
                    best_signal["signal_source"] = model

            if best_signal and highest_conf >= CONFIDENCE_THRESHOLD:
                print(
                    f"  HIGH CONFIDENCE SIGNAL FOR {symbol} from "
                    f"{best_signal['signal_source'].capitalize()} (conf={highest_conf})"
                )
                print(json.dumps(best_signal, indent=2))

                # Phase 3 integration — execution
                print("  Execution trigger → order_manager.execute_trade_signal(...)")
                execute_trade_signal(best_signal, clients)
                trades_today += 1

                # Small cool-down after a trade
                time.sleep(120)
                break  # exit symbol loop, start next full cycle

        print(f"--- Cycle complete. Waiting {ANALYSIS_INTERVAL_SECONDS} seconds... ---")
        time.sleep(ANALYSIS_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_analysis_loop()
