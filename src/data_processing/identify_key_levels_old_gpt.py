#!/usr/bin/env python3
"""
Identify HTF Key Levels (1h & 4h) with an LLM
Part of the pre-market pipeline.

Output:
  data/daily_news/YYYY-MM-DD/raw/key_levels.json

Prompts:
  prompts/prompts.json must include:
    { "htf_key_levels_prompt": { "content": "... {{price_data}} ..." } }
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path for local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.settings import WATCHLIST
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# --- Configuration ---
ET_TZ = pytz.timezone("US/Eastern")
LOOKBACK_DAYS = 30  # HTF lookback


# ----------------------------
# Env & Clients
# ----------------------------
def _load_env_and_clients():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("Alpaca API credentials not found in .env.")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OpenAI API key not found in .env.")

    alpaca_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
    llm_client = OpenAI(api_key=openai_key)

    feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX

    return alpaca_client, llm_client, feed


def _get_file_paths():
    today_str = datetime.now(ET_TZ).strftime("%Y-%m-%d")
    out_dir = os.path.join(PROJECT_ROOT, "data", "daily_news", today_str, "raw")
    os.makedirs(out_dir, exist_ok=True)
    return {
        "prompt_template": os.path.join(PROJECT_ROOT, "prompts", "prompts.json"),
        "output": os.path.join(out_dir, "key_levels.json"),
    }


# ----------------------------
# Bars → JSON records
# ----------------------------
def _one_tf(symbol, client, feed, tf, tf_label, start_utc, end_utc):
    """
    Fetch one timeframe and return a tidy DataFrame with ET timestamps.
    Columns: timestamp_et, open, high, low, close, volume, timeframe
    """
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=tf,
        start=start_utc,
        end=end_utc,
        feed=feed,
        adjustment=Adjustment.RAW,
    )
    df = client.get_stock_bars(req).df
    if df is None or df.empty:
        return pd.DataFrame()

    # If MultiIndex (symbol, timestamp), slice by symbol; else pass through
    sdf = df.xs(symbol) if isinstance(df.index, pd.MultiIndex) else df

    # Convert index to ET, then flatten
    sdf = sdf.tz_convert(ET_TZ).reset_index().rename(columns={"timestamp": "timestamp_et"})
    sdf["timestamp_et"] = sdf["timestamp_et"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    sdf["timeframe"] = tf_label
    return sdf[["timestamp_et", "open", "high", "low", "close", "volume", "timeframe"]]


def _fetch_htf_data(symbol, client, feed):
    """
    Fetch 1h + 4h bars over LOOKBACK_DAYS and return a single list[dict].
    """
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=LOOKBACK_DAYS)

    h1 = _one_tf(symbol, client, feed, TimeFrame.Hour, "1h", start_utc, end_utc)
    h4 = _one_tf(symbol, client, feed, TimeFrame(4, TimeFrameUnit.Hour), "4h", start_utc, end_utc)

    merged = pd.concat([h1, h4], ignore_index=True).sort_values("timestamp_et")
    return merged.to_dict(orient="records")


# ----------------------------
# Prompting
# ----------------------------
def _construct_prompt(price_data, prompt_template_path):
    """
    Build the HTF-levels prompt.
    prompts/prompts.json must include:
      { "htf_key_levels_prompt": { "content": "... {{price_data}} ..." } }
    """
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = json.load(f)["htf_key_levels_prompt"]["content"]

    price_data_str = json.dumps(price_data, separators=(",", ":"))
    return template.replace("{{price_data}}", price_data_str)


def _query_llm(prompt, client: OpenAI, model="gpt-5-nano"):
    """Ask the LLM to return strict JSON with levels."""
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise technical analyst. "
                        "You MUST return valid JSON only—no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ! LLM query failed: {e}")
        return None


# ----------------------------
# Main
# ----------------------------
def main():
    paths = _get_file_paths()
    alpaca_client, llm_client, feed = _load_env_and_clients()

    all_levels = {}
    for symbol in WATCHLIST:
        print(f"\n--- Analyzing {symbol} ---")
        price_data = _fetch_htf_data(symbol, alpaca_client, feed)

        if not price_data:
            print(f"  ! No data for {symbol}. Skipping.")
            continue

        prompt = _construct_prompt(price_data, paths["prompt_template"])
        print("  > Querying LLM for key levels...")
        llm_resp = _query_llm(prompt, llm_client)

        if isinstance(llm_resp, dict):
            # Expect a schema like: { "symbol":"SPY", "levels":[ {...}, ... ] }
            # But tolerate a minimal { "levels": [...] } too.
            levels = (
                llm_resp.get("levels", [])
                if "levels" in llm_resp
                else llm_resp.get(symbol, [])
            )
            all_levels[symbol] = levels
            print(f"  > Identified {len(levels)} levels for {symbol}.")
        else:
            print("  ! Invalid LLM response; skipping.")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbols": all_levels,
    }

    with open(paths["output"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSuccessfully saved key levels {paths['output']}")


if __name__ == "__main__":
    main()
