#!/usr/bin/env python3
"""
Identify HTF Key Levels (1h & 4h) with an LLM
Part of the pre-market pipeline.

Output:
  data/daily_news/YYYY-MM-DD/raw/key_levels.json

Requires:
  - .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, GOOGLE_API_KEY
  - prompts/prompts.json containing {"htf_key_levels_prompt": {"content": "... {{price_data}} ..."}}

Notes:
  - Uses gemini-2.5-flash-lite with response_schema to force valid JSON.
  - Falls back to a lightweight JSON "repair" step then a 1-shot "fixer" call if needed.
  - Caps levels per symbol by proximity to latest price.
"""

import os
import sys
import json
import re
import time
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ---- Project path & imports ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.settings import WATCHLIST
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# ---- Config ----
ET_TZ = pytz.timezone("US/Eastern")
LOOKBACK_DAYS = 30
MAX_LEVELS = 6  # total per symbol (we’ll target ~3 below + ~3 above)
REQUEST_TIMEOUT_SEC = 25

# ----------------------------
# Env & Clients
# ----------------------------
def _load_env_and_clients():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("Alpaca API credentials not found in .env.")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Google API key not found in .env.")

    genai.configure(api_key=google_api_key)

    system_instruction = (
        "You are a precise technical analyst. "
        "You MUST return valid JSON only—no prose. Your entire response must be a single JSON object."
    )
    llm_model = genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        system_instruction=system_instruction
    )

    alpaca_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)

    feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX

    return alpaca_client, llm_model, feed

def _get_file_paths():
    today_str = datetime.now(ET_TZ).strftime("%Y-%m-%d")
    out_dir = os.path.join(PROJECT_ROOT, "data", "daily_news", today_str, "raw")
    os.makedirs(out_dir, exist_ok=True)
    return {
        "prompt_template": os.path.join(PROJECT_ROOT, "prompts", "prompts.json"),
        "output": os.path.join(out_dir, "key_levels.json"),
        "prompts_dir": out_dir,
    }

# ----------------------------
# Bars → JSON records
# ----------------------------
def _one_tf(symbol, client, feed, tf, tf_label, start_utc, end_utc):
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

    sdf = df.xs(symbol) if isinstance(df.index, pd.MultiIndex) else df
    sdf = sdf.tz_convert(ET_TZ).reset_index().rename(columns={"timestamp": "timestamp_et"})
    sdf["timestamp_et"] = sdf["timestamp_et"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    sdf["timeframe"] = tf_label
    return sdf[["timestamp_et", "open", "high", "low", "close", "volume", "timeframe"]]

def _fetch_htf_data(symbol, client, feed):
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=LOOKBACK_DAYS)

    h1 = _one_tf(symbol, client, feed, TimeFrame.Hour, "1h", start_utc, end_utc)
    h4 = _one_tf(symbol, client, feed, TimeFrame(4, TimeFrameUnit.Hour), "4h", start_utc, end_utc)

    merged = pd.concat([h1, h4], ignore_index=True).sort_values("timestamp_et")
    return merged.to_dict(orient="records")

def _latest_close(price_data):
    """Return latest close from the most recent 1h bar if available; else latest overall."""
    if not price_data:
        return None
    # Parse timestamps; they’re ET with offset like -0400; treat as aware datetimes
    def parse_ts(ts):
        # 2025-09-09T08:00:00-0400 → add colon in tz if missing for fromisoformat
        if re.match(r".*[+-]\d{4}$", ts):
            ts = ts[:-5] + ts[-5:-2] + ":" + ts[-2:]
        return datetime.fromisoformat(ts)
    # Prefer 1h bars
    h1 = [r for r in price_data if r.get("timeframe") == "1h"]
    target = h1 if h1 else price_data
    latest = max(target, key=lambda r: parse_ts(r["timestamp_et"]))
    return float(latest["close"])

# ----------------------------
# Prompting
# ----------------------------
def _construct_prompt(price_data, prompt_template_path):
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = json.load(f)["htf_key_levels_prompt"]["content"]
    price_data_str = json.dumps(price_data, separators=(",", ":"))
    return template.replace("{{price_data}}", price_data_str)

def _append_focus_instructions(prompt: str, symbol: str, last_price: float, max_items: int = MAX_LEVELS) -> str:
    focus = (
        f"\n\nFOCUS:\n"
        f"- Current price for {symbol}: {last_price:.4f} (approx now, ET→UTC data provided). "
        f"Prefer the nearest levels to this price.\n"
        f"- Return AT MOST {max_items} levels total (ideally ~3 supports below and ~3 resistances above). "
        f"Never exceed {max_items}.\n"
        f"- Use ISO 8601 UTC for 'timestamp' (e.g., 2025-10-08T15:00:00Z). Only '1h' or '4h' for 'timeframe'.\n"
        f"- JSON only. No prose. Required shape: "
        f'{{"levels":[{{"price":number,"type":"support|resistance","timeframe":"1h|4h","timestamp":"ISO-UTC","reasoning":"string"}}]}}'
    )
    return prompt + focus

def _response_schema():
    # No "additionalProperties" — the SDK proto doesn’t support it.
    return {
        "type": "object",
        "properties": {
            "levels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "price": {"type": "number"},
                        "type":  {"type": "string", "enum": ["support", "resistance"]},
                        "timeframe": {"type": "string", "enum": ["1h", "4h"]},
                        "timestamp": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["price", "type", "timeframe", "timestamp", "reasoning"],
                },
            }
        },
        "required": ["levels"],
    }

# ----------------------------
# JSON parsing & repair
# ----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove ```json ... ``` or ``` ...
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_outer_json(s: str) -> str:
    """Return substring from first '{' to matching final '}' (best effort)."""
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return s
    return s[start:end + 1]

def _normalize_quotes_and_commas(s: str) -> str:
    # Replace smart quotes with ASCII
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    return s

def _loads_strict_or_repair(raw: str):
    text = _strip_code_fences(raw)
    text = _extract_outer_json(text)
    text = _normalize_quotes_and_commas(text)
    try:
        return json.loads(text)
    except Exception as e:
        # Log a short preview for debugging
        preview = text[:1000].replace("\n", " ")  # keep logs readable
        print(f"  ! JSON parse failed: {e} | preview: {preview[:300]}...")
        return None

def _repair_via_llm(raw_text: str, model) -> dict | None:
    """One-shot 'fixer' call: ask the model to re-emit valid JSON conforming to the schema."""
    fixer_prompt = (
        "Return VALID JSON only (no code fences, no preamble) that conforms to this schema:\n"
        + json.dumps(_response_schema(), separators=(",", ":"))
        + "\n\nHere is model output to fix. Preserve the original values; only fix formatting/escaping:\n"
        + raw_text[:8000]
    )
    try:
        gen_cfg = {
            "response_mime_type": "application/json",
            "response_schema": _response_schema(),
            "temperature": 0.1,
        }
        resp = model.generate_content(fixer_prompt, generation_config=gen_cfg,
                                      request_options={"timeout": REQUEST_TIMEOUT_SEC})
        repaired = getattr(resp, "text", "") or ""
        return _loads_strict_or_repair(repaired)
    except Exception as e:
        print(f"  ! LLM fixer failed: {e}")
        return None

# ----------------------------
# LLM call
# ----------------------------
def _query_llm(prompt, model, max_items=MAX_LEVELS):
    gen_cfg = {
        "response_mime_type": "application/json",
        "response_schema": _response_schema(),
        "temperature": 0.2,
    }
    try:
        resp = model.generate_content(prompt, generation_config=gen_cfg,
                                      request_options={"timeout": REQUEST_TIMEOUT_SEC})
        raw = getattr(resp, "text", "") or ""
        parsed = _loads_strict_or_repair(raw)
        if parsed is None:
            # Try a tighter re-ask once
            tightened = prompt + "\n\nREMINDER: Output must be VALID JSON exactly matching the schema. No code fences."
            resp2 = model.generate_content(tightened, generation_config=gen_cfg,
                                           request_options={"timeout": REQUEST_TIMEOUT_SEC})
            raw2 = getattr(resp2, "text", "") or ""
            parsed = _loads_strict_or_repair(raw2)
            if parsed is None:
                # Last resort: fixer
                parsed = _repair_via_llm(raw2 or raw, model)
        return parsed
    except Exception as e:
        print(f"  ! LLM query failed: {e}")
        return None

# ----------------------------
# Post-filtering: keep only nearest levels to current price
# ----------------------------
def _filter_near_price(levels, last_price: float, max_items: int = MAX_LEVELS):
    if not isinstance(levels, list) or last_price is None:
        return []

    # Deduplicate by price (rounded) & type/timeframe to avoid clusters
    seen = set()
    cleaned = []
    for lv in levels:
        try:
            key = (round(float(lv["price"]), 2), lv.get("type"), lv.get("timeframe"))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(lv)
        except Exception:
            continue

    # Split supports below and resistances above
    supports = [lv for lv in cleaned if str(lv.get("type")) == "support" and float(lv["price"]) <= last_price]
    resistances = [lv for lv in cleaned if str(lv.get("type")) == "resistance" and float(lv["price"]) >= last_price]

    # Sort each side by distance to last_price
    supports.sort(key=lambda lv: abs(last_price - float(lv["price"])))
    resistances.sort(key=lambda lv: abs(float(lv["price"]) - last_price))

    # Target ~3 each side (but don’t exceed max_items)
    half = max_items // 2
    take_sup = supports[:half + (max_items % 2)]  # if odd, bias a tad below
    take_res = resistances[:half]

    result = take_sup + take_res

    # If one side is missing, fill from nearest overall
    if len(result) < max_items:
        remaining = [lv for lv in cleaned if lv not in result]
        remaining.sort(key=lambda lv: abs(float(lv["price"]) - last_price))
        fill = remaining[: (max_items - len(result))]
        result.extend(fill)

    # Final cap
    return result[:max_items]

# ----------------------------
# Main
# ----------------------------
def main():
    pipeline_start_time = time.monotonic()
    print(f"--- Pipeline started at {datetime.now(ET_TZ).strftime('%Y-%m-%d %H:%M:%S')} ET ---")

    paths = _get_file_paths()

    print("Loading environment variables and clients...")
    t0 = time.monotonic()
    alpaca_client, llm_model, feed = _load_env_and_clients()
    print(f" > Clients loaded in {time.monotonic() - t0:.2f} seconds.")

    all_levels = {}
    for symbol in WATCHLIST:
        sym_t0 = time.monotonic()
        print(f"\n--- Analyzing {symbol} ---")
        print(f"  > Fetching price data for {LOOKBACK_DAYS} days...")
        f0 = time.monotonic()
        price_data = _fetch_htf_data(symbol, alpaca_client, feed)
        print(f"  > Data fetched in {time.monotonic() - f0:.2f} seconds.")

        if not price_data:
            print(f"  ! No data for {symbol}. Skipping.")
            continue

        last_price = _latest_close(price_data)
        if last_price is None:
            print(f"  ! Could not determine last price for {symbol}. Skipping.")
            continue

        print("  > Constructing prompt...")
        prompt = _construct_prompt(price_data, paths["prompt_template"])
        prompt = _append_focus_instructions(prompt, symbol, last_price, MAX_LEVELS)
        print(f"  > Prompt constructed. Length: {len(prompt)} characters.")

        # Persist prompt for debugging
        try:
            prompt_filepath = os.path.join(paths["prompts_dir"], f"key_level_prompt_{symbol}.json")
            with open(prompt_filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "symbol": symbol,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "prompt_length_chars": len(prompt),
                    "prompt_content": prompt
                }, f, indent=2)
            print("  > Prompt for {0} saved to file.".format(symbol))
        except Exception as e:
            print(f"  ! Failed to save prompt for {symbol}: {e}")

        print("  > Querying LLM for key levels...")
        q0 = time.monotonic()
        llm_resp = _query_llm(prompt, llm_model, max_items=MAX_LEVELS)
        print(f"  > LLM response received in {time.monotonic() - q0:.2f} seconds.")

        if isinstance(llm_resp, dict):
            raw_levels = llm_resp.get("levels", []) if "levels" in llm_resp else llm_resp.get(symbol, [])
            filtered = _filter_near_price(raw_levels, last_price, MAX_LEVELS)
            all_levels[symbol] = filtered
            print(f"  > Identified {len(filtered)} levels for {symbol} (filtered to nearest).")
        else:
            print("  ! Invalid LLM response; skipping.")

        print(f"  > Total time for {symbol}: {time.monotonic() - sym_t0:.2f} seconds.")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbols": all_levels,
    }

    print("\nWriting all levels to output file...")
    w0 = time.monotonic()
    with open(paths["output"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f" > File written in {time.monotonic() - w0:.2f} seconds.")

    pipeline_end_time = time.monotonic()
    print(f"\n--- Pipeline finished in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")
    print(f"Successfully saved key levels to {paths['output']}")

if __name__ == "__main__":
    main()
