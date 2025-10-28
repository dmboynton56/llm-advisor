#!/usr/bin/env python3
"""
Live analyzer with time-compressed TEST mode, now with:
- Focus list ingestion (SPY/QQQ/IWM + 1–2 picks from premarket step)
- Per-symbol LLM calls (fan-out) with concurrency
- Strict post-validation of trade outputs
- SPLIT-adjusted IEX by default
- Rolling window: 150 minutes
- Session window: 07:30–11:00 ET (adjustable via env)
- Preopen context: start at 06:30 ET
- Fast loop (signals): every 60s live / 10s test (== 1 simulated minute)
- Slow loop (context): every 250s live / 25s test (== 2.5 simulated minutes)
- TEST mode (--date YYYY-MM-DD): prefetch all bars once (06:30→11:00 ET) and slice by simulated clock

Env:
  GOOGLE_API_KEY
  ALPACA_API_KEY, ALPACA_SECRET_KEY
  ALPACA_DATA_FEED=iex|sip (default iex)
  RUN_START_ET=07:30
  RUN_END_ET=11:00
  PREOPEN_ET=06:30
  FAST_SECS (default 60 live / 10 test)
  SLOW_SECS (default 250 live / 25 test)
  CONF_THRESHOLD=70
  FAST_MODEL=gemini-2.5-flash-lite
  SLOW_MODEL=gemini-2.5-flash-lite
"""

import os, sys, json, time, argparse, uuid, re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import pytz
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# ---- Google Generative AI SDK (Gemini) ----
# pip install google-generativeai
import google.generativeai as genai

# ---- Alpaca Data (alpaca-py) ----
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# Concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# Project root + settings
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import WATCHLIST  # list[str]
ET = pytz.timezone("US/Eastern")

# Deterministic confluence engine modules
try:
    from src.features.feature_packager import pack_features_for_symbol
    from src.features.tracker import update_symbol_state
except Exception:
    # Allow relative import fallback if running from project root in some tools
    from features.feature_packager import pack_features_for_symbol  # type: ignore
    from features.tracker import update_symbol_state  # type: ignore

# ---------------------- Pydantic schemas (structured LLM output) ----------------------
class Plan(BaseModel):
    narrative: Optional[str] = None
    invalidation: Optional[str] = None
    keep_above_take_profits: Optional[str] = None

class OptionLeg(BaseModel):
    strike: Optional[float] = None
    expiry_days: Optional[int] = None
    contract_hint: Optional[str] = None

class TradeParams(BaseModel):
    entry_price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: List[float] = []
    risk_reward: Optional[float] = None

class SymbolCtx(BaseModel):
    symbol: str
    active_confluences: List[str] = []
    ltf_key_levels: List[float] = []
    price_bias: Optional[str] = None         # 'long'|'short'|'choppy'
    favored_position: Optional[str] = None   # 'long'|'short'|'none'
    confidence: int = 0                      # 0-100
    plan: Optional[Plan] = None
    trade_params: Optional[TradeParams] = None
    option_leg: Optional[OptionLeg] = None

class OverallCtx(BaseModel):
    market_direction: Optional[str] = None
    regime: Optional[str] = None
    key_drivers: List[str] = []
    notes: Optional[str] = None

class FastResp(BaseModel):
    generated_at_utc: str
    overall: Optional[OverallCtx] = None
    symbols: List[SymbolCtx]

class SlowResp(BaseModel):
    generated_at_utc: str
    overall: OverallCtx
    symbols: List[SymbolCtx]

# Per-symbol LLM response (used in fast loop fan-out)
class PerSymbolResp(BaseModel):
    generated_at_utc: str
    symbol: str
    active_confluences: List[str] = []
    ltf_key_levels: List[float] = []
    price_bias: Optional[str] = None
    favored_position: Optional[str] = None
    confidence: int = 0
    plan: Optional[Plan] = None
    trade_params: Optional[TradeParams] = None
    option_leg: Optional[OptionLeg] = None

# ---------------------- IO helpers ----------------------
def load_json(path: Path) -> Any:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def append_jsonl(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def day_paths(day_et: str) -> Dict[str, Path]:
    base_processed = PROJECT_ROOT / "data" / "daily_news" / day_et / "processed"
    base_raw = PROJECT_ROOT / "data" / "daily_news" / day_et / "raw"
    base_processed.mkdir(parents=True, exist_ok=True)
    base_raw.mkdir(parents=True, exist_ok=True)
    return {
        "prompts": PROJECT_ROOT / "prompts" / "prompts.json",
        "processed_briefing": base_processed / "processed_briefing.json",
        "key_levels": base_raw / "key_levels.json",
        "strategy": (PROJECT_ROOT / "strategy" / "strategy.json"
                     if (PROJECT_ROOT / "strategy" / "strategy.json").exists()
                     else (PROJECT_ROOT / "src" / "strategy" / "strategy.json")),
        "confluences": (PROJECT_ROOT / "src" / "execution" / "confluences.json"
                        if (PROJECT_ROOT / "src" / "execution" / "confluences.json").exists()
                        else PROJECT_ROOT / "execution" / "confluences.json"),
        "current_context": base_processed / "current_context.json",
        "current_context_log": base_processed / "current_context_log.jsonl",
        "signals": base_processed / "signals.json",
        "day": day_et,
        "events_log": base_processed / "backtest_events.jsonl",
        # NEW: focus + capsules + raw llm dump directory
        "focus_list": base_processed / "focus_list.json",
        "capsules_dir": base_processed / "context_capsules",
        "llm_raw_dir": base_processed / "llm_raw",
    }

def load_focus_symbols(paths: Dict[str, Path]) -> List[str]:
    doc = load_json(paths["focus_list"]) or {}
    focus = doc.get("focus")
    if isinstance(focus, list) and focus:
        return [s.upper() for s in focus]
    # Fallback: always-on trio if focus list not prepared
    return ["SPY", "QQQ", "IWM"]

def load_capsule(paths: Dict[str, Path], symbol: str) -> dict:
    p = paths["capsules_dir"] / f"{symbol}.json"
    return load_json(p) or {"symbol": symbol, "notes": ["no capsule available"]}

def log_minute(paths, mode: str, et_time: datetime, cur_utc: datetime,
               start_utc: datetime, end_utc: datetime, watchlist: list, bars: dict):
    """
    Append a one-line JSON record indicating the current minute context.
    mode: 'test' | 'live'
    """
    record = {
        "event": "minute_tick",
        "mode": mode,
        "et_time": et_time.isoformat(),
        "utc_time": cur_utc.isoformat(),
        "window_start_utc": start_utc.isoformat(),
        "window_end_utc": end_utc.isoformat(),
        "watchlist": watchlist,
        "counts": {
            "symbols": len(watchlist),
            "bars_1m_total": sum(len(bars.get("bars_1m", {}).get(s, [])) for s in watchlist),
            "bars_5m_total": sum(len(bars.get("bars_5m", {}).get(s, [])) for s in watchlist),
        },
    }
    append_jsonl(paths["current_context_log"], record)
    # concise console line:
    print(f"[TICK-{mode.upper()}] {et_time.strftime('%H:%M')} ET | "
          f"{record['counts']['bars_1m_total']}x1m, {record['counts']['bars_5m_total']}x5m")

# ---------------------- LLM schemas (dicts for response_schema) ----------------------
def fast_schema_dict() -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "generated_at_utc": {"type": "STRING"},
            "overall": {
                "type": "OBJECT",
                "properties": {
                    "market_direction": {"type": "STRING"},
                    "regime": {"type": "STRING"},
                    "key_drivers": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "notes": {"type": "STRING"},
                },
            },
            "symbols": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "symbol": {"type": "STRING"},
                        "active_confluences": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "ltf_key_levels": {"type": "ARRAY", "items": {"type": "NUMBER"}},
                        "price_bias": {"type": "STRING"},
                        "favored_position": {"type": "STRING"},
                        "confidence": {"type": "INTEGER"},
                        "plan": {
                            "type": "OBJECT",
                            "properties": {
                                "narrative": {"type": "STRING"},
                                "invalidation": {"type": "STRING"},
                                "keep_above_take_profits": {"type": "STRING"},
                            },
                        },
                        "trade_params": {
                            "type": "OBJECT",
                            "properties": {
                                "entry_price_target": {"type": "NUMBER"},
                                "stop_loss": {"type": "NUMBER"},
                                "take_profits": {"type": "ARRAY", "items": {"type": "NUMBER"}},
                                "risk_reward": {"type": "NUMBER"},
                            },
                        },
                        "option_leg": {
                            "type": "OBJECT",
                            "properties": {
                                "strike": {"type": "NUMBER"},
                                "expiry_days": {"type": "INTEGER"},
                                "contract_hint": {"type": "STRING"},
                            },
                        },
                    },
                },
            },
        },
    }

def slow_schema_dict() -> dict:
    return fast_schema_dict()

def symbol_schema_dict() -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "generated_at_utc": {"type": "STRING"},
            "symbol": {"type": "STRING"},
            "active_confluences": {"type": "ARRAY", "items": {"type": "STRING"}},
            "ltf_key_levels": {"type": "ARRAY", "items": {"type": "NUMBER"}},
            "price_bias": {"type": "STRING"},
            "favored_position": {"type": "STRING"},
            "confidence": {"type": "INTEGER"},
            "plan": {
                "type": "OBJECT",
                "properties": {
                    "narrative": {"type": "STRING"},
                    "invalidation": {"type": "STRING"},
                    "keep_above_take_profits": {"type": "STRING"},
                },
            },
            "trade_params": {
                "type": "OBJECT",
                "properties": {
                    "entry_price_target": {"type": "NUMBER"},
                    "stop_loss": {"type": "NUMBER"},
                    "take_profits": {"type": "ARRAY", "items": {"type": "NUMBER"}},
                    "risk_reward": {"type": "NUMBER"},
                },
            },
            "option_leg": {
                "type": "OBJECT",
                "properties": {
                    "strike": {"type": "NUMBER"},
                    "expiry_days": {"type": "INTEGER"},
                    "contract_hint": {"type": "STRING"},
                },
            },
        },
        "required": ["generated_at_utc", "symbol"],
    }

# ---------------------- Time utilities ----------------------
def parse_hhmm(s: str) -> tuple[int,int]:
    h, m = s.split(":")
    return int(h), int(m)

def et_dt(day: datetime.date, hhmm: str) -> datetime:
    h, m = parse_hhmm(hhmm)
    return ET.localize(datetime(day.year, day.month, day.day, h, m))

def to_utc(dt_et: datetime) -> datetime:
    return dt_et.astimezone(timezone.utc)

def session_bounds(date_et: datetime.date) -> dict:
    run_start = os.getenv("RUN_START_ET", "07:30")
    run_end = os.getenv("RUN_END_ET", "11:00")
    preopen = os.getenv("PREOPEN_ET", "06:30")
    return {
        "run_start_et": et_dt(date_et, run_start),
        "run_end_et": et_dt(date_et, run_end),
        "preopen_et": et_dt(date_et, preopen),
    }

# ---------------------- Alpaca clients & bar fetch ----------------------
class Clients:
    def __init__(self):
        load_dotenv()
        k = os.getenv("ALPACA_API_KEY")
        s = os.getenv("ALPACA_SECRET_KEY")
        if not (k and s):
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY")
        self.data = StockHistoricalDataClient(k, s)
        feed_env = (os.getenv("ALPACA_DATA_FEED") or "iex").lower()
        self.feed = DataFeed.SIP if feed_env == "sip" else DataFeed.IEX  # default IEX

def _coerce_ts_iso(ts_like: Any) -> str:
    """
    Robustly coerce a timestamp (pd.Timestamp, datetime, or (symbol, ts) tuples from multiindex iter)
    to ISO string.
    """
    if isinstance(ts_like, tuple):
        # Sometimes iterrows on multiindex yields (ts, row) already; guard anyway.
        ts_like = ts_like[0] if len(ts_like) >= 1 else None
    if isinstance(ts_like, pd.Timestamp):
        ts = ts_like.to_pydatetime()
    elif isinstance(ts_like, datetime):
        ts = ts_like
    else:
        # best effort
        try:
            ts = pd.to_datetime(ts_like).to_pydatetime()
        except Exception:
            ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()

def _bars_to_records(df: Optional[pd.DataFrame], symbol: str) -> List[dict]:
    if df is None or isinstance(df, list):
        return [] if not df else df  # already a list of records
    if df.empty:
        return []
    out = []
    try:
        sdf = df.xs(symbol, level=0).sort_index()
    except Exception:
        # If df is already filtered to one symbol or index differs
        sdf = df.sort_index()
    # Reset index to access timestamp robustly
    if not isinstance(sdf.index, pd.MultiIndex):
        sdf = sdf.reset_index().rename(columns={"index": "timestamp"})
    else:
        sdf = sdf.reset_index()
    # Normalize column names
    cols = {c.lower(): c for c in sdf.columns}
    ts_col = "timestamp" if "timestamp" in cols else list(sdf.columns)[0]
    for _, r in sdf.iterrows():
        ts_val = r[ts_col]
        out.append({
            "t": _coerce_ts_iso(ts_val),
            "o": float(r[cols.get("open", "open")]),
            "h": float(r[cols.get("high", "high")]),
            "l": float(r[cols.get("low", "low")]),
            "c": float(r[cols.get("close", "close")]),
            "v": float(r[cols.get("volume", "volume")]),
        })
    return out

def fetch_window_bars(clients: Clients, symbols: List[str], start_utc: datetime, end_utc: datetime) -> dict:
    """Live mode: fetch SPLIT-adjusted IEX/SIP 1m & 5m bars for the window."""
    req_1m = StockBarsRequest(symbol_or_symbols=symbols,
                              timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                              start=start_utc, end=end_utc,
                              feed=clients.feed, adjustment=Adjustment.SPLIT)
    req_5m = StockBarsRequest(symbol_or_symbols=symbols,
                              timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                              start=start_utc, end=end_utc,
                              feed=clients.feed, adjustment=Adjustment.SPLIT)
    df_1m = clients.data.get_stock_bars(req_1m).df
    df_5m = clients.data.get_stock_bars(req_5m).df
    out = {"bars_1m": {}, "bars_5m": {}}
    for sym in symbols:
        out["bars_1m"][sym] = _bars_to_records(df_1m, sym)
        out["bars_5m"][sym] = _bars_to_records(df_5m, sym)
    return out

class TestBarCache:
    """Test mode: prefetch full session once (06:30→11:00 ET) and slice by simulated time."""
    def __init__(self, clients: Clients, symbols: List[str], preopen_utc: datetime, end_utc: datetime):
        req_1m = StockBarsRequest(symbol_or_symbols=symbols,
                                  timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                  start=preopen_utc, end=end_utc,
                                  feed=clients.feed, adjustment=Adjustment.SPLIT)
        req_5m = StockBarsRequest(symbol_or_symbols=symbols,
                                  timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                  start=preopen_utc, end=end_utc,
                                  feed=clients.feed, adjustment=Adjustment.SPLIT)
        self.df_1m = clients.data.get_stock_bars(req_1m).df
        self.df_5m = clients.data.get_stock_bars(req_5m).df
        self.symbols = symbols

    def slice(self, start_utc: datetime, end_utc: datetime) -> dict:
        out = {"bars_1m": {}, "bars_5m": {}}
        df1 = self.df_1m
        df5 = self.df_5m
        if df1 is not None and not df1.empty:
            df1 = df1.reset_index()
            df1 = df1[(df1["timestamp"] >= pd.Timestamp(start_utc)) & (df1["timestamp"] <= pd.Timestamp(end_utc))]
            df1 = df1.set_index(["symbol", "timestamp"]).sort_index()
        if df5 is not None and not df5.empty:
            df5 = df5.reset_index()
            df5 = df5[(df5["timestamp"] >= pd.Timestamp(start_utc)) & (df5["timestamp"] <= pd.Timestamp(end_utc))]
            df5 = df5.set_index(["symbol", "timestamp"]).sort_index()
        for sym in self.symbols:
            out["bars_1m"][sym] = _bars_to_records(df1, sym) if df1 is not None else []
            out["bars_5m"][sym] = _bars_to_records(df5, sym) if df5 is not None else []
        return out

# ---------------------- Gemini init ----------------------
def init_gemini():
    # Configure API key for google-generativeai
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

# ---------------------- Prompt builders ----------------------
def build_slow_prompt(paths: Dict[str, Path], watchlist: List[str], bars: dict) -> str:
    prompts = load_json(paths["prompts"]) or {}
    system_prompt = (prompts.get("system_prompt") or {}).get("content", "You are a trading analyst.")
    main_prompt = (prompts.get("main_analysis_prompt") or {}).get("content", "Analyze the market per strategy.")
    processed_briefing = load_json(paths["processed_briefing"]) or {}
    current_context = load_json(paths["current_context"]) or {}
    return (
        f"## SYSTEM_PROMPT\n{system_prompt}\n\n"
        f"## MAIN_ANALYSIS_PROMPT (big-picture refresh)\n{main_prompt}\n\n"
        f"## WATCHLIST\n{json.dumps(watchlist)}\n\n"
        f"## PROCESSED_BRIEFING\n{json.dumps(processed_briefing)[:120000]}\n\n"
        f"## CURRENT_CONTEXT_SNAPSHOT\n{json.dumps(current_context)[:60000]}\n\n"
        f"## INTRADAY_BARS_JSON (overview)\n{json.dumps(bars)[:180000]}\n\n"
        f"### TASK\nSummarize overall market regime/direction and per-symbol bigger-picture context. Return JSON per schema."
    )

def _bars_for_symbol(bars: dict, symbol: str, max_1m: int = 30, max_5m: int = 12) -> dict:
    """Return a small, capped slice of bars for the symbol to keep prompts tiny."""
    def tail(lst, n):
        return lst[-n:] if isinstance(lst, list) and len(lst) > n else (lst or [])
    one = bars.get("bars_1m", {}).get(symbol, [])
    five = bars.get("bars_5m", {}).get(symbol, [])
    return {"bars_1m": {symbol: tail(one, max_1m)}, "bars_5m": {symbol: tail(five, max_5m)}}

def build_symbol_prompt(paths: Dict[str, Path], symbol: str, capsule: dict, bars_for_sym: dict) -> str:
    prompts = load_json(paths["prompts"]) or {}
    system_prompt = (prompts.get("system_prompt") or {}).get("content", "You are a trading analyst.")
    main_prompt = (prompts.get("main_analysis_prompt") or {}).get("content", "Analyze the market per strategy.")
    strategy = load_json(paths["strategy"]) or {}

    # optional: current_context symbol slice
    current_ctx = load_json(paths["current_context"]) or {}
    cur_by_sym = {}
    if "symbols" in current_ctx and isinstance(current_ctx["symbols"], list):
        for s in current_ctx["symbols"]:
            if isinstance(s, dict) and s.get("symbol") == symbol:
                cur_by_sym = s
                break

    state_blob = (cur_by_sym.get("state") or {})

    # Stage/trend summary for the model
    bos_record = next(
        (c for c in state_blob.get("confluences", []) or []
         if c.get("status") == "active" and str(c.get("kind", "")).startswith("BOS")),
        {}
    )
    context_state = {
        "htf_stage": state_blob.get("htf_stage"),
        "ltf_stage": state_blob.get("ltf_stage"),
        "trend": state_blob.get("trend"),
        "last_bos": state_blob.get("last_bos"),
        "last_bos_level": bos_record.get("level"),
        "last_update_utc": state_blob.get("last_update_utc"),
        "price_bias": cur_by_sym.get("price_bias"),
    }

    # Provide compact active confluence records (full + short form)
    active_records: List[Dict[str, Any]] = []
    for rec in state_blob.get("confluences", []) or []:
        if rec.get("status") != "active":
            continue
        active_records.append({
            "kind": rec.get("kind"),
            "tf": rec.get("tf"),
            "direction": rec.get("direction"),
            "low": rec.get("low"),
            "high": rec.get("high"),
            "level": rec.get("level"),
            "time": rec.get("time"),
            "id": rec.get("id"),
        })
    active_records = active_records[:12]
    active_short = cur_by_sym.get("active_confluences", []) or []

    # Compact bar payload (very small window for anchor)
    recent_bars: Dict[str, Any] = {}
    for key, max_n in (("bars_1m", 6), ("bars_5m", 4)):
        raw = (bars_for_sym.get(key, {}) or {}).get(symbol, [])
        trimmed = raw[-max_n:] if isinstance(raw, list) else []
        recent_bars[key] = trimmed

    # Trim strategy to a lightweight brief if present
    strategy_brief: Dict[str, Any] = {}
    if isinstance(strategy, dict):
        for field in ("name", "summary", "confidence_rules", "execution_rules"):
            if field in strategy:
                strategy_brief[field] = strategy[field]
    strategy_text = json.dumps(strategy_brief)[:8000] if strategy_brief else "{}"

    capsule_text = json.dumps(capsule)[:20000]
    context_state_text = json.dumps(context_state)
    active_records_text = json.dumps(active_records)
    active_short_text = json.dumps(active_short)
    recent_bars_text = json.dumps(recent_bars)

    return (
        f"## SYSTEM_PROMPT\n{system_prompt}\n\n"
        f"## MAIN_ANALYSIS_PROMPT\n{main_prompt}\n\n"
        f"## SYMBOL\n{symbol}\n\n"
        f"## CONTEXT_STATE\n{context_state_text}\n\n"
        f"## ACTIVE_CONFLUENCE_RECORDS\n{active_records_text}\n\n"
        f"## ACTIVE_CONFLUENCES_SHORT\n{active_short_text}\n\n"
        f"## RECENT_BARS\n{recent_bars_text}\n\n"
        f"## CONTEXT_CAPSULE\n{capsule_text}\n\n"
        f"## STRATEGY_BRIEF\n{strategy_text}\n\n"
        f"### TASK\n"
        f"Base your reasoning strictly on the provided context. Do not invent new confluences. \n"
        f"Respond with favored_position ('long'|'short'|'none'), confidence (0-100), narrative plan, invalidation, and trade parameters (entry, stop, targets, risk_reward) consistent with the context.\n"
        f"Return STRICT JSON matching the schema exactly (no prose)."
    )

# ---------------------- LLM caller (safe) ----------------------
def _write_raw_llm(paths: Dict[str, Path], kind: str, text: str) -> Path:
    paths["llm_raw_dir"].mkdir(parents=True, exist_ok=True)
    fname = paths["llm_raw_dir"] / f"{kind}-{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}.txt"
    with fname.open("w", encoding="utf-8") as f:
        f.write(text)
    return fname

def _extract_json_loose(text: str) -> Optional[dict]:
    # Try plain parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Loose brace extraction
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        pass
    # Try to strip code fences if any
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def call_gemini_json_safe(model_name: str,
                          prompt: str,
                          schema_dict: dict,
                          parse_model: BaseModel.__class__,
                          paths: Dict[str, Path],
                          kind: str,
                          timeout: Optional[float] = None) -> Any:
    """Call Gemini with JSON schema; write raw dump; robust parse with one retry on JSON errors."""
    model = genai.GenerativeModel(model_name)

    def _invoke(text_prompt: str):
        return model.generate_content(
            text_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema_dict,
            },
        )

    def _invoke_with_timeout(text_prompt: str):
        if timeout is None:
            return _invoke(text_prompt)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_invoke, text_prompt)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                future.cancel()
                raise TimeoutError(f"Gemini call timed out after {timeout}s for {kind}")

    resp = _invoke_with_timeout(prompt)
    raw_text = getattr(resp, "text", "") or ""
    _write_raw_llm(paths, kind, raw_text)

    data = _extract_json_loose(raw_text)
    if data is None:
        # Retry once with a terse correction instruction appended
        retry_prompt = prompt + "\n\nIMPORTANT: Your previous output was not valid JSON. Re-send ONLY valid JSON that matches the schema. No prose."
        resp2 = _invoke_with_timeout(retry_prompt)
        raw_text2 = getattr(resp2, "text", "") or ""
        _write_raw_llm(paths, kind + "-retry", raw_text2)
        data = _extract_json_loose(raw_text2)
        if data is None:
            raise ValueError("LLM returned non-JSON twice")

    # Pydantic validation
    try:
        return parse_model.model_validate(data)
    except ValidationError as ve:
        # If generated_at_utc missing on per-symbol, patch with now and retry validation once
        if parse_model is PerSymbolResp and isinstance(data, dict):
            data.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
            try:
                return parse_model.model_validate(data)
            except Exception:
                raise ve
        raise ve

# ---------------------- Signal normalization & post-validation ----------------------
def to_signal_packets(fr: FastResp, threshold: int) -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []
    for s in fr.symbols:
        if (s.confidence or 0) >= threshold and (s.favored_position in ("long", "short")):
            sigs.append({
                "symbol": s.symbol,
                "trade_signal": s.favored_position,
                "trade_parameters": {
                    "entry_price_target": s.trade_params.entry_price_target if s.trade_params else None,
                    "stop_loss": s.trade_params.stop_loss if s.trade_params else None,
                    "take_profit": (s.trade_params.take_profits[0] if (s.trade_params and s.trade_params.take_profits) else None),
                    "all_take_profits": (s.trade_params.take_profits if s.trade_params else []),
                    "risk_reward": s.trade_params.risk_reward if s.trade_params else None,
                },
                "option_hint": (s.option_leg.model_dump() if s.option_leg else {}),
                "context": {
                    "active_confluences": s.active_confluences,
                    "ltf_key_levels": s.ltf_key_levels,
                    "price_bias": s.price_bias,
                    "plan": (s.plan.model_dump() if s.plan else None),
                    "confidence": s.confidence,
                },
                "overall_context_snapshot": fr.overall.model_dump(),
                "generated_at_utc": fr.generated_at_utc,
            })
    return sigs

def post_validate_symbol_ctx(sym: str, ctx: PerSymbolResp, rr_min: float = 1.0) -> Tuple[SymbolCtx, List[str]]:
    errs: List[str] = []
    position = (ctx.favored_position or "").lower()
    tps = (ctx.trade_params.take_profits if ctx.trade_params and ctx.trade_params.take_profits else [])
    entry = ctx.trade_params.entry_price_target if ctx.trade_params else None
    sl = ctx.trade_params.stop_loss if ctx.trade_params else None
    rr = ctx.trade_params.risk_reward if ctx.trade_params else None

    # If proposing a trade idea
    if position in ("long", "short") and ctx.confidence >= 1:
        if entry is None or sl is None or not tps:
            errs.append("missing entry/SL/TPs for active idea")
        if entry is not None and sl is not None:
            if position == "long" and not (sl < entry):
                errs.append("long idea but SL >= entry")
            if position == "short" and not (sl > entry):
                errs.append("short idea but SL <= entry")
        if entry is not None and tps:
            if position == "long" and not all(tp > entry for tp in tps):
                errs.append("long idea but at least one TP <= entry")
            if position == "short" and not all(tp < entry for tp in tps):
                errs.append("short idea but at least one TP >= entry")
        if rr is not None and rr < rr_min:
            errs.append(f"risk_reward {rr} < {rr_min}")

    sc = SymbolCtx(
        symbol=sym,
        active_confluences=ctx.active_confluences or [],
        ltf_key_levels=ctx.ltf_key_levels or [],
        price_bias=ctx.price_bias,
        favored_position=ctx.favored_position,
        confidence=int(ctx.confidence or 0),
        plan=ctx.plan,
        trade_params=ctx.trade_params,
        option_leg=ctx.option_leg,
    )
    return sc, errs

# ---------------------- Main (supports live + time-compressed test) ----------------------
def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD test session (time-compressed 07:30→11:00 ET)", default=None)
    args = ap.parse_args()

    test_mode = bool(args.date)
    day_str = (args.date or datetime.now(ET).strftime("%Y-%m-%d"))
    day_date = datetime.strptime(day_str, "%Y-%m-%d").date()

    # Cadence (real seconds)
    fast_secs = int(os.getenv("FAST_SECS") or (10 if test_mode else 60))
    slow_secs = int(os.getenv("SLOW_SECS") or (25 if test_mode else 250))

    # Models
    FAST_MODEL = os.getenv("FAST_MODEL", "gemini-2.5-flash-lite")
    SLOW_MODEL = os.getenv("SLOW_MODEL", "gemini-2.5-flash-lite")
    CONF_THRESHOLD = int(os.getenv("CONF_THRESHOLD", "70"))

    # Paths/clients
    paths = day_paths(day_str)
    init_gemini()
    alp = Clients()

    # Session bounds
    bounds = session_bounds(day_date)
    run_start_et = bounds["run_start_et"]  # 07:30
    run_end_et = bounds["run_end_et"]      # 11:00
    preopen_et = bounds["preopen_et"]      # 06:30
    preopen_utc = to_utc(preopen_et)

    # Focus symbols
    focus_symbols = load_focus_symbols(paths)
    # Backstop: if focus list contains names not in WATCHLIST, it's fine; we fetch only focus.

    # Prepare test cache if in test mode (prefetch once 06:30→11:00 ET)
    cache = None
    if test_mode:
        cache = TestBarCache(alp, focus_symbols, preopen_utc, to_utc(run_end_et))

    paths["current_context"].parent.mkdir(parents=True, exist_ok=True)
    paths["llm_raw_dir"].mkdir(parents=True, exist_ok=True)

    # Sim clock (in test) vs real clock (live)
    sim_et = run_start_et  # starts at 07:30 for test
    last_slow_sim_et: Optional[datetime] = None
    last_slow_real_utc: Optional[datetime] = None

    print(f"Watchlist: {focus_symbols}")
    print(f"Feed={alp.feed.name} Adjustment={Adjustment.SPLIT.name} Window=150m Preopen={preopen_et.strftime('%H:%M')} ET")
    print(f"Session {run_start_et.strftime('%H:%M')}->{run_end_et.strftime('%H:%M')} ET | fast={fast_secs}s slow={slow_secs}s")
    print(f"Models: fast={FAST_MODEL} slow={SLOW_MODEL} threshold={CONF_THRESHOLD}")

    slow_call_timeout = float(os.getenv("SLOW_CALL_TIMEOUT", str(slow_secs)))

    # ----------- helper: per-tick fast loop executor (concurrent) -----------
    def run_fast_fanout(bars: dict, tick_label: str, log_extra: dict = None):
        """Run per-symbol LLM calls concurrently; update current_context & signals."""
        # Pre-compute deterministic features and update per-symbol state before LLM calls
        base_ctx = load_json(paths["current_context"]) or {"symbols": []}
        ex_by_symbol = {s.get("symbol"): s for s in base_ctx.get("symbols", []) if isinstance(s, dict)}

        # Optional SMT reference mapping
        ref_pairs = {"QQQ": "SPY", "IWM": "SPY"}

        features_start = perf_counter()
        for s in focus_symbols:
            try:
                sym_bars = _bars_for_symbol(bars, s)
                feats = pack_features_for_symbol(s, sym_bars, ref_pairs=ref_pairs)
                print(f"[ENGINE] {tick_label} {s}: features ready | tf5 keys={list((feats.get('tf_5m') or {}).keys())}")
                # Persist feature capsule for traceability
                save_json(paths["capsules_dir"] / f"{s}.json", feats)
                prev_slice = ex_by_symbol.get(s, {"symbol": s})
                updated_slice = update_symbol_state(prev_slice, feats)
                dbg_state = updated_slice.get("state", {})
                print(
                    f"[ENGINE] {tick_label} {s}: stage htf={dbg_state.get('htf_stage')} "
                    f"ltf={dbg_state.get('ltf_stage')} active={len(updated_slice.get('active_confluences', []))}"
                )
                ex_by_symbol[s] = updated_slice
            except Exception as e:
                print(f"[ENGINE] {s}: feature compute error: {e}")

        feature_elapsed = perf_counter() - features_start
        print(f"[ENGINE] {tick_label}: feature+state pass took {feature_elapsed:.2f}s")

        # Write intermediate state so prompts can read stage/active confluences
        base_ctx["symbols"] = list(ex_by_symbol.values())
        save_json(paths["current_context"], base_ctx)

        fast_call_timeout = float(os.getenv("FAST_CALL_TIMEOUT", "12"))

        # Load capsules once (from the freshly written features)
        symbol_capsules = {s: load_capsule(paths, s) for s in focus_symbols}

        def worker(sym: str) -> Tuple[str, Optional[SymbolCtx], Optional[str]]:
            try:
                worker_start = perf_counter()
                print(f"[LLM] {tick_label} {sym}: building prompt")
                sprompt = build_symbol_prompt(
                    paths=paths,
                    symbol=sym,
                    capsule=symbol_capsules.get(sym, {}),
                    bars_for_sym=_bars_for_symbol(bars, sym),
                )
                prompt_time = perf_counter() - worker_start
                print(f"[LLM] {tick_label} {sym}: prompt ready in {prompt_time:.2f}s; calling Gemini")
                call_start = perf_counter()
                sresp: PerSymbolResp = call_gemini_json_safe(
                    FAST_MODEL,
                    sprompt,
                    symbol_schema_dict(),
                    PerSymbolResp,
                    paths,
                    kind=f"fast-{sym}",
                    timeout=fast_call_timeout,
                )
                call_time = perf_counter() - call_start
                total_time = perf_counter() - worker_start
                print(
                    f"[LLM] {tick_label} {sym}: received response | favored={sresp.favored_position} "
                    f"conf={sresp.confidence} | call={call_time:.2f}s total={total_time:.2f}s"
                )
                ctx_valid, errs = post_validate_symbol_ctx(sym, sresp, rr_min=1.0)
                if errs:
                    # annotate into plan.keep_above_take_profits for visibility
                    if ctx_valid.plan is None:
                        ctx_valid.plan = Plan()
                    msg = "; ".join(errs)[:200]
                    ctx_valid.plan.keep_above_take_profits = (ctx_valid.plan.keep_above_take_profits or "")
                    if ctx_valid.plan.keep_above_take_profits:
                        ctx_valid.plan.keep_above_take_profits += " | "
                    ctx_valid.plan.keep_above_take_profits += f"POST-VALIDATOR: {msg}"
                return sym, ctx_valid, sresp.generated_at_utc
            except Exception as e:
                print(f"[FAST] {sym}: ERROR {e}")
                return sym, None, None

        results: List[SymbolCtx] = []
        gen_times: List[str] = []

        max_workers = min(5, max(1, len(focus_symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(worker, s): s for s in focus_symbols}
            for fut in as_completed(futs):
                _, ctx, gen_t = fut.result()
                if ctx is not None:
                    results.append(ctx)
                if gen_t:
                    gen_times.append(gen_t)

        gen_utc = (gen_times[-1] if gen_times else datetime.now(timezone.utc).isoformat())
        fr = FastResp(generated_at_utc=gen_utc, overall=None, symbols=results)

        # overall backfill
        if fr.overall is None:
            prev = load_json(paths["current_context"]) or {}
            prev_overall = prev.get("overall") or {"market_direction": None, "regime": None, "key_drivers": [], "notes": None}
            fr.overall = OverallCtx.model_validate(prev_overall)

        # Merge LLM outputs with deterministic state slices (preserve state/active_confluences)
        merged_by_sym = {k: v for k, v in ex_by_symbol.items()}
        for s in fr.symbols:
            cur = merged_by_sym.get(s.symbol, {"symbol": s.symbol})
            # Prefer deterministic active_confluences over LLM-provided
            merged = {**s.model_dump(), **cur}
            merged["active_confluences"] = cur.get("active_confluences", s.active_confluences or [])
            merged_by_sym[s.symbol] = merged

        current_context_doc = {
            "generated_at_utc": fr.generated_at_utc,
            "overall": fr.overall.model_dump(),
            "symbols": list(merged_by_sym.values()),
        }
        save_json(paths["current_context"], current_context_doc)
        line = {"type": "fast", **(log_extra or {}), **current_context_doc}
        append_jsonl(paths["current_context_log"], line)

        sigs = to_signal_packets(fr, CONF_THRESHOLD)
        save_json(paths["signals"], {"generated_at_utc": fr.generated_at_utc, "signals": sigs})
        print(f"[FAST] {tick_label} | symbols={len(fr.symbols)} | signals={len(sigs)}")

    # ------------------ main loop ------------------
    while True:
        if test_mode:
            # End when simulated clock passes 11:00 ET
            if sim_et > run_end_et:
                print("[TEST] Session complete.")
                break
            cur_utc = to_utc(sim_et)
            start_utc = max(preopen_utc, cur_utc - timedelta(minutes=150))
            bars = cache.slice(start_utc, cur_utc)  # no API calls during test

            log_minute(paths, mode="test", et_time=sim_et, cur_utc=cur_utc,
                       start_utc=start_utc, end_utc=cur_utc, watchlist=focus_symbols, bars=bars)

            # FAST (per-symbol concurrent)
            try:
                run_fast_fanout(bars, tick_label=sim_et.strftime("%H:%M:%S ET"),
                                log_extra={"sim_time_et": sim_et.isoformat()})
            except Exception as e:
                print(f"[FAST-TEST] ERROR: {e}")

            # SLOW loop every 2.5 simulated minutes
            try:
                if (last_slow_sim_et is None) or ((sim_et - last_slow_sim_et) >= timedelta(minutes=2, seconds=30)):
                    sprompt = build_slow_prompt(paths, focus_symbols, bars)
                    sr: SlowResp = call_gemini_json_safe(
                        SLOW_MODEL, sprompt, slow_schema_dict(), SlowResp, paths, kind="slow",
                        timeout=slow_call_timeout
                    )
                    base = load_json(paths["current_context"]) or {"symbols": []}
                    base["generated_at_utc"] = sr.generated_at_utc
                    base["overall"] = sr.overall.model_dump()
                    ex_by_symbol = {s.get("symbol"): s for s in base.get("symbols", []) if isinstance(s, dict)}
                    for s in sr.symbols:
                        ex_by_symbol[s.symbol] = {**ex_by_symbol.get(s.symbol, {}), **s.model_dump()}
                    base["symbols"] = list(ex_by_symbol.values())
                    save_json(paths["current_context"], base)
                    append_jsonl(paths["current_context_log"], {"type": "slow", "sim_time_et": sim_et.isoformat(), **base})
                    last_slow_sim_et = sim_et
                    print(f"[SLOW-TEST] {sim_et.strftime('%H:%M:%S ET')} | refreshed overall + {len(sr.symbols)}")
            except Exception as e:
                print(f"[SLOW-TEST] ERROR: {e}")

            # Advance simulated time by 1 minute per fast tick, sleep 10s
            sim_et = sim_et + timedelta(minutes=1)
            time.sleep(fast_secs)

        else:
            # LIVE mode
            wall_et = datetime.now(ET)
            if wall_et < run_start_et:
                sleep_s = max(1, int((run_start_et - wall_et).total_seconds()))
                print(f"[LIVE] Before session. Sleeping {sleep_s}s until {run_start_et.strftime('%H:%M')} ET")
                time.sleep(min(sleep_s, fast_secs))
                continue
            if wall_et > run_end_et:
                print("[LIVE] Session complete (after 11:00 ET). Exiting.")
                break

            cur_utc = wall_et.astimezone(timezone.utc)
            start_utc = max(preopen_utc, cur_utc - timedelta(minutes=150))

            try:
                bars = fetch_window_bars(alp, focus_symbols, start_utc, cur_utc)
            except Exception as e:
                print(f"[BARS-LIVE] ERROR: {e}")
                time.sleep(fast_secs)
                continue

            log_minute(paths, mode="live", et_time=wall_et, cur_utc=cur_utc,
                       start_utc=start_utc, end_utc=cur_utc, watchlist=focus_symbols, bars=bars)

            # FAST (per-symbol concurrent)
            try:
                run_fast_fanout(bars, tick_label=wall_et.strftime("%H:%M:%S ET"))
            except Exception as e:
                print(f"[FAST] ERROR: {e}")

            # SLOW loop every slow_secs (real seconds)
            try:
                now_utc = datetime.now(timezone.utc)
                if (last_slow_real_utc is None) or ((now_utc - last_slow_real_utc).total_seconds() >= slow_secs):
                    sprompt = build_slow_prompt(paths, focus_symbols, bars)
                    sr: SlowResp = call_gemini_json_safe(
                        SLOW_MODEL, sprompt, slow_schema_dict(), SlowResp, paths, kind="slow",
                        timeout=slow_call_timeout
                    )
                    base = load_json(paths["current_context"]) or {"symbols": []}
                    base["generated_at_utc"] = sr.generated_at_utc
                    base["overall"] = sr.overall.model_dump()
                    ex_by_symbol = {s.get("symbol"): s for s in base.get("symbols", []) if isinstance(s, dict)}
                    for s in sr.symbols:
                        ex_by_symbol[s.symbol] = {**ex_by_symbol.get(s.symbol, {}), **s.model_dump()}
                    base["symbols"] = list(ex_by_symbol.values())
                    save_json(paths["current_context"], base)
                    append_jsonl(paths["current_context_log"], {"type": "slow", **base})
                    last_slow_real_utc = now_utc
                    print(f"[SLOW] {wall_et.strftime('%H:%M:%S ET')} | refreshed overall + {len(sr.symbols)}")
            except Exception as e:
                print(f"[SLOW] ERROR: {e}")

            time.sleep(fast_secs)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down.")
