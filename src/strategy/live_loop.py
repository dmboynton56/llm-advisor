#!/usr/bin/env python3
"""
Live analyzer with time-compressed TEST mode.

Core behavior
- SPLIT-adjusted bars
- Default feed: IEX (override with ALPACA_DATA_FEED=sip)
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

import os, sys, json, time, argparse, re, uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type

import pytz
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

# ---- Google Generative AI SDK (Gemini) ----
#   pip install google-generativeai
import google.generativeai as genai

# ---- Alpaca Data (alpaca-py) ----
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# Project root + settings
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import WATCHLIST  # list[str]
ET = pytz.timezone("US/Eastern")

# ---------------------- Pydantic schemas (structured LLM output) ----------------------
class Plan(BaseModel):
    narrative: str | None = None
    invalidation: str | None = None
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
    overall: Optional[OverallCtx] = None   # tolerant; we backfill if missing
    symbols: List[SymbolCtx]

class SlowResp(BaseModel):
    generated_at_utc: str
    overall: OverallCtx
    symbols: List[SymbolCtx]

# ---------------------- IO helpers ----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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
    print(f"[TICK-{mode.upper()}] {et_time.strftime('%H:%M')} ET | "
          f"{record['counts']['bars_1m_total']}x1m, {record['counts']['bars_5m_total']}x5m")

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
        "llm_raw_dir": base_processed / "llm_raw",
    }

def fast_schema_dict() -> dict:
    # No "default" fields anywhere; only permitted schema keys.
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

def _bars_to_records(df: pd.DataFrame, symbol: str) -> list[dict]:
    """
    Robustly converts Alpaca bars DF to a list of dicts for a single symbol.
    Works whether df is MultiIndex(symbol,timestamp) or already flat.
    Returns [] if the symbol isn't present in the window.
    """
    if df is None or df.empty:
        return []

    # Make a flat frame with 'symbol' and 'timestamp' columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()  # columns: ['symbol','timestamp', ...]
    else:
        df = df.reset_index()  # timestamp may be named 'timestamp' already; ok either way

    # If there's no 'symbol' column (edge cases), assume df is already filtered
    if "symbol" in df.columns:
        sdf = df[df["symbol"] == symbol].copy()
    else:
        sdf = df.copy()

    if sdf.empty:
        return []

    # Ensure we have a timestamp column
    if "timestamp" not in sdf.columns:
        # Sometimes the DatetimeIndex name is not preserved; try to find a datetime-like column
        time_cols = [c for c in sdf.columns if "time" in c.lower() or "stamp" in c.lower()]
        if time_cols:
            sdf["timestamp"] = sdf[time_cols[0]]
        else:
            # Last resort: the first column after reset_index could be datetime
            sdf["timestamp"] = sdf.iloc[:, 0]

    sdf = sdf.sort_values("timestamp")

    out: list[dict] = []
    for _, r in sdf.iterrows():
        ts = pd.Timestamp(r["timestamp"]).to_pydatetime()
        out.append({
            "t": ts.isoformat(),
            "o": float(r["open"]),
            "h": float(r["high"]),
            "l": float(r["low"]),
            "c": float(r["close"]),
            "v": float(r["volume"]),
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
        if self.df_1m is not None and not self.df_1m.empty:
            df1 = self.df_1m.reset_index()
            df1 = df1[(df1["timestamp"] >= pd.Timestamp(start_utc)) & (df1["timestamp"] <= pd.Timestamp(end_utc))]
            df1 = df1.set_index(["symbol", "timestamp"]).sort_index()
        else:
            df1 = self.df_1m
        if self.df_5m is not None and not self.df_5m.empty:
            df5 = self.df_5m.reset_index()
            df5 = df5[(df5["timestamp"] >= pd.Timestamp(start_utc)) & (df5["timestamp"] <= pd.Timestamp(end_utc))]
            df5 = df5.set_index(["symbol", "timestamp"]).sort_index()
        else:
            df5 = self.df_5m
        for sym in self.symbols:
            out["bars_1m"][sym] = _bars_to_records(df1, sym) if df1 is not None else []
            out["bars_5m"][sym] = _bars_to_records(df5, sym) if df5 is not None else []
        return out

# ---------------------- Gemini client & prompts ----------------------
def init_gemini():
    # Configure API key for google-generativeai
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

def build_fast_prompt(paths: Dict[str, Path], watchlist: List[str], bars: dict) -> str:
    prompts = load_json(paths["prompts"]) or {}
    system_prompt = (prompts.get("system_prompt") or {}).get("content", "You are a trading analyst.")
    main_prompt = (prompts.get("main_analysis_prompt") or {}).get("content", "Analyze the market per strategy.")

    processed_briefing = load_json(paths["processed_briefing"]) or {}
    key_levels = load_json(paths["key_levels"]) or {}
    strategy = load_json(paths["strategy"]) or {}
    confluences = load_json(paths["confluences"]) or {}
    current_context = load_json(paths["current_context"]) or {}

    return (
        f"## SYSTEM_PROMPT\n{system_prompt}\n\n"
        f"## MAIN_ANALYSIS_PROMPT\n{main_prompt}\n\n"
        f"## WATCHLIST\n{json.dumps(watchlist)}\n\n"
        f"## PROCESSED_BRIEFING\n{json.dumps(processed_briefing)[:120000]}\n\n"
        f"## KEY_LEVELS\n{json.dumps(key_levels)[:80000]}\n\n"
        f"## STRATEGY\n{json.dumps(strategy)[:60000]}\n\n"
        f"## CONFLUENCES_LIBRARY\n{json.dumps(confluences)[:60000]}\n\n"
        f"## CURRENT_CONTEXT_SNAPSHOT\n{json.dumps(current_context)[:60000]}\n\n"
        f"## INTRADAY_BARS_JSON\n{json.dumps(bars)[:180000]}\n\n"
        f"### TASK\n"
        f"For EACH symbol in WATCHLIST, identify NEW/RECENT confluences, favored_position ('long'|'short'|'none'), "
        f"confidence (1-100), LTF key levels, and if confidence >= threshold propose option leg hints and trade params "
        f"(entry_price_target, stop_loss, take_profits[], risk_reward).\n\n"
        f"### OUTPUT REQUIREMENTS\n"
        f"Return ONE JSON object that EXACTLY matches the schema. No commentary before/after. "
        f"If uncertain, use nulls/empty arrays. Keep strings concise (<=200 chars)."
    )

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
        f"### TASK\nSummarize overall market regime/direction and per-symbol bigger-picture context.\n\n"
        f"### OUTPUT REQUIREMENTS\n"
        f"Return ONE JSON object that EXACTLY matches the schema. No commentary before/after."
    )

# ---------- robust LLM JSON handling (raw logs + basic repair) ----------
def _log_raw(paths, kind: str, text: str):
    ensure_dir(paths["llm_raw_dir"])
    fname = paths["llm_raw_dir"] / f"{kind}-{datetime.utcnow().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}.txt"
    fname.write_text(text or "", encoding="utf-8")

def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def _basic_json_repair(text: str) -> str | None:
    obj = _extract_json_object(text)
    if not obj:
        return None
    # remove trailing commas before } or ]
    obj = re.sub(r",\s*([}\]])", r"\1", obj)
    return obj

def call_gemini_json_safe(model_name: str, prompt: str, schema_dict: dict, pyd_model: Type[BaseModel], paths, kind: str):
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema_dict,
            "max_output_tokens": 2048,
        },
    )
    raw = getattr(resp, "text", "") or ""
    # first attempt
    try:
        data = json.loads(raw)
        return pyd_model.model_validate(data)
    except Exception:
        _log_raw(paths, kind+"-raw", raw)
        repaired = _basic_json_repair(raw)
        if repaired:
            try:
                data = json.loads(repaired)
                return pyd_model.model_validate(data)
            except Exception:
                _log_raw(paths, kind+"-repaired", repaired)
        # last resort minimal skeleton to keep the loop alive
        if pyd_model is FastResp:
            return pyd_model.model_validate({
                "generated_at_utc": datetime.utcnow().isoformat()+"Z",
                "overall": {"market_direction": None, "regime": None, "key_drivers": [], "notes": None},
                "symbols": []
            })
        else:  # SlowResp fallback requires overall
            return pyd_model.model_validate({
                "generated_at_utc": datetime.utcnow().isoformat()+"Z",
                "overall": {"market_direction": None, "regime": None, "key_drivers": [], "notes": None},
                "symbols": []
            })

# ---------------------- Signal normalization ----------------------
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
                "overall_context_snapshot": (fr.overall.model_dump() if fr.overall else {}),
                "generated_at_utc": fr.generated_at_utc,
            })
    return sigs

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
    init_gemini()  # configure google-generativeai with GOOGLE_API_KEY
    alp = Clients()

    # Session bounds
    bounds = session_bounds(day_date)
    run_start_et = bounds["run_start_et"]  # 07:30
    run_end_et = bounds["run_end_et"]      # 11:00
    preopen_et = bounds["preopen_et"]      # 06:30
    preopen_utc = to_utc(preopen_et)

    # Prepare test cache if in test mode (prefetch once 06:30→11:00 ET)
    cache = None
    if test_mode:
        cache = TestBarCache(alp, WATCHLIST, preopen_utc, to_utc(run_end_et))

    paths["current_context"].parent.mkdir(parents=True, exist_ok=True)

    # Sim clock (in test) vs real clock (live)
    sim_et = run_start_et  # starts at 07:30 for test
    last_slow_sim_et: Optional[datetime] = None
    last_slow_real_utc: Optional[datetime] = None

    print(f"Watchlist: {WATCHLIST}")
    print(f"Feed={alp.feed.name} Adjustment={Adjustment.SPLIT.name} Window=150m Preopen={preopen_et.strftime('%H:%M')} ET")
    print(f"Session {run_start_et.strftime('%H:%M')}→{run_end_et.strftime('%H:%M')} ET | fast={fast_secs}s slow={slow_secs}s")
    print(f"Models: fast={FAST_MODEL} slow={SLOW_MODEL} threshold={CONF_THRESHOLD}")

    while True:
        if test_mode:
            if sim_et > run_end_et:
                print("[TEST] Session complete.")
                break
            cur_utc = to_utc(sim_et)

            # Rolling window: max(preopen, sim_et - 150m) → sim_et
            start_utc = max(preopen_utc, cur_utc - timedelta(minutes=150))
            bars = cache.slice(start_utc, cur_utc)  # zero API calls during test

            log_minute(paths, mode="test", et_time=sim_et, cur_utc=cur_utc,
                       start_utc=start_utc, end_utc=cur_utc, watchlist=WATCHLIST, bars=bars)

            # FAST loop
            try:
                prompt = build_fast_prompt(paths, WATCHLIST, bars)
                fr: FastResp = call_gemini_json_safe(FAST_MODEL, prompt, fast_schema_dict(), FastResp, paths, "fast")
                if fr.overall is None:
                    prev = load_json(paths["current_context"]) or {}
                    prev_overall = prev.get("overall") or {"market_direction": None, "regime": None, "key_drivers": [], "notes": None}
                    fr.overall = OverallCtx.model_validate(prev_overall)
                current_context_doc = {
                    "generated_at_utc": fr.generated_at_utc,
                    "overall": fr.overall.model_dump(),
                    "symbols": [s.model_dump() for s in fr.symbols],
                }
                save_json(paths["current_context"], current_context_doc)
                append_jsonl(paths["current_context_log"], {"type": "fast", "sim_time_et": sim_et.isoformat(), **current_context_doc})

                sigs = to_signal_packets(fr, CONF_THRESHOLD)
                save_json(paths["signals"], {"generated_at_utc": fr.generated_at_utc, "signals": sigs})
                for ev in sigs:
                    event_line = {
                        "event": "signal_over_threshold",
                        "sim_time_et": sim_et.isoformat(),
                        "generated_at_utc": fr.generated_at_utc,
                        "day": paths["day"],
                        "symbol": ev["symbol"],
                        "direction": ev["trade_signal"],
                        "confidence": ev["context"]["confidence"],
                        "entry": ev["trade_parameters"]["entry_price_target"],
                        "stop_loss": ev["trade_parameters"]["stop_loss"],
                        "take_profits": ev["trade_parameters"]["all_take_profits"],
                        "risk_reward": ev["trade_parameters"]["risk_reward"],
                        "active_confluences": ev["context"]["active_confluences"],
                        "ltf_key_levels": ev["context"]["ltf_key_levels"],
                        "price_bias": ev["context"]["price_bias"],
                        "overall_market": ev["overall_context_snapshot"],
                        "option_hint": ev.get("option_hint", {}),
                        "eval_defaults": {
                            "timeframe": "1m",
                            "horizon_min": 60,
                            "fill_policy": "next_open",
                            "tiebreak": "sl_first"
                        }
                    }
                    append_jsonl(paths["events_log"], event_line)
                print(f"[FAST-TEST] {sim_et.strftime('%H:%M:%S ET')} | symbols={len(fr.symbols)} | signals={len(sigs)}")
            except Exception as e:
                print(f"[FAST-TEST] ERROR: {e}")

            # SLOW loop every 2.5 simulated minutes
            try:
                if (last_slow_sim_et is None) or ((sim_et - last_slow_sim_et) >= timedelta(minutes=2, seconds=30)):
                    sprompt = build_slow_prompt(paths, WATCHLIST, bars)
                    sr: SlowResp = call_gemini_json_safe(SLOW_MODEL, sprompt, slow_schema_dict(), SlowResp, paths, "slow")
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

            # fetch once per tick
            try:
                bars = fetch_window_bars(alp, WATCHLIST, start_utc, cur_utc)
            except Exception as e:
                print(f"[BARS-LIVE] ERROR: {e}")
                time.sleep(fast_secs)
                continue

            log_minute(paths, mode="live", et_time=wall_et, cur_utc=cur_utc,
                       start_utc=start_utc, end_utc=cur_utc, watchlist=WATCHLIST, bars=bars)

            # FAST loop
            try:
                prompt = build_fast_prompt(paths, WATCHLIST, bars)
                fr: FastResp = call_gemini_json_safe(FAST_MODEL, prompt, fast_schema_dict(), FastResp, paths, "fast")
                if fr.overall is None:
                    prev = load_json(paths["current_context"]) or {}
                    prev_overall = prev.get("overall") or {"market_direction": None, "regime": None, "key_drivers": [], "notes": None}
                    fr.overall = OverallCtx.model_validate(prev_overall)
                current_context_doc = {
                    "generated_at_utc": fr.generated_at_utc,
                    "overall": fr.overall.model_dump(),
                    "symbols": [s.model_dump() for s in fr.symbols],
                }
                save_json(paths["current_context"], current_context_doc)
                append_jsonl(paths["current_context_log"], {"type": "fast", **current_context_doc})

                sigs = to_signal_packets(fr, CONF_THRESHOLD)
                save_json(paths["signals"], {"generated_at_utc": fr.generated_at_utc, "signals": sigs})
                print(f"[FAST] {wall_et.strftime('%H:%M:%S ET')} | symbols={len(fr.symbols)} | signals={len(sigs)}")
            except Exception as e:
                print(f"[FAST] ERROR: {e}")

            # SLOW loop every slow_secs (real seconds)
            try:
                now_utc = datetime.now(timezone.utc)
                if (last_slow_real_utc is None) or ((now_utc - last_slow_real_utc).total_seconds() >= slow_secs):
                    sprompt = build_slow_prompt(paths, WATCHLIST, bars)
                    sr: SlowResp = call_gemini_json_safe(SLOW_MODEL, sprompt, slow_schema_dict(), SlowResp, paths, "slow")
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
