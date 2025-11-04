"""Premarket bias gatherer - combines news scraping and daily bias computation.

Wraps existing news_scraper.py and daily_bias_computing.py functionality.
Actually runs these scripts (like main.py does) to generate the data files.
"""
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys
import subprocess
import os
import pytz
from datetime import timedelta


@dataclass
class PremarketBias:
    """Premarket bias data for a single symbol."""
    symbol: str
    daily_bias: str  # "bullish", "bearish", "choppy"
    confidence: int   # 0-100
    model_output: Dict[str, Any]  # Raw ML model predictions
    news_summary: str  # Summarized news context
    premarket_price: float
    premarket_context: str = ""  # LLM-generated context (optional)


@dataclass
class PremarketContext:
    """Complete premarket context for all symbols."""
    date: str  # YYYY-MM-DD
    symbols: Dict[str, PremarketBias]
    market_context: Dict[str, Any]  # SPY bias, VIX, macro news


def gather_premarket_bias(
    trading_date: Optional[date] = None,
    symbols: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    enable_llm_validation: bool = True
) -> PremarketContext:
    """
    Gather premarket bias and news context for symbols.
    
    This function wraps the existing news_scraper and daily_bias_computing modules.
    
    Args:
        trading_date: Date to gather data for (defaults to today)
        symbols: List of symbols to process (defaults to WATCHLIST)
        output_dir: Directory to save intermediate results
        
    Returns:
        PremarketContext with bias and news data
    """
    if trading_date is None:
        et = pytz.timezone("US/Eastern")
        trading_date = datetime.now(et).date()
    
    date_str = trading_date.strftime("%Y-%m-%d")
    
    # Determine output directory
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "daily_news" / date_str / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    project_root = Path(__file__).resolve().parents[2]
    et = pytz.timezone("US/Eastern")
    today = datetime.now(et).date()
    
    # Step 1: Run news scraper (if needed)
    news_output_path = raw_dir / "news.json"
    
    # Set environment variables for date if not today
    env = os.environ.copy()
    if trading_date != today:
        # Set date environment variables that scripts might use
        # Note: These scripts use current date by default, so we might need to 
        # set NEWS_WINDOW env vars or modify scripts to accept date params
        # For now, try running with date context
        prev_day = trading_date - timedelta(days=1)
        env["NEWS_WINDOW_START_ET"] = f"{prev_day}T16:00"
        env["NEWS_WINDOW_END_ET"] = f"{trading_date}T09:30"
    
    # Run news scraper (always run, scripts determine their own date)
    print(f"  Running news_scraper.py...")
    news_script = project_root / "src" / "data_processing" / "news_scraper.py"
    try:
        result = subprocess.run(
            [sys.executable, str(news_script)],
            cwd=str(project_root),
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"  ! Warning: News scraper failed: {e.stderr}")
        print(f"  STDOUT: {e.stdout}")
    
    # Read news data (scripts write to their own date folder)
    # If requesting different date, try to read from requested date folder first
    news_output_path = raw_dir / "news.json"
    
    # Also check if script created file in "today's" folder but we want different date
    if not news_output_path.exists() and trading_date != today:
        today_raw_dir = project_root / "data" / "daily_news" / today.strftime("%Y-%m-%d") / "raw"
        today_news_path = today_raw_dir / "news.json"
        if today_news_path.exists():
            # Copy or read from today's folder
            import shutil
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(today_news_path, news_output_path)
            print(f"  Copied news from {today.strftime('%Y-%m-%d')} to {date_str}")
    
    news_data = {}
    if news_output_path.exists():
        with open(news_output_path, 'r') as f:
            news_data = json.load(f)
    
    # Step 2: Run daily bias computation
    print(f"  Running daily_bias_computing.py...")
    bias_script = project_root / "src" / "data_processing" / "daily_bias_computing.py"
    try:
        result = subprocess.run(
            [sys.executable, str(bias_script)],
            cwd=str(project_root),
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"  ! Warning: Daily bias computation failed: {e.stderr}")
        print(f"  STDOUT: {e.stdout}")
    
    # Read bias data
    bias_output_path = raw_dir / "daily_bias.json"
    
    # Also check if script created file in "today's" folder but we want different date
    if not bias_output_path.exists() and trading_date != today:
        today_raw_dir = project_root / "data" / "daily_news" / today.strftime("%Y-%m-%d") / "raw"
        today_bias_path = today_raw_dir / "daily_bias.json"
        if today_bias_path.exists():
            # Copy from today's folder
            import shutil
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(today_bias_path, bias_output_path)
            print(f"  Copied bias data from {today.strftime('%Y-%m-%d')} to {date_str}")
    
    bias_data = {}
    if bias_output_path.exists():
        with open(bias_output_path, 'r') as f:
            bias_data = json.load(f)
    
    # Check if date in bias_data matches requested date
    bias_date = bias_data.get("et_date", bias_data.get("date_et", ""))
    if bias_date and bias_date != date_str:
        print(f"  ! Note: Bias data is for {bias_date}, requested {date_str} (scripts use current date)")
        # For historical dates, we'd need to modify daily_bias_computing.py
        # For now, proceed with what we have
    
    # Step 3: Combine into PremarketContext structure
    symbol_biases: Dict[str, PremarketBias] = {}
    
    # Extract bias data per symbol
    # daily_bias.json has structure: {"symbols": {"SPY": {...}, "QQQ": {...}}}
    if "symbols" in bias_data:
        symbols_dict = bias_data["symbols"]
        
        # Check if it's a dict (new format) or list (old format)
        if isinstance(symbols_dict, dict):
            # New format: {"SPY": {...}, "QQQ": {...}}
            for symbol, symbol_info in symbols_dict.items():
                if not symbol or not isinstance(symbol_info, dict):
                    continue
                
                # Extract bias (normalize to lowercase)
                bias_str = symbol_info.get("bias", "choppy")
                if isinstance(bias_str, str):
                    bias_str = bias_str.lower()
                
                # Extract confidence (might be float 0-1 or int 0-100)
                confidence_val = symbol_info.get("confidence", 50)
                if isinstance(confidence_val, float) and confidence_val <= 1.0:
                    confidence_val = int(confidence_val * 100)
                confidence = int(confidence_val)
                
                symbol_biases[symbol] = PremarketBias(
                    symbol=symbol,
                    daily_bias=bias_str,
                    confidence=confidence,
                    model_output=symbol_info.get("model_output", symbol_info),  # Use full dict if no model_output key
                    news_summary=_extract_news_summary(symbol, news_data),
                    premarket_price=float(symbol_info.get("premarket_price", symbol_info.get("asof_open_meta", {}).get("open_0930", 0.0))),
                    premarket_context=symbol_info.get("premarket_context", ""),
                )
        elif isinstance(symbols_dict, list):
            # Old format: [{"symbol": "SPY", ...}, ...]
            for symbol_info in symbols_dict:
                if not isinstance(symbol_info, dict):
                    continue
                symbol = symbol_info.get("symbol", "")
                if not symbol:
                    continue
                
                bias_str = symbol_info.get("bias", "choppy").lower()
                confidence_val = symbol_info.get("confidence", 50)
                if isinstance(confidence_val, float) and confidence_val <= 1.0:
                    confidence_val = int(confidence_val * 100)
                confidence = int(confidence_val)
                
                symbol_biases[symbol] = PremarketBias(
                    symbol=symbol,
                    daily_bias=bias_str,
                    confidence=confidence,
                    model_output=symbol_info.get("model_output", symbol_info),
                    news_summary=_extract_news_summary(symbol, news_data),
                    premarket_price=float(symbol_info.get("premarket_price", 0.0)),
                    premarket_context=symbol_info.get("premarket_context", ""),
                )
        
        # Filter to only requested symbols if provided
        if symbols:
            symbol_biases = {sym: bias for sym, bias in symbol_biases.items() if sym in symbols}
    
    # Extract market context
    market_context = {
        "spy_bias": bias_data.get("spy_bias", {}),
        "vix": bias_data.get("vix", {}),
        "macro_news": news_data.get("macro", []),
    }
    
    context = PremarketContext(
        date=date_str,
        symbols=symbol_biases,
        market_context=market_context,
    )
    
    # Step 4: LLM validation (optional)
    if enable_llm_validation:
        try:
            print(f"  Running LLM bias validation...")
            from src.premarket.bias_validator import validate_biases_with_llm, enhance_premarket_context_with_llm_validation
            from src.core.config import Settings
            
            settings = Settings.load()
            llm_validations = validate_biases_with_llm(context, settings=settings)
            
            # Log validation results
            for symbol, validation in llm_validations.items():
                if validation.agreement == "disagree":
                    print(f"    {symbol}: LLM DISAGREES - ML says {validation.ml_bias} ({validation.ml_confidence}%), "
                          f"LLM says {validation.llm_bias} ({validation.llm_confidence}%)")
                elif validation.agreement == "partial":
                    print(f"    {symbol}: LLM PARTIAL - ML: {validation.ml_bias} ({validation.ml_confidence}%), "
                          f"LLM: {validation.llm_bias} ({validation.llm_confidence}%)")
                else:
                    print(f"    {symbol}: LLM AGREES - {validation.llm_bias} ({validation.llm_confidence}%)")
            
            # Enhance context with LLM validations
            context = enhance_premarket_context_with_llm_validation(context, llm_validations)
            print(f"  LLM validation complete")
        except Exception as e:
            print(f"  ! Warning: LLM validation failed: {e}")
            print(f"  Continuing with ML-only predictions...")
    
    return context


def _extract_news_summary(symbol: str, news_data: Dict[str, Any]) -> str:
    """Extract news summary for a symbol from news data."""
    if "symbols" not in news_data:
        return ""
    
    symbol_articles = news_data["symbols"].get(symbol, [])
    if not symbol_articles:
        return ""
    
    # Summarize top articles
    headlines = [article.get("headline", "") for article in symbol_articles[:5]]
    return " | ".join(headlines)


def save_premarket_context(context: PremarketContext, output_path: Path) -> None:
    """Save premarket context to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(asdict(context), f, indent=2, default=str)


def load_premarket_context(date_str: str, base_dir: Optional[Path] = None) -> PremarketContext:
    """Load premarket context from JSON file."""
    if base_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        base_dir = project_root / "data" / "daily_news" / date_str / "processed"
    
    context_path = base_dir / "premarket_context.json"
    
    if not context_path.exists():
        raise FileNotFoundError(f"Premarket context not found at {context_path}")
    
    with open(context_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct from dict
    symbols = {
        sym: PremarketBias(**bias_data)
        for sym, bias_data in data.get("symbols", {}).items()
    }
    
    return PremarketContext(
        date=data["date"],
        symbols=symbols,
        market_context=data.get("market_context", {}),
    )

