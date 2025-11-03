"""Premarket bias gatherer - combines news scraping and daily bias computation.

Wraps existing news_scraper.py and daily_bias_computing.py functionality.
"""
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

# Note: These modules will be called via subprocess or direct import when needed
# For now, we assume their output files exist


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
    output_dir: Optional[Path] = None
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
        from datetime import datetime
        import pytz
        trading_date = datetime.now(pytz.timezone("US/Eastern")).date()
    
    date_str = trading_date.strftime("%Y-%m-%d")
    
    # Determine output directory
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "daily_news" / date_str / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Run news scraper
    # The news_scraper writes to data/daily_news/{date}/raw/news.json
    # We'll read from there after it runs
    news_output_path = output_dir.parent / "raw" / "news.json"
    
    # If news file doesn't exist, we'd need to run the scraper
    # For now, we'll assume it exists or will be created
    news_data = {}
    if news_output_path.exists():
        with open(news_output_path, 'r') as f:
            news_data = json.load(f)
    
    # Step 2: Run daily bias computation
    # The daily_bias_computing writes to data/daily_news/{date}/raw/daily_bias.json
    bias_output_path = output_dir.parent / "raw" / "daily_bias.json"
    
    # If bias file doesn't exist, we'd need to run the computation
    # For now, we'll assume it exists or will be created
    bias_data = {}
    if bias_output_path.exists():
        with open(bias_output_path, 'r') as f:
            bias_data = json.load(f)
    
    # Step 3: Combine into PremarketContext structure
    symbol_biases: Dict[str, PremarketBias] = {}
    
    # Extract bias data per symbol
    if "symbols" in bias_data:
        for symbol_info in bias_data["symbols"]:
            symbol = symbol_info.get("symbol", "")
            if not symbol:
                continue
            
            # Extract from daily_bias.json structure
            symbol_biases[symbol] = PremarketBias(
                symbol=symbol,
                daily_bias=symbol_info.get("bias", "choppy"),
                confidence=int(symbol_info.get("confidence", 50)),
                model_output=symbol_info.get("model_output", {}),
                news_summary=_extract_news_summary(symbol, news_data),
                premarket_price=float(symbol_info.get("premarket_price", 0.0)),
                premarket_context=symbol_info.get("premarket_context", ""),
            )
    
    # Extract market context
    market_context = {
        "spy_bias": bias_data.get("spy_bias", {}),
        "vix": bias_data.get("vix", {}),
        "macro_news": news_data.get("macro", []),
    }
    
    return PremarketContext(
        date=date_str,
        symbols=symbol_biases,
        market_context=market_context,
    )


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

