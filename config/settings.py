# ==============================================================================
# Liquidity Flow Agent - Configuration File
# ==============================================================================

# --- API Keys -----------------------------------------------------------------
# These should be stored in your .env file, not here. This file will read them.
# Example .env file:
# ALPACA_API_KEY="PK..."
# ALPACA_SECRET_KEY="sk..."
# ALPACA_PAPER_TRADING="true"
# OPENAI_API_KEY="sk-..."
# ANTHROPIC_API_KEY="sk-..."

# --- Watchlist ----------------------------------------------------------------
# List of symbols the bot will actively monitor.
WATCHLIST = [
    "SPY", "QQQ", "IWM", "ES=F", "NQ=F",  # Indices & Futures
    "NVDA", "TSLA", "AAPL", "AMZN",       # Tech Stocks
    "META", "MSFT", "GOOG"                # More Tech Stocks
]

# --- Trading Parameters -------------------------------------------------------
# The bot will only trade one position at a time.
MAX_CONCURRENT_TRADES = 1

# The window during which the bot is allowed to open new trades.
# Format: "HH:MM" in US/Eastern timezone.
TRADING_WINDOW_START = "09:30"
TRADING_WINDOW_END = "12:00"

# The time to close all open positions, regardless of P/L.
END_OF_DAY_CLOSE_TIME = "15:50"

# --- Risk Management ----------------------------------------------------------
# The maximum percentage of total account equity to risk on a single trade.
# Example: 1.0 means 1% risk.
MAX_RISK_PER_TRADE_PERCENT = 50.0

# The minimum required risk-to-reward ratio for a trade to be considered valid.
# Example: 2.5 means the potential profit must be at least 2.5 times the potential loss.
MINIMUM_RISK_REWARD_RATIO = 2.0

# --- Strategy & AI Parameters -------------------------------------------------
# The confidence score threshold required to trigger a trade.
# This is the aggregated score from the LLM analysis.
CONFIDENCE_THRESHOLD = 85  # Integer from 0 to 100

# The LLM provider(s) to use for analysis. Can be "openai", "anthropic", or a list.
LLM_PROVIDERS = ["openai"]

# The specific model to use for the main analysis.
LLM_MODEL = "gpt-4o"

# The time in seconds the bot will wait between each analysis loop.
ANALYSIS_INTERVAL_SECONDS = 60

# --- Logging Configuration ----------------------------------------------------
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE = "logs/trading_agent.log"
