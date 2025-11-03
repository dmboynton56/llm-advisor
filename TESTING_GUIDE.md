# Testing Guide - Command Line Flags & Execution Modes

## Overview

The system has two main scripts for testing and execution:
1. **Premarket Pipeline** (`scripts/run_premarket.py`) - Prepares daily data
2. **Live Trading Loop** (`scripts/run_live_loop.py`) - Runs the trading system

---

## 1. Premarket Pipeline (`scripts/run_premarket.py`)

### Purpose
Gathers premarket data: news, daily bias predictions, and STDEV snapshots.

### Available Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--date` | string | Today's date | Trading date in `YYYY-MM-DD` format |
| `--symbols` | list | Watchlist from config | Symbols to process (space-separated) |
| `--output` | string | `data/daily_news/{date}/processed/` | Custom output directory |

### Examples

```bash
# Run for today with default symbols
python scripts/run_premarket.py

# Run for specific date
python scripts/run_premarket.py --date 2025-01-07

# Run for specific symbols
python scripts/run_premarket.py --symbols SPY QQQ NVDA

# Run for specific date and symbols
python scripts/run_premarket.py --date 2025-01-07 --symbols SPY QQQ NVDA TSLA

# Custom output directory
python scripts/run_premarket.py --date 2025-01-07 --output /path/to/output
```

### Output
Creates `premarket_context.json` in the output directory containing:
- Daily bias predictions per symbol
- News summaries
- STDEV snapshots (HTF stats, 5m bands)

---

## 2. Live Trading Loop (`scripts/run_live_loop.py`)

### Purpose
Runs the main trading loop that:
- Loads premarket data
- Computes features
- Evaluates thresholds
- Runs periodic LLM market analysis
- Executes trades (if enabled)

### Available Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--date` | string | Today's date | Trading date in `YYYY-MM-DD` format |
| `--symbols` | list | Watchlist from config | Symbols to trade (space-separated) |
| `--test` | flag | False | **Test mode** - Uses historical data (currently not fully implemented) |
| `--fast` | integer | 60 | Loop interval in seconds (how often to check markets) |
| `--output` | string | `data/daily_news/{date}/processed/` | Output directory for logs |

### Environment Variables (Affect Execution)

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_PAPER_TRADING` | `true` | Use paper trading account (`true`/`false`) |
| `ALPACA_API_KEY` | Required | Alpaca API key |
| `ALPACA_SECRET_KEY` | Required | Alpaca secret key |
| `LLM_PROVIDER` | `openai` | LLM provider (`openai`, `anthropic`) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `MARKET_ANALYSIS_INTERVAL_MINUTES` | `15` | How often to run LLM market analysis |
| `ENABLE_TRADE_VALIDATION` | `true` | Enable LLM trade validation (`true`/`false`) |

### Execution Modes

#### Mode 1: Dry Run (No Trading)
**Default behavior when Alpaca credentials are missing or order manager fails to initialize.**

- All calculations run normally
- Signals are detected and logged
- **No actual trades are executed**
- Perfect for testing logic without risking capital

```bash
# Dry run (no trades executed)
python scripts/run_live_loop.py --date 2025-01-07
```

#### Mode 2: Paper Trading
**Default when `ALPACA_PAPER_TRADING=true`**

- Uses Alpaca paper trading account
- **Real API calls, fake money**
- Trades execute but don't affect real account
- Best for testing execution logic

```bash
# Paper trading (requires Alpaca credentials)
export ALPACA_PAPER_TRADING=true
python scripts/run_live_loop.py --date 2025-01-07 --symbols SPY QQQ
```

#### Mode 3: Live Trading
**When `ALPACA_PAPER_TRADING=false`**

- Uses real Alpaca account
- **Real trades with real money**
- Only use after thorough testing!

```bash
# Live trading (REAL MONEY - use with caution!)
export ALPACA_PAPER_TRADING=false
python scripts/run_live_loop.py --date 2025-01-07 --symbols SPY
```

#### Mode 4: Fast Testing (`--fast` flag)
**Adjusts loop speed for testing**

- Lower values = faster iterations (more frequent checks)
- Higher values = slower iterations (less frequent checks)
- Useful for rapid testing or backtesting

```bash
# Check every 15 seconds (for testing)
python scripts/run_live_loop.py --date 2025-01-07 --fast 15

# Check every 5 minutes (normal production)
python scripts/run_live_loop.py --date 2025-01-07 --fast 300
```

#### Mode 5: Test Mode (`--test` flag)
**Note: Currently not fully implemented, but framework exists**

- Intended to use historical data instead of live feeds
- Would simulate past dates for backtesting
- Future enhancement

```bash
# Test mode (framework exists, needs implementation)
python scripts/run_live_loop.py --date 2025-01-07 --test
```

### Examples

```bash
# Basic dry run (no trades)
python scripts/run_live_loop.py --date 2025-01-07

# Paper trading with specific symbols
python scripts/run_live_loop.py --date 2025-01-07 --symbols SPY QQQ NVDA

# Fast testing (15 second intervals)
python scripts/run_live_loop.py --date 2025-01-07 --fast 15

# Custom output directory
python scripts/run_live_loop.py --date 2025-01-07 --output /path/to/logs

# Disable LLM trade validation (for faster testing)
export ENABLE_TRADE_VALIDATION=false
python scripts/run_live_loop.py --date 2025-01-07
```

---

## 3. Logging & Output

### Premarket Pipeline Output
- `data/daily_news/{date}/processed/premarket_context.json` - Combined premarket data

### Live Loop Output
- `data/daily_news/{date}/processed/live_loop_log.jsonl` - JSON Lines log file
  - Each line is a JSON object with symbol states, signals, and metadata
  - Can be parsed for analysis or debugging

### Log Format Example
```json
{
  "ts": "2025-01-07T14:30:00Z",
  "symbols": {
    "SPY": {
      "status": "mr_armed",
      "side": "long",
      "z": 1.25,
      "mu": 450.50,
      "sigma": 2.30,
      "trade": null
    }
  },
  "signals": [],
  "label": "14:30:00 ET",
  "loop_count": 30
}
```

---

## 4. Testing Workflow

### Step 1: Run Premarket Pipeline
```bash
# Prepare data for a specific date
python scripts/run_premarket.py --date 2025-01-07 --symbols SPY QQQ NVDA
```

### Step 2: Run Live Loop (Dry Run)
```bash
# Test without executing trades
python scripts/run_live_loop.py --date 2025-01-07 --symbols SPY QQQ NVDA --fast 15
```

### Step 3: Check Logs
```bash
# View live loop logs
cat data/daily_news/2025-01-07/processed/live_loop_log.jsonl | jq '.'

# View only signals
cat data/daily_news/2025-01-07/processed/live_loop_log.jsonl | jq '.signals'
```

### Step 4: Paper Trading (When Ready)
```bash
# Test with paper account
export ALPACA_PAPER_TRADING=true
python scripts/run_live_loop.py --date 2025-01-07 --symbols SPY
```

---

## 5. Troubleshooting

### Premarket Pipeline Fails
- **Check**: Do you have Alpaca API credentials?
- **Check**: Are the symbols valid?
- **Check**: Is there enough historical data for the date?

### Live Loop Fails to Start
- **Check**: Did you run premarket pipeline first?
- **Check**: Does `premarket_context.json` exist?
- **Check**: Are Alpaca credentials set?

### No Trades Executing
- **Check**: Is `ALPACA_PAPER_TRADING` set correctly?
- **Check**: Are thresholds being met?
- **Check**: Is LLM validation rejecting trades?
- **Check**: Logs for signal detection

### LLM Calls Failing
- **Check**: Do you have OpenAI/Anthropic API keys?
- **Check**: Is `LLM_PROVIDER` set correctly?
- **Note**: System will continue with default multipliers if LLM fails (graceful degradation)

---

## 6. Configuration

### Settings File: `src/core/config.py`
Default settings can be overridden via environment variables:

```bash
# Trading settings
export WATCHLIST="SPY,QQQ,NVDA"
export TRADING_WINDOW_START="09:30"
export TRADING_WINDOW_END="12:00"

# Risk settings
export MAX_RISK_PER_TRADE_PERCENT=1.0
export MIN_RISK_REWARD_RATIO=1.5

# LLM settings
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4o-mini"
export MARKET_ANALYSIS_INTERVAL_MINUTES=15
export ENABLE_TRADE_VALIDATION=true
```

---

## 7. Best Practices

1. **Always test in dry-run mode first** - Verify logic works before risking capital
2. **Use paper trading for extended testing** - Test execution without real money
3. **Start with `--fast 15` for testing** - Faster iterations help catch issues quickly
4. **Check logs regularly** - Monitor signal detection and state transitions
5. **Use specific dates for reproducibility** - `--date` flag ensures consistent testing
6. **Limit symbols during initial testing** - Start with 1-2 symbols before scaling up

