# Pipeline Fix Summary

## Issues Identified

### 1. **0 Bias Symbols** ❌
**Problem:** `bias_gatherer.py` was only reading files, not actually running the scripts that create them.

**Root Cause:** 
- `bias_gatherer.py` assumed files already existed
- Never executed `news_scraper.py` or `daily_bias_computing.py`
- Files didn't exist → empty bias data

**Fix Applied:**
- Now actually runs `news_scraper.py` via subprocess (like `main.py` does)
- Now actually runs `daily_bias_computing.py` via subprocess  
- Handles output file detection and copying between date folders

### 2. **Authentication Error** ❌
**Problem:** `snapshot_builder.py` created `StockHistoricalDataClient()` without credentials.

**Root Cause:**
- Direct instantiation: `StockHistoricalDataClient()`
- Should use `AlpacaDataClient` wrapper which handles `.env` credentials

**Fix Applied:**
- Now uses \`AlpacaDataClient\` which properly loads credentials from environment
- Falls back to provided client if one is passed

### 3. **Date Parameter Support**
**Problem:** Old scripts (`news_scraper.py`, `daily_bias_computing.py`) use current date by default, no date parameter.

**Current Limitation:**
- Scripts run for "today's" date
- For historical dates (like 2025-10-29), scripts will generate data for today
- Copy logic attempts to handle this but it's a workaround

**Better Solution Needed (Future):**
- Modify `news_scraper.py` and `daily_bias_computing.py` to accept `--date` parameter
- OR set date-based environment variables more comprehensively

## Changes Made

### `src/premarket/bias_gatherer.py`
✅ Now runs `news_scraper.py` via subprocess  
✅ Now runs `daily_bias_computing.py` via subprocess  
✅ Handles different JSON formats (dict vs list structure)  
✅ Filters symbols to requested list  
✅ Normalizes bias strings (lowercase)  
✅ Handles confidence as both float (0-1) and int (0-100)  

### `src/premarket/snapshot_builder.py`
✅ Uses `AlpacaDataClient` for proper credential handling  
✅ Added `trading_date` parameter (for future use)  

### `scripts/run_premarket.py`
✅ Passes `trading_date` to snapshot builder  

## Testing

Now you can run:

```bash
# Should work (if you have Alpaca API keys in .env)
python scripts/run_premarket.py --date 2025-10-29 --symbols SPY QQQ NVDA
```

**Expected behavior:**
1. Runs `news_scraper.py` → creates `data/daily_news/{today}/raw/news.json`
2. Runs `daily_bias_computing.py` → creates `data/daily_news/{today}/raw/daily_bias.json`
3. Copies files to requested date folder if different
4. Reads and combines into `premarket_context.json`
5. Builds snapshots using Alpaca API (with credentials)

## Remaining Limitations

1. **Historical dates**: Scripts use today's date. For 2025-10-29, you'd need to either:
   - Run on that actual date
   - Modify scripts to accept date parameter
   - Manually copy pre-generated files

2. **Date inconsistency**: If you request Oct 29 but today is Nov 3, scripts generate Nov 3 data but we copy it to Oct 29 folder (workaround)

## Next Steps (Optional Improvements)

1. Add `--date` parameter to `news_scraper.py` and `daily_bias_computing.py`
2. Create wrapper functions that can be imported (not just subprocess calls)
3. Better date handling throughout the pipeline

