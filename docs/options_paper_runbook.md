# Options Paper Runbook

This project is now options-first for live paper execution. Stock STDEV signals
still drive entries, but live paper orders are expressed as option contracts.

## Required Environment

```text
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_PAPER_TRADING=true
TRADING_INSTRUMENT=options
OPTIONS_PAPER_ONLY=true
ALLOW_STOCK_FALLBACK=false
OPTIONS_STRATEGY_TYPE=single_long
OPTION_DTE_MIN=7
OPTION_DTE_MAX=14
OPTION_DELTA_MIN=0.35
OPTION_DELTA_MAX=0.55
MAX_RISK_PER_TRADE_PERCENT=3.0
MAX_CONCURRENT_TRADES=2
MAX_OPTION_PREMIUM_PER_TRADE=3000
OPTION_FALLBACK_MAX_PREMIUM_PER_TRADE=3000
MAX_OPTION_BID_ASK_SPREAD_PCT=0.15
MIN_OPTION_OPEN_INTEREST=100
OPTION_STRIKE_WINDOW_PCT=0.10
OPTION_PROFIT_TARGET_PCT=0.25
OPTION_STOP_LOSS_PCT=0.35
OPTION_TIERED_EXIT_ENABLED=true
OPTION_TIERED_EXIT_UNDERLYINGS=SPY,QQQ
OPTION_TIERED_MIN_CONTRACTS=4
OPTION_TIERED_TP1_RETURN_PCT=0.25
OPTION_TIERED_TP1_FRACTION=0.50
OPTION_TIERED_TP2_RETURN_PCT=0.50
OPTION_TIERED_TP2_FRACTION=0.25
OPTION_TIERED_POST_TP1_STOP_RETURN_PCT=-0.05
OPTION_TIERED_RUNNER_FLOOR_RETURN_PCT=0.25
OPTION_TIERED_RUNNER_GIVEBACK_PCT=0.25
OPTION_TIERED_EXIT_FILL_TIMEOUT_SECONDS=120
OPTION_TIERED_EMERGENCY_FLATTEN=false
OPTION_MAX_HOLD_MINUTES=2880
OPTION_CLOSE_AT_ENTRY_WINDOW_END=false
OPTION_ALLOW_OVERNIGHT=false
OPTION_EOD_FLATTEN_MAX_DTE=0
OPTION_DATA_FEED=indicative
```

`OPTION_DATA_FEED=opra` requires the appropriate Alpaca data subscription.

## Execution Flow

1. Run premarket context.

```bash
python3 scripts/run_premarket.py --symbols SPY QQQ IWM --use-db
```

2. Use MCP readonly checks from `docs/alpaca_mcp_workflow.md`.

3. Run the live loop.

```bash
python3 scripts/run_live_loop.py --symbols SPY QQQ IWM --use-db --fast 60
```

4. Review order events.

```bash
tail -n 50 data/daily_news/$(date +%F)/processed/order_events.jsonl
```

5. Run EOD aggregation after the live loop completes.

```bash
python3 scripts/run_eod_aggregate.py --date $(date +%F)
```

Review the trial after five market sessions. Continue until at least 12 eligible
tiered lifecycles or ten sessions, whichever comes first. Keep paper size fixed;
the replay gate must show non-negative cumulative P/L delta, no lower average
winner, and no lifecycle return more than five percentage points worse than the
legacy counterfactual before extending the trial.

## Current Strategy Mapping

- Bullish signal: buy one or more calls.
- Bearish signal: buy one or more puts.
- Contract filter: active, tradable, 7-14 DTE by default, strike within 10
  percent of underlying, absolute delta 0.35-0.55, open interest at least 100,
  and bid/ask spread at most 15 percent.
- Order type: paper limit buy to open at midpoint plus configured buffer.
- Max loss: premium paid.
- Current paper sizing trial: at most two concurrent positions, a 3 percent
  equity premium budget (capped at $3,000) per trade, and a matching $3,000
  fallback cap. This is approximately 6 percent gross planned premium
  exposure, matching the prior 2 percent x 3 profile while increasing
  individual contract counts. The 25 percent premium target, 35 percent
  premium stop, 48-hour time stop, and end-of-day/DTE safety rules remain in
  force; this sizing change is paper-only.
- Tiered paper trial: newly filled SPY/QQQ positions with at least four
  contracts use 50 percent TP1 at +25 percent, 25 percent TP2 at +50 percent,
  then one or more runner contracts with a +25 percent floor and 25-point
  giveback. IWM, smaller positions, stocks, and positions recovered without a
  persisted tier state remain on the legacy full-position policy.

Debit spreads are intentionally not enabled yet. The SDK supports multi-leg
request shapes, but single-leg long premium gives the first clean comparison
between current STDEV signals and option expression without adding spread
construction risk.

## Verification Points

- Startup fails if `TRADING_INSTRUMENT=options`, `OPTIONS_PAPER_ONLY=true`, and
  `ALPACA_PAPER_TRADING=false`.
- `order_events.jsonl` should include `option_plan` details on successful or
  rejected option attempts.
- Storage `trades.symbol` and `positions.symbol` can hold option symbols in new
  databases created from the base migration.
- Existing databases created before this change may need a migration from
  `VARCHAR(10)` to `VARCHAR(32)` for symbol columns before persisting option
  symbols.
