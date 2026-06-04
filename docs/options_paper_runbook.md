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
MAX_OPTION_PREMIUM_PER_TRADE=200
MAX_OPTION_BID_ASK_SPREAD_PCT=0.15
MIN_OPTION_OPEN_INTEREST=100
OPTION_STRIKE_WINDOW_PCT=0.10
OPTION_PROFIT_TARGET_PCT=0.25
OPTION_STOP_LOSS_PCT=0.35
OPTION_MAX_HOLD_MINUTES=30
OPTION_CLOSE_AT_ENTRY_WINDOW_END=true
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

## Current Strategy Mapping

- Bullish signal: buy one or more calls.
- Bearish signal: buy one or more puts.
- Contract filter: active, tradable, 7-14 DTE by default, strike within 10
  percent of underlying, absolute delta 0.35-0.55, open interest at least 100,
  and bid/ask spread at most 15 percent.
- Order type: paper limit buy to open at midpoint plus configured buffer.
- Max loss: premium paid.
- Default risk profile: at most one live position, $200 max premium per trade,
  25 percent premium profit target, 35 percent premium stop, 30 minute time
  stop, and forced option close when the entry window ends.

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
