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
WATCHLIST=SPY,QQQ,IWM
MONITOR_ONLY_SYMBOLS=AAPL,MSFT,GOOG,TSLA
EXPERIMENTAL_PAPER_START=2026-10-05
OPTION_DTE_MIN=7
OPTION_DTE_MAX=14
OPTION_DELTA_MIN=0.35
OPTION_DELTA_MAX=0.55
MAX_RISK_PER_TRADE_PERCENT=3.0
MAX_CONCURRENT_TRADES=3
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
- Current paper sizing trial: at most three concurrent positions, a 3 percent
  equity premium budget (capped at $3,000) per trade, and a matching $3,000
  fallback cap. The current workflow's late-session canary caps gross planned
  premium at 9 percent / $9,000. The 25 percent premium target, 35 percent
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

## Monitoring Expansion

Premarket and Live Loop monitor SPY, QQQ, IWM, AAPL, MSFT, GOOG, and TSLA.
The scheduler still passes the existing three ETF entry candidates; both
entrypoints append `MONITOR_ONLY_SYMBOLS` without changing workflow inputs or
artifact paths. Stock bars remain batched across the universe. Missing bias
models for observation symbols remain explicit in the premarket artifact and
do not become fabricated neutral approvals or daily degraded-mode alerts.

Signals for observation symbols emit `monitor_signal_detected` and a terminal
`signal_outcome` with `outcome=monitor_only`. They are also saved in the
warehouse signal table. Live signals also emit `shadow_trade_decision` with
`action=would_buy|skipped|error`, validation evidence, the proposed option plan
when one exists, and entry-check failures. The preview shares candidate,
pricing, risk, buying-power, and broker exposure checks with real paper entry,
but never calls order submission. Backtests continue to record monitor signals
without broker/LLM calls. Previews do not enter trading approval-rate or fill
counts. EOD sync makes them available in the Overview's Stock experiments card.

These are decisions at the observed quotes, **not simulated fills or a shadow
portfolio**: they do not reserve hypothetical capital or produce hypothetical
P&L. Each later preview checks the actual paper account at that moment.

The configured trial is October 1 and October 2, 2026. Both Premarket and
Live Segment set `EXPERIMENTAL_PAPER_START=2026-10-05`; on/after that ET date,
only the four named stock experiments become entry eligible even though the
scheduler still passes three ETFs. This date controls eligibility, not a
guarantee that trades will pass the normal gates. Unset the date or move it
later in both environments to extend the trial if a bug appears. Other monitor
symbols remain blocked regardless of this date.

The four stocks do not yet have trained daily-bias models. They can use their
recorded news and technical context, with an explicit `premarket_data_quality`
warning for missing ML. Missing symbol context, invalid LLM responses, risk,
HTF alignment, liquidity, and broker checks still reject. ETF ML failures remain
hard vetoes. Any future stock outside this first cohort requires available ML
bias and explicit promotion through the entry list and monitor-list removal.
An empty `MONITOR_ONLY_SYMBOLS` value disables the extra list locally.

Every option buy checks broker positions and pending buys immediately before
submission. Any open option or pending buy on the same underlying blocks the
new entry, regardless of call/put, strike, or expiry. A duplicate contract is
terminal for that signal; selecting another strike cannot bypass the rule.
Position or open-order lookup failures block entry until broker state is known.
Existing positions continue through their normal management and exit paths.

Run the local policy gate before merging:

```bash
python -m pytest tests/unit
npx --yes promptfoo@0.123.1 eval --config evals/paper-entry-policy/promptfooconfig.yaml \
  --output evals/paper-entry-policy/results/eval-results.json
```

The Promptfoo provider exercises the real order-entry boundary with simulated
broker responses. It makes no broker or LLM network calls.

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
