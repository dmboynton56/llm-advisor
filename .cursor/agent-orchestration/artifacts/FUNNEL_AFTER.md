# Trade Funnel After Sizing and Window Changes

Generated: 2026-07-15

Backtests were force-rerun for every recent date with locally available historical bars (2025-04-30 through 2025-05-02), using SPY/QQQ/IWM, a 09:30–15:30 ET entry window, and the new 2% risk default. Only three runnable dates are present in this checkout; no additional 5–10-date history is locally available.

## Before/after

| Metric | Previous artifacts | New run | Change |
|---|---:|---:|---:|
| Signals | unavailable in old structured telemetry | 99 | n/a |
| Execution attempts | unavailable | 90 | n/a |
| Successful executions / trades | 29 | 27 | -2 |
| Execution rate (fills / signals) | unavailable | 27.3% | n/a |
| Average stock position notional | $522,625 | $190,556 | -63.5% |
| Aggregate PnL | $628.36 | $2,310.05 | +$1,681.69 |

| Date | Signals | Attempts | Failed | Filled | Trades | Avg position | PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-04-30 | 34 | 31 | 18 | 12 | 12 | $191,725.88 | $2,344.87 |
| 2025-05-01 | 20 | 19 | 6 | 9 | 9 | $190,589.63 | $789.67 |
| 2025-05-02 | 45 | 40 | 30 | 6 | 6 | $188,164.80 | -$824.49 |
| **Total** | **99** | **90** | **54** | **27** | **27** | **$190,555.78** | **$2,310.05** |

Dominant execution failure: `rr_below_min` (54). This is the stock `MockOrderManager` backtest path; it does not exercise option-contract selection or the $2,000 premium cap, so it cannot prove live option fill rates.

Three positions were closed at simulated EOD and are now explicitly tagged `backtest_eod`. `MockOrderManager` still flattens at the end of each simulated day, so multi-day option holds remain untested and those exits should be excluded from exit-quality conclusions.

The large change in average stock notional exposes a backtest/live parity problem: `MockOrderManager` sizes underlying shares and does not model option premium. These notionals are not comparable to the live $2,000 option budget, so live validation still requires a new paper run with complete `order_events.jsonl` telemetry.
