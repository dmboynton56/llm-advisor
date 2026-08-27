# EOD Broker Reconciliation Gap Analysis

## Overview

This document explains why the broker reconciliation gap exists and why it's expected behavior, not a bug.

## The Two P&L Measures

### 1. Broker Daily P&L (from Alpaca)

**Source**: Alpaca `/v2/account` endpoint, field `daily_pnl`

**Calculation**: `equity - last_equity`
- `equity`: Current account equity (cash + market value of positions)
- `last_equity`: Equity as of prior trading day's 16:00 ET close

**Includes**:
- ✅ All realized P&L from fills
- ✅ **Unrealized P&L from open positions**
- ✅ **Commissions and fees**
- ✅ Other broker adjustments
- ⚠️ Timezone: Based on broker's trading day definition (16:00 ET to 16:00 ET)

### 2. Booked Realized P&L (from telemetry)

**Source**: Live loop telemetry `option_exit_filled` events

**Calculation**: Sum of `realized_pnl` field from all exit fill events for the day

**Includes**:
- ✅ Realized P&L from completed exits
- ❌ **Does NOT include unrealized P&L from open positions**
- ❌ **Does NOT include commissions/fees** (unless captured in event details)
- ❌ Fills that didn't emit events (rare, but possible if telemetry fails)
- ⚠️ Timezone: Based on `run_date` (likely UTC day boundary)

## Why Gaps Are Expected

### 1. Open Positions at EOD ($200-400 typical)
If any position is held past the account snapshot time, broker daily_pnl includes its unrealized P&L, but booked_realized_pnl does not.

**Example**:
- Enter 3 contracts at $6.06, now worth $6.50
- Unrealized: +$132 (= 3 × 100 × ($6.50 - $6.06))
- Broker daily_pnl: includes +$132
- Booked realized_pnl: $0 (no exit yet)
- **Gap: $132**

### 2. Commissions and Fees ($1-10 per trade)
Alpaca charges:
- Options: $0.65 per contract (entry + exit = $1.30/contract round trip)
- Regulatory fees: $0.01-0.03 per contract

**Example**:
- 3 contracts round trip = 3 × $1.30 = $3.90 in commissions
- Booked P&L from fill: +$483 (gross)
- Broker P&L after fees: +$479.10 (net)
- **Gap: -$3.90**

### 3. Timezone Day Boundaries (variable)
- Alpaca trading day: 16:00 ET to 16:00 ET (next day)
- Telemetry run_date: Typically UTC date when exit occurred
- Edge case: Fill at 8:00 PM ET might be "tomorrow" in UTC, "today" in ET

### 4. Partial Exits / Tiered Exits (rare)
If the live loop uses tiered exits (TP1, TP2, runner), multiple partial fills may occur. Our reconciliation sums all `option_exit_filled` events, but timing of `option_partial_exit_filled` vs final exit can affect which day the P&L is booked.

## August 27, 2026 Gap: -$320.22

**Facts**:
- Booked lifecycle PnL: -$925.00
- Broker daily PnL: -$1,245.22
- Gap: -$320.22 (broker P&L more negative)

**Likely explanation**:
1. **Open position unrealized loss at EOD** (~$300): If a position entered near market close and moved against us before the snapshot, broker equity includes the unrealized loss but booked P&L is still $0.
2. **Commissions**: 7 exits × ~$4-5 round trip = ~$30-35
3. **Rounding**: Minor differences in how broker vs telemetry calculate P&L

**Not a bug if**:
- No data entry errors in telemetry
- Broker account was genuinely down more due to open positions
- Fees are not tracked in realized_pnl field

## When to Investigate

**Safe gaps** (do not investigate):
- |gap| < $500 and no other anomalies
- Hints: `likely_fees_or_open_positions`

**Investigate immediately**:
- |gap| > $1000 consistently
- Gap trends in one direction over multiple days
- Booked P&L = $0 but broker daily_pnl ≠ $0 (missing telemetry)
- Hints: `significant_gap_investigate`

## How to Debug a Gap

1. **Check reconciliation details JSON** in Supabase:
   ```sql
   SELECT reconciliation_date, booked_realized_pnl, broker_daily_pnl, pnl_gap, details
   FROM llm_advisor_broker_reconciliation_daily
   WHERE reconciliation_date = '2026-08-27';
   ```

2. **Look at details fields**:
   - `broker_equity`: Current account equity
   - `broker_last_equity`: Prior day equity
   - `snapshot_captured_at`: When broker snapshot was taken
   - `final_exit_at`: When last exit occurred
   - `snapshot_after_final_exit`: Did snapshot happen after all exits?

3. **Check trade lifecycles**:
   ```sql
   SELECT lifecycle_uid, symbol, opened_at, closed_at, realized_pnl, status
   FROM llm_advisor_trade_lifecycles
   WHERE opened_at::date = '2026-08-27' OR closed_at::date = '2026-08-27';
   ```

4. **Check order events**:
   ```sql
   SELECT event_type, event_ts, symbol, details->>'realized_pnl'
   FROM llm_advisor_order_events
   WHERE run_date = '2026-08-27' AND event_type = 'option_exit_filled';
   ```

5. **Look for open positions at EOD**:
   ```sql
   SELECT lifecycle_uid, symbol, opened_at, status, details
   FROM llm_advisor_trade_lifecycles
   WHERE opened_at::date <= '2026-08-27' 
     AND (closed_at IS NULL OR closed_at::date > '2026-08-27')
     AND status = 'open';
   ```

## Recommendations

1. **Do not widen tolerance** just to make alerts go away
2. **Expect gaps < $500** as normal operational variance
3. **Investigate patterns** (not one-off gaps)
4. **Track open position count** at EOD to correlate with gap size
5. **Consider adding commission tracking** to fill events if precision matters

## Future Improvements

1. **Parse commission details** from Alpaca fill responses and include in `realized_pnl`
2. **Add open_positions_count** to reconciliation details
3. **Track unrealized_pnl_at_snapshot** separately
4. **Align timezone boundaries** between broker and telemetry
5. **Add reconciliation.expected_gap** field with formula: `commissions + unrealized_open_pnl`

## References

- Alpaca Account API: https://docs.alpaca.markets/reference/getaccount-1
- Code: `scripts/run_eod_aggregate.py::build_broker_reconciliations()`
- Schema: `sql/006_broker_reconciliation.sql`
