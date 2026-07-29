# Trading System Fix Plan — 2026-07-29

Synthesized from two independent analyses (cross-verified against code, Supabase, and Alpaca).
Shared verdict: **not lifetime-profitable (−$1,614 normalized / −$2,089 at the broker), no measured
edge yet, and the largest loss driver is execution, not signal.** The recent green run is not
statistically significant (Fisher p≈0.061 / binomial P≈0.10) and ran on unchanged code — treat it
as regime + variance, not evidence.

## Consensus findings (all verified in code)

1. **No broker-resident stop.** `options_order_manager.py:129` submits `OrderClass.SIMPLE` (entry
   only). Exits are 60s software polls on `unrealized_plpc` (`trade_tracker.py:291-295`). Loop
   death or overnight gaps mean no protection → realized stops averaged −41.7% vs the −35% policy,
   worst −69.8%. Excess loss beyond designed stops ≈ $1,500–$1,960 — larger than the lifetime loss.
2. **Runner dies before the EOD flatten.** `live_loop.yml` sets `timeout-minutes: 420`, but
   GitHub-hosted runners hard-cap at 6h, so the loop is killed ~15:28 ET — before
   `END_OF_DAY_CLOSE_TIME: 15:50`. This produced the `startup_reconcile_orphan_bq` rows.
3. **Same-contract duplicates net into one Alpaca position.** No OCC-symbol dedup, no
   per-underlying lock, no cooldown; observed 5 concurrent vs `MAX_CONCURRENT_TRADES: 3`, and
   aggregate positions of 8 and 16 contracts vs displayed 4 and 6. Aggregate PnL lands on one DB
   row (two rows show loss > premium paid). Per-row stats and the 2%-per-trade risk cap are
   unreliable whenever this happens.
4. **Exit geometry was never a decision.** `threshold_evaluator._create_signal` computes an
   ATR-based stop with `min_rr_ratio: 1.5`; `trade_tracker._option_exit_reason` discards it and
   checks fixed +25%/−35% on premium. Nominal breakeven hit rate 58.3%; realized breakeven with
   observed stop slippage ≈ 62.4%; realized hit rate 58.1% (18 PT / 13 SL). No demonstrated edge.
5. **The LLM gate is the de facto strategy.** 7/29 funnel: 448 signals → 447 LLM rejections → 1
   trade. MR arms at |z|≥1.2 and triggers on return to ∓0.6, so entries land near z≈0; MR entries
   at |z|<0.6 are −$2,722 across 9 trades (suggestive, small n). Free-text LLM cautions are not
   enforced; confidence is uncalibrated.
6. **Accounting gaps.** Broker equity −$2,089.08 vs booked −$1,613.57 → $475.51 unexplained
   (fees/slippage/unbooked closes). Dashboard attributes PnL to entry date; exit-date attribution
   disagrees on 8 of the last 10 days.
7. **Weak segments (attribution-caveated):** IWM PF 0.61 (−$1,730), MR PF 0.74 (−$1,990), TC ≈
   breakeven. TC entries at |z|≥2: 2W/6L, −$2,344.

## Disagreements — resolutions

- **"Green run = afternoon window change" (AI 2): rejected.** The claim dropped a −$2,196
  afternoon loser; corrected, afternoon = +$182 on 8 trades. No proven cause → no scaling.
- **"Signal is provably zero-edge random walk" (AI 1): softened.** Correct conclusion is *no
  durable edge demonstrated*; barrier arithmetic on 31 resolved trades can't prove equivalence to
  a random walk, especially with realized breakeven ≈62.4%, not 58.3%.
- **Fix ordering:** run execution safety (Phase 1) and measurement (Phase 2) in parallel — safety
  has the P&L impact; measurement is cheap and gates every strategy decision.
- **Bracket/OTOCO orders:** not available for Alpaca options. Design = separate broker-resident
  stop order after the entry fills (Alpaca supports simple stop/stop-limit option orders).
- **Exit inversion to +35/−20:** do NOT ship without a replay backtest — it lowers the breakeven
  rate but may sharply increase stop frequency.
- **MR trigger:** instrument first (log armed z + trajectory), change only with replay evidence.
  Entering after reversion begins may be intentional confirmation.

---

## Phase 0 — Today (config only, stop the tail risk)

- [ ] `live_loop.yml`: set `OPTION_ALLOW_OVERNIGHT: false`. With it false,
      `should_flatten_at_eod` returns true for all options, so the EOD step closes everything.
- [ ] Temporarily set `END_OF_DAY_CLOSE_TIME: '15:00'` so the flatten runs *before* the ~6h
      runner kill (~15:28 ET). Revert to 15:50 once Phase 1.2 lands.
- [ ] No other config changes: no sizing changes, no scaling on the green run.

*Rationale: until a broker-resident stop exists, holding overnight means holding unprotected
through gaps — that is where −69.8% fills and ~$2k of excess loss came from.*

## Phase 1 — Execution safety (highest expected P&L impact)

### 1.1 Broker-resident protective stop
- After entry fill confirmation, submit a resting STOP (or stop-limit) sell on the option for the
  **actual filled qty**, at premium ≈ entry × (1 − stop_loss_pct).
- Profit-target path: cancel the stop first, confirm cancellation, then close (handle the race
  where the stop fills during cancellation).
- Startup reconcile: any open option position without a live stop gets one immediately.
- Options orders are DAY tif → re-submit stops at each session open (required before overnight is
  ever re-enabled).
- **Gate for re-enabling overnight:** demonstrate (paper) that a position stays protected after
  killing the loop mid-session.

### 1.2 Fix the runner lifetime
- Split the live loop into two chained jobs, each well under GitHub's 6h cap (e.g., 09:30–12:45
  and 12:45–16:05), with the afternoon job owning the EOD flatten; or move to a self-hosted
  runner.
- Add a fatal alert (Discord) if the loop process exits before the EOD step has run.
- Then restore `END_OF_DAY_CLOSE_TIME: '15:50'`.

### 1.3 Exposure and duplicate control
- Refuse entry when the broker already holds the same OCC contract.
- One live position per underlying+direction (kills the 3×-stacked-signal days: 7/16, 7/17, 7/21,
  7/27 — the two worst days were triple-stacked).
- Enforce `MAX_CONCURRENT_TRADES` against **live broker positions**, not internal state
  (observed max concurrent = 5 vs cap 3).
- Cooldown after a stop-out (e.g., 60 min per underlying).

## Phase 2 — Measurement & accounting (parallel with Phase 1)

### 2.1 Broker-lifecycle trade records
- Persist actual fill qty/price from order events; link entry→exit by order/position IDs; FIFO or
  pro-rata allocation when lots aggregate. Fixes the loss-greater-than-premium rows and makes
  per-trade stats trustworthy. (1.3 prevents new aggregation; this fixes the record-keeping.)

### 2.2 Daily broker reconciliation
- Nightly job: booked PnL vs Alpaca account equity; alert on divergence. Chase the current
  $475.51 gap (fees, slippage, unbooked closes).

### 2.3 Dashboard attribution
- Attribute daily PnL by **exit date** (or show entry/exit views side by side); label
  orphan-reconcile rows distinctly.

### 2.4 Signal telemetry (prerequisite for all Phase 3 decisions)
- For every signal (approved or rejected): armed_z, z at trigger, z trajectory since arming, ATR,
  HTF bias, and the LLM's structured output (confidence + typed flags), keyed by order_id.

### 2.5 Stop-overshoot monitor
- Alert when a realized loss exceeds the policy stop by >5 pts. Backstop for 1.1, not a
  substitute.

## Phase 3 — Strategy & edge (gated on Phases 1–2 + replay backtest)

### 3.1 Replay backtest harness, then exit geometry
- Build replay using recorded option quotes (or underlying-price invalidation levels).
- Candidates to test: (a) wire the existing ATR/1.5:1 underlying plan through to the option exit
  — the design intent that `trade_tracker` currently discards; (b) alternative premium
  thresholds. Judge against **realized** breakeven (~62% with observed slippage), never the
  nominal 58.3%.

### 3.2 MR trigger
- With 2.4 telemetry + replay: compare trigger-on-return-to-∓0.6 (current) vs entering into
  extension vs a minimum |z| floor at trigger. The |z|<0.6 = −$2,722/9-trades evidence is
  suggestive but too small to act on alone.

### 3.3 LLM gate
- Replace free-text cautions with structured deterministic veto flags; track confidence
  calibration against outcomes. Target end state: the deterministic layer carries the signal and
  the LLM is a reproducible veto — not the strategy (currently 1 approval per 448 signals).

### 3.4 Segments
- Keep sizing flat. Revisit IWM and MR only after 2.1 fixes attribution — current per-row
  segment stats are built on unreliable rows.

### 3.5 Pre-registered scaling criteria (write down before the data arrives)
- Example gate to revisit sizing or overnight: ≥40 post-fix broker-lifecycle exits, PF > 1.2,
  realized hit rate > realized breakeven, zero orphan reconciles, equity-vs-book gap < $50,
  overnight-protection test (1.1 gate) passed.

## Success metrics

| Metric | Now | Target |
|---|---|---|
| Realized stop vs −35% policy | avg −41.7%, worst −69.8% | within 5 pts except gap events |
| Orphan-reconcile trades | 4 | 0 |
| Max concurrent vs cap 3 | 5 | ≤ 3 |
| Same-contract duplicate entries | recurring | 0 |
| Booked PnL vs broker equity gap | $475.51 | < $50, reconciled daily |
| Rows with loss > premium | 2 | 0 |
| Realized hit rate vs realized breakeven | 58.1% vs ~62.4% | tracked per-regime post-fix |

## Explicit non-actions

- No position-size increases or overnight re-enable on the strength of the 7/23–7/29 run.
- No exit-threshold inversion without replay backtest.
- No pausing IWM/MR yet — attribution must be fixed first.
- `mancala_stats` RLS: real but unrelated; enable RLS + policies when next touching that app.
