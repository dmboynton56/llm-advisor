# LLM Advisor — agent brief

Live **options paper-trading advisor dashboard**. Rules + STDEV signals own entries; `gpt-5.4-nano` reviews context and thresholds. Alpaca paper executes. Next.js ops UI at `web/` reads Supabase; BigQuery holds run detail. No live money.

Default implementer model: **grok-4.6**. Do not default to Sonnet.

## Workflows (do not break)

| Cadence | Workflow | What it does |
|---|---|---|
| Premarket | `.github/workflows/premarket.yml` | Bias + news → `premarket-context` artifact |
| Live Loop | `live_loop.yml` / `live_segment.yml` | Ingest premarket, reconcile warehouse vs Alpaca, trade, upload `llm-advisor-daily-news-*` |
| EOD | `eod_aggregate.yml` | Ingest artifact → BigQuery merge + Supabase upsert; keep running when recon alerts |

Runbook: `LIVE_LOOP_RUNBOOK.md`. Options policy: `docs/options_paper_runbook.md`. Scheduler owns weekday timing; workflows fail closed on NYSE holidays.

## Recon honesty

Dashboard and EOD must **show both** booked lifecycle PnL and broker (Alpaca `daily_pnl` = equity − last_equity). Gaps are expected: open-position MTM, commissions, ET vs UTC day bounds. Do not paper over gaps, widen tolerance to silence alerts, or collapse the two series into one number. See `docs/eod-reconciliation-gap-analysis.md` and `web/app/page.tsx`.

## Runtime model

Advisor LLM is **`gpt-5.4-nano`** (`src/core/config.py`, `config/settings.py`, README). Override only via `LLM_MODEL`. Parse errors are not approvals.

## Non-goals

- Live-money trading, Robinhood enablement, or inventing broker APIs
- Changing Premarket / Live Loop / EOD job contracts, artifact paths, or fail-closed calendar checks
- Hiding lifecycle-vs-broker disagreement
- Adding AI for its own sake — evaluate first

## Before merging

1. Prompt or trading-policy diffs → run **Promptfoo** (PR eval gate). Skill: `.cursor/skills/promptfoo-evaluation/`.
2. Every merge candidate → run **thermo reviews** (`/thermo-nuclear-review` + `/thermo-nuclear-code-quality-review`).
3. Live-loop LLM behavior / traces → **Langfuse**, not Promptfoo. Skill: `.cursor/skills/langfuse/`.

Official skill bodies live **globally** (`~/.cursor/skills/`). This repo only wires when/why. Install once:

```bash
npx skills add https://github.com/cursor/plugins --skill thermo-nuclear-review
npx skills add https://github.com/cursor/plugins --skill thermo-nuclear-code-quality-review
npx skills add https://github.com/daymade/claude-code-skills --skill promptfoo-evaluation
npx skills add langfuse/skills --skill langfuse
```

## Layout

- `src/live/`, `src/execution/`, `src/analysis/` — loop, orders, LLM gates
- `src/premarket/` — daily bias + LLM posterior
- `scripts/run_{premarket,live_loop,eod_aggregate}.py` — job entrypoints
- `web/` — Next.js dashboard (App Router)
- `tests/unit/` — `pytest tests/unit` (do not dispatch trading workflows from a PR)

## Guardrails

- Alpaca is source of truth for open positions at loop startup.
- `TRADING_WINDOW_END` is last new-entry time, not process shutdown.
- EOD no-ops (does not fail) when a live run uploaded only an artifact anchor.
- Secrets stay in GitHub/Vercel env — never commit keys or invent `.cursor/environment.json` secrets.
