# LLM Advisor

LLM Advisor is my paper-trading project. It tests whether an LLM can add
useful context to a rules-based trading system without taking control away
from the rules.

## The journey

I started in October 2025 with a daily-bias script. It collected market news
and context and produced a premarket briefing. I then added a live loop for
technical signals. Standard-deviation features became the core of the system,
with mean-reversion and trend-continuation setups.

From there, the project grew through several layers:

- Machine-learning models added a daily bias for each symbol.
- OpenAI gpt-5.4-nano added a periodic review of market conditions and adjusted
  signal thresholds.
- Alpaca added paper execution and position tracking.
- Risk checks and state recovery made the loop safer to run.
- BigQuery and Supabase made runs, trades, heartbeats, and order events
  visible.
- A Next.js operations dashboard made the decision trail easier to follow.

The old standalone ICTML daily-bias project is now part of this story. Bringing
it into LLM Advisor made the system easier to explain and maintain.

## What it is now

LLM Advisor has a premarket pipeline, a live signal loop, an LLM context layer,
and a paper execution path. BigQuery stores detailed run data. Supabase serves
telemetry and portfolio metrics. GitHub Actions and Google Cloud Scheduler run
the weekday workflows. Discord sends alerts and heartbeat updates.

The default execution mode is options-first paper trading. Stock execution is
opt-in. The system does not trade live money.

The first coverage expansion follows AAPL, MSFT, GOOG, and TSLA alongside
SPY, QQQ, and IWM. On October 1 and October 2, 2026, the four stocks run
full decision previews: validation, option selection, risk, buying power, and
broker exposure checks. Proposed contracts and reasons to skip trades are
recorded without placing orders. From October 5, they are eligible for paper
entries under the same limits. A ticker with an open option or pending buy
cannot receive another entry, even in the opposite direction.

`WATCHLIST` (or `--symbols`) selects entry candidates. `MONITOR_ONLY_SYMBOLS`
adds tracked symbols and blocks their entries even if they also appear in
the entry list. The explicit `EXPERIMENTAL_PAPER_START` date promotes only
AAPL, MSFT, GOOG, and TSLA after their trial; leaving it unset keeps them in
preview mode. Those four experiments use news and technical context while
trained ML bias models are unavailable, with that limitation recorded as a
warning. Core ETFs still require available ML bias. Both Premarket and Live
Loop append the tracked list, so the scheduler's existing inputs continue to
work. See the
[options paper runbook](docs/options_paper_runbook.md) for promotion and checks.

The Overview starts with actual broker equity (cash plus open-position value),
with Day, Week, Month, and Year ranges. P&L by exit date stays below it. Booked
lifecycle P&L, broker daily P&L, and their gap remain separately available under
“Why P&L numbers differ.” The equity curve does not reconstruct balances from
realized trade results.

### Built with

- Python, pandas, NumPy, scikit-learn, and XGBoost
- OpenAI gpt-5.4-nano
- Alpaca Markets paper trading API
- Google BigQuery
- Supabase and Postgres
- GitHub Actions and Google Cloud Scheduler
- Next.js, React, TypeScript, and Tailwind CSS for the dashboard
- Discord webhooks for alerts

## Next

The current focus is evaluation and observability. I want to understand which
parts of the system help, which parts add noise, and how well the process holds
up over time. The project is about disciplined experimentation, not adding AI
for its own sake.

## Contact

- Email: [dmboynton6@gmail.com](mailto:dmboynton6@gmail.com)
- LinkedIn: [Drew Boynton](https://www.linkedin.com/in/drew-boynton-1bba16180/)
