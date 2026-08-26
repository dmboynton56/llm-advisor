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

If you want to talk about trading systems, machine learning, or a collaboration:

- Email: [dmboynton6@gmail.com](mailto:dmboynton6@gmail.com)
- LinkedIn: [Drew Boynton](https://www.linkedin.com/in/drew-boynton-1bba16180/)
- GitHub: [dmboynton56](https://github.com/dmboynton56)
