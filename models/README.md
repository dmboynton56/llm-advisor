# Daily-bias models

The paper advisor has models for **SPY, QQQ, IWM, AAPL, MSFT, GOOG, and TSLA**.
The original ETF artifacts came from the synthetic CI exporter; SPY alone had
a warning sidecar. Each replacement has an encoder, ordered features and
provenance describing actual historical training and validation.

## Information and target

- Source: Alpaca SIP historical bars, split-adjusted, since January 2020
  (TSLA uses a January 2023 start to keep the history download bounded).
  Actual available dates, skipped sessions and row counts are in provenance.
- Features: 20 price/volume ratios at a fixed **09:00 ET** cutoff. Only completed
  bars contribute. Absolute price/volume levels are excluded so future split
  adjustments cancel out of the feature ratios.
- Target: the NYSE session's final completed 20-minute bar close versus the
  previous Alpaca daily-bar close. A move above/below **0.35 prior daily ATR**
  is bullish/bearish; smaller moves are choppy. Early closes follow the NYSE
  calendar. This predicts total-day closing bias, not a return from an entry.
- Training and inference call the same feature function. Inference requests
  history ending just before 09:00. At the scheduled 09:20 premarket run this
  satisfies free SIP's 15-minute delay. No paid data plan is enabled. Trading
  and independent snapshot feeds retain their existing configuration.

IEX premarket history was too sparse to train several stocks with adequate
coverage. Each model carries its required feed; do not score SIP models with
IEX features.

## Reproduce

From the repository root, with existing Alpaca credentials in the environment
or local `.env` (never commit credentials):

```powershell
python scripts/train_daily_bias_models.py --symbols SPY QQQ IWM AAPL MSFT GOOG `
  --feed sip --end 2026-11-03 `
  --cache-dir tmp/bias-model-history --report-dir tmp/bias-validation `
  --output-dir models
python scripts/train_daily_bias_models.py --symbols TSLA --start 2023-01-01 `
  --feed sip --end 2026-11-03 `
  --cache-dir tmp/bias-model-history --report-dir tmp/bias-validation `
  --output-dir models
```

`--end` is exclusive and excludes today's unfinished session. All seven symbols
are selected by default. Cache keys include feed, split adjustment, dates and
timeframe. Keep caches and training feature tables out of Git.

Candidates are two regularized logistic regressions and two shallow random
forests. Selection uses three expanding chronological folds with a one-session
gap. The last quarter of usable sessions (up to 252, at least 126) is held out;
at least 400 earlier sessions and 20 observations/class are required. Promotion
requires lower holdout log loss than a training-frequency baseline and balanced
accuracy above one third. Rejected models are not exported. Accepted models are
then refit on all completed labeled sessions; provenance distinguishes evaluation
training dates from final refit dates.

The script emits validation JSON and per-session held-out probabilities.
Committed `validation/` CSVs allow independent metric recalculation. Provenance
records dataset and model/encoder/feature hashes. Runtime checks accepted
validation, feature version, cutoff, exact sklearn version and hashes.
`scikit-learn` is pinned for pickle compatibility.

## Limits and operations

These are development holdouts, not an independent forward trading evaluation.
An initial remaining-session-return target did not clear the baseline gate;
the shipped target describes total-day closing bias instead. Classification
skill, confidence percentages and known overnight gaps do not establish a
profitable option strategy. Continue observing actual paper decisions and PnL.

Run unit tests and the existing Promptfoo paper-policy gate before merging.
Do not dispatch a trading workflow from a PR. Scheduled premarket checks all
seven models' loadability and provenance, not just three file paths.

`export_minimal_bias_models.py` remains a CI/development tool. Its random-data
models are placeholders and must not overwrite these historical artifacts.

## Recorded development holdout

Each comparison uses the same held-out sessions for that ticker. The baseline
uses training-set class frequencies. These are classification results, not
trade win rates. Full-precision metrics and date boundaries are in provenance.

| Ticker | Train sessions | Holdout sessions | Accuracy | Baseline accuracy | Log loss | Baseline log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY | 1451 | 252 | 57.1% | 38.9% | 0.921 | 1.090 |
| QQQ | 1451 | 252 | 59.9% | 39.3% | 0.895 | 1.093 |
| IWM | 1451 | 252 | 58.3% | 38.5% | 0.930 | 1.093 |
| AAPL | 1451 | 252 | 53.2% | 49.6% | 0.977 | 1.054 |
| MSFT | 1451 | 252 | 52.8% | 40.1% | 0.981 | 1.094 |
| GOOG | 1315 | 252 | 54.4% | 46.0% | 0.968 | 1.071 |
| TSLA | 711 | 236 | 55.1% | 41.1% | 0.952 | 1.087 |
