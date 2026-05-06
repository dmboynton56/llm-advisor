# LLM Advisor Notebooks

This folder contains Week-2 analysis notebooks that back portfolio claims with reproducible outputs.

## Notebooks

- `trade_journal.ipynb`: trade-by-trade narrative and context.
- `pnl_attribution.ipynb`: P&L decomposition by symbol/exit reason/regime.
- `threshold_sensitivity.ipynb`: threshold grid search and sensitivity analysis.
- `prompt_ablation.ipynb`: prompt-version comparison scaffold.
- `premarket_bias_evaluation.ipynb`: premarket bias accuracy and calibration.
- `backtest_roll_summary.ipynb`: aggregate many `backtest_results.json` files into one summary JSON.

## Batch backtests & rollups

From repo root:

```bash
python scripts/run_backtest_batch.py --start YYYY-MM-DD --end YYYY-MM-DD
python scripts/aggregate_backtest_results.py --output notebooks/_exported/backtest_roll_summary.json
```

(`notebooks/_exported/` is gitignored; commit HTML via `publish-notebooks` when ready.)

## Template

- `_templates/llm_advisor_eval.ipynb`: starter structure for new analysis notebooks.

## Local Execution

```bash
pip install -r requirements.txt papermill jupyter nbconvert
cd notebooks
papermill trade_journal.ipynb outputs/trade_journal.ipynb -p LOOKBACK_DAYS 30
jupyter nbconvert outputs/trade_journal.ipynb --to html --output-dir rendered/
```

## Notes

- Each notebook includes a papermill `parameters` cell.
- These are starter implementations and will be expanded with production-grade queries in subsequent Week-2 PRs.
