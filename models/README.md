# Daily bias model artifacts

Premarket calls [`src/data_processing/daily_bias_computing.py`](../src/data_processing/daily_bias_computing.py), which loads per symbol:

| File | Purpose |
|------|---------|
| `{SYMBOL}_daily_bias.pkl` | `sklearn` classifier |
| `{SYMBOL}_label_encoder.pkl` | `LabelEncoder` for `bullish` / `bearish` / `choppy` |
| `{SYMBOL}_feature_names.json` | Optional; defaults to the built-in 20-feature list if missing |

## CI / dev placeholder

From repo root:

```bash
python scripts/export_minimal_bias_models.py SPY QQQ IWM
```

Replace these with **production** models trained on your feature pipeline before relying on ML bias for trading.

## GitHub Actions

[`.github/workflows/premarket.yml`](../.github/workflows/premarket.yml) verifies `SPY`, `QQQ`, and `IWM` artifacts exist before the pipeline runs.
