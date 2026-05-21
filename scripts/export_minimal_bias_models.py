#!/usr/bin/env python3
"""Train minimal sklearn bias models for CI / local dev when production pickles are unavailable.

Produces ``models/{SYMBOL}_daily_bias.pkl`` and ``models/{SYMBOL}_label_encoder.pkl``
using the default feature list from ``daily_bias_computing._load_model_and_encoder``.

This is a **placeholder** prior (mostly ``choppy``); replace with real trained artifacts for production.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"

DEFAULT_FEATURES = [
    "overnight_gap_pct",
    "premarket_range_pct",
    "premarket_vol",
    "premarket_vol_vs_prev5d",
    "premarket_sweep_prev_high",
    "premarket_sweep_prev_low",
    "premarket_close_vs_prev_close_pct",
    "premarket_return_pct",
    "prev_close",
    "prev_day_range_pct",
    "prev_day_body_pct",
    "prev_day_bull",
    "prev_day_swept_prior_high",
    "prev_day_swept_prior_low",
    "open_pos_in_prev_range",
    "open_to_prev_high_pct_rng",
    "open_to_prev_low_pct_rng",
    "daily_atr14_pct",
    "h1_close_vs_sma20_pct",
    "h4_close_vs_sma20_pct",
    "h1_mom_5bars_pct",
    "h4_mom_3bars_pct",
]


def main() -> None:
    symbols = [s.strip().upper() for s in sys.argv[1:]] or ["SPY", "QQQ", "IWM"]
    MODELS.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_feat = len(DEFAULT_FEATURES)
    X = rng.normal(size=(400, n_feat))
    y = rng.choice(["bullish", "bearish", "choppy"], size=400, p=[0.25, 0.25, 0.5])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=30,
        max_depth=6,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(X, y_enc)

    for sym in symbols:
        pkl = MODELS / f"{sym}_daily_bias.pkl"
        enc = MODELS / f"{sym}_label_encoder.pkl"
        feat_path = MODELS / f"{sym}_feature_names.json"
        with open(pkl, "wb") as f:
            pickle.dump(clf, f)
        with open(enc, "wb") as f:
            pickle.dump(le, f)
        feat_path.write_text(json.dumps(DEFAULT_FEATURES), encoding="utf-8")
        print(f"Wrote {pkl.name}, {enc.name}, {feat_path.name}")


if __name__ == "__main__":
    main()
