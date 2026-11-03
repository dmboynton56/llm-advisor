"""Chronological training and honest validation for the 09:00 ET daily prior."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sklearn

from src.premarket.bias_features import (
    compute_bias_features, FEATURE_NAMES, FEATURE_VERSION,
)

FEATURES = list(FEATURE_NAMES)
LABEL_ATR_FRACTION = 0.35


def build_dataset(symbol: str, frames: dict) -> tuple[pd.DataFrame, dict]:
    daily = frames["daily"]
    intraday = frames["20min"]
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=str(daily.index[0].date()), end_date=str(daily.index[-1].date()),
    )
    groups = dict(tuple(intraday.groupby(intraday.index.date)))
    rows, skipped = [], Counter()
    for session, trading_times in schedule.iterrows():
        day = session.date()
        bars = groups.get(day)
        if bars is None:
            skipped["missing_intraday_session"] += 1
            continue
        cutoff = pd.Timestamp(f"{day} 09:00", tz="America/New_York")
        prior = daily[daily.index.date < day].tail(60)
        h1 = frames["hourly"].loc[:cutoff].tail(40)
        h4 = frames["4hour"].loc[:cutoff].tail(40)
        features, error, meta = compute_bias_features(
            symbol, prior, bars, h1, h4, FEATURES, as_of=cutoff, bar_minutes=20,
        )
        if error:
            skipped[error["error"]] += 1
            continue
        close_time = trading_times["market_close"]
        completed = bars[bars.index + pd.Timedelta(minutes=20) <= close_time]
        if completed.empty or completed.index[-1] + pd.Timedelta(minutes=20) < close_time:
            skipped["missing_closing_bar"] += 1
            continue
        reference = float(prior.iloc[-1]["close"])
        close = float(completed.iloc[-1]["close"])
        move = (close / reference - 1) * 100
        band = float(features.iloc[0]["daily_atr14_pct"]) * LABEL_ATR_FRACTION
        label = "bullish" if move > band else "bearish" if move < -band else "choppy"
        rows.append({"date": str(day), **features.iloc[0].to_dict(), "label": label,
                     "reference_price": reference, "close_price": close,
                     "move_pct": move, "label_band_pct": band})
    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError(f"No complete training sessions for {symbol}: {dict(skipped)}")
    return dataset, {
        "skipped_sessions": dict(skipped), "usable_sessions": len(dataset),
        "source_bar_counts": {name: len(frame) for name, frame in frames.items()},
        "history_start": str(daily.index[0].date()), "history_end": str(daily.index[-1].date()),
        "training_data_checks": "unique sorted timestamps, complete cutoff/closing bars, finite features",
    }


def _metrics(y, probabilities) -> dict:
    prediction = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "class_counts": {str(k): int(v) for k, v in Counter(y).items()},
    }


def fit_and_evaluate(dataset: pd.DataFrame, holdout_size: int | None = None) -> tuple:
    """Select by expanding-window CV, then evaluate ONCE on the final holdout.

    Promotion requires beating a train-frequency baseline on holdout log loss,
    better-than-chance balanced accuracy, and sufficient coverage in every class.
    Refit on all labeled sessions only after the fixed promotion gate passes.
    """
    dates = pd.to_datetime(dataset["date"])
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Training sessions must be unique and chronological")
    if not np.isfinite(dataset[FEATURES].to_numpy()).all():
        raise ValueError("Training features must be finite")
    holdout_size = holdout_size or min(252, len(dataset) // 4)
    if holdout_size < 126 or len(dataset) - holdout_size < 400:
        raise ValueError("Need at least 400 training and 126 holdout sessions")
    X = dataset[FEATURES]
    encoder = LabelEncoder().fit(["bearish", "bullish", "choppy"])
    y = encoder.transform(dataset["label"])
    split = len(dataset) - holdout_size
    train_X, test_X, train_y, test_y = X.iloc[:split], X.iloc[split:], y[:split], y[split:]
    if any(min(Counter(part).get(c, 0) for c in range(3)) < 20 for part in [train_y, test_y]):
        raise ValueError("Insufficient train/holdout coverage for all three classes")
    candidates = {
        "logistic_c001": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)),
        "logistic_c01": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "forest_depth3": RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=40, random_state=42, n_jobs=2),
        "forest_depth5": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=40, random_state=42, n_jobs=2),
    }
    scores = {}
    for name, candidate in candidates.items():
        losses = []
        for fit_indices, val_indices in TimeSeriesSplit(n_splits=3, gap=1).split(train_X):
            fitted = clone(candidate).fit(train_X.iloc[fit_indices], train_y[fit_indices])
            losses.append(float(log_loss(train_y[val_indices], fitted.predict_proba(train_X.iloc[val_indices]), labels=[0, 1, 2])))
        scores[name] = losses
    selected = min(scores, key=lambda name: np.mean(scores[name]))
    model = clone(candidates[selected]).fit(train_X, train_y)
    probability = model.predict_proba(test_X)
    baseline = DummyClassifier(strategy="prior").fit(train_X, train_y)
    model_metrics = _metrics(test_y, probability)
    baseline_metrics = _metrics(test_y, baseline.predict_proba(test_X))
    accepted = (model_metrics["log_loss"] < baseline_metrics["log_loss"]
                and model_metrics["balanced_accuracy"] > 1 / 3)
    report = {
        "status": "accepted" if accepted else "rejected",
        "promotion_rule": "holdout log_loss < train-prior baseline AND balanced_accuracy > 1/3; min 20/class",
        "selected_model": selected, "cv_log_losses": scores,
        "training_start": dataset.iloc[0]["date"], "evaluation_train_end": dataset.iloc[split - 1]["date"],
        "holdout_start": dataset.iloc[split]["date"], "holdout_end": dataset.iloc[-1]["date"],
        "train_sessions": split, "holdout_sessions": holdout_size,
        "training_class_counts": {str(k): int(v) for k, v in Counter(train_y).items()},
        "baseline_probabilities": baseline.class_prior_.tolist(),
        "model": model_metrics, "baseline": baseline_metrics,
        "final_refit_end": dataset.iloc[-1]["date"] if accepted else None,
    }
    predictions = dataset.iloc[split:][["date", "label", "move_pct", "label_band_pct"]].copy()
    for i, label in enumerate(encoder.classes_):
        predictions[f"probability_{label}"] = probability[:, i]
    if accepted:
        model = clone(candidates[selected]).fit(X, y)
    return model, encoder, report, predictions


def export_model(symbol: str, model, encoder, report: dict, output: Path,
                 dataset_path: Path, data_quality: dict, feed: str) -> None:
    """Publish validated artifacts only. Provenance is written last."""
    if report["status"] != "accepted":
        raise ValueError(f"Refusing to promote rejected model: {symbol}")
    model.bias_metadata_ = {"data_feed": feed, "feature_cutoff_et": "09:00", "feature_version": FEATURE_VERSION}
    payloads = {
        f"{symbol}_daily_bias.pkl": pickle.dumps(model, protocol=4),
        f"{symbol}_label_encoder.pkl": pickle.dumps(encoder, protocol=4),
        f"{symbol}_feature_names.json": json.dumps(FEATURES).encode(),
    }
    provenance = {
        "source": "train_daily_bias_models.py", "purpose": "historically trained paper-trading prior",
        "feature_version": FEATURE_VERSION, "feature_cutoff_et": "09:00",
        "data_feed": feed, "adjustment": "split", "sklearn_version": sklearn.__version__,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "artifact_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
        "label": "NYSE session close versus previous Alpaca daily-bar close; +/- 0.35 prior daily ATR; otherwise choppy",
        "data_quality": data_quality, "validation": report,
        "class_order": encoder.classes_.tolist(),
        "limitations": "Single historical holdout after chronological CV; not evidence of profitable option trades or calibrated live confidence.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        temporary = output / (name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output / name)
    path = output / f"{symbol}_provenance.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    temporary.replace(path)
