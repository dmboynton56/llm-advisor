import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.premarket.bias_features import FEATURE_NAMES, compute_bias_features
from src.premarket.bias_training import build_dataset, fit_and_evaluate, export_model


def frames():
    day = pd.Timestamp("2026-11-03 09:00", tz="America/New_York")
    daily_index = pd.bdate_range(end="2026-11-02", periods=60, tz=day.tz)
    daily = pd.DataFrame({"open": 100., "high": 103., "low": 98., "close": 101., "volume": 10000.}, index=daily_index)
    minute_index = pd.date_range("2026-11-03 04:00", "2026-11-03 10:00", freq="min", tz=day.tz)
    minute = pd.DataFrame({"open": 101., "high": 102., "low": 100., "close": 101.5, "volume": 10.}, index=minute_index)
    hourly = pd.DataFrame({"close": np.linspace(95, 102, 80)}, index=pd.date_range(end="2026-11-03 12:00", periods=80, freq="h", tz=day.tz))
    four_hour = pd.DataFrame({"close": np.linspace(95, 102, 80)}, index=pd.date_range(end="2026-11-03 12:00", periods=80, freq="4h", tz=day.tz))
    return day, daily, minute, hourly, four_hour


def test_features_exclude_future_daily_minute_and_unfinished_higher_bars():
    day, daily, minute, hourly, four_hour = frames()
    expected, error, meta = compute_bias_features("SPY", daily, minute, hourly, four_hour, FEATURE_NAMES, as_of=day)
    assert error is None
    minute.loc[minute.index >= day, :] = 1e9
    hourly.loc[hourly.index + pd.Timedelta(hours=1) > day, :] = 1e9
    four_hour.loc[four_hour.index + pd.Timedelta(hours=4) > day, :] = 1e9
    daily.loc[day.normalize(), :] = 1e9
    actual, error, _ = compute_bias_features("SPY", daily, minute, hourly, four_hour, FEATURE_NAMES, as_of=day)
    pd.testing.assert_frame_equal(expected, actual)
    assert error is None
    assert meta["feature_cutoff"] == day.isoformat()


def test_training_twenty_minute_features_match_live_one_minute_features():
    day, daily, minute, hourly, four_hour = frames()
    twenty = minute.resample("20min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    live, _, _ = compute_bias_features("SPY", daily, minute, hourly, four_hour, FEATURE_NAMES, as_of=day)
    training, _, _ = compute_bias_features("SPY", daily, twenty, hourly, four_hour, FEATURE_NAMES, as_of=day, bar_minutes=20)
    pd.testing.assert_frame_equal(live, training)


def test_stale_premarket_and_early_run_fail_closed():
    day, daily, minute, hourly, four_hour = frames()
    _, error, _ = compute_bias_features("SPY", daily, minute, hourly, four_hour, FEATURE_NAMES, as_of=day - pd.Timedelta(minutes=1))
    assert error["error"] == "before_bias_feature_cutoff"
    minute = minute.loc[:day - pd.Timedelta(hours=1)]
    _, error, _ = compute_bias_features("SPY", daily, minute, hourly, four_hour, FEATURE_NAMES, as_of=day)
    assert error["error"] == "stale_premarket_bars"


def dataset(informative):
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 800)
    result = pd.DataFrame(0., index=range(800), columns=FEATURE_NAMES)
    if informative:
        result["overnight_gap_pct"] = y
    result["date"] = pd.bdate_range("2020-01-01", periods=800).strftime("%Y-%m-%d")
    result["label"] = np.array(["bearish", "bullish", "choppy"])[y]
    result["move_pct"] = 0.
    result["label_band_pct"] = 1.
    return result


def test_chronological_validation_and_artifact_integrity(tmp_path, monkeypatch):
    from src.data_processing import daily_bias_computing as runtime
    data = dataset(True)
    model, encoder, report, predictions = fit_and_evaluate(data)
    assert report["status"] == "accepted"
    assert report["evaluation_train_end"] < report["holdout_start"]
    assert report["holdout_sessions"] == len(predictions) == 200
    data_path = tmp_path / "dataset.csv"
    data.to_csv(data_path, index=False)
    output = tmp_path / "models"
    export_model("TEST", model, encoder, report, output, data_path, {}, "iex")
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    loaded, _, names = runtime._load_model_and_encoder("TEST")
    np.testing.assert_allclose(loaded.predict_proba(data[names].head()), model.predict_proba(data[names].head()))
    (output / "TEST_feature_names.json").write_text("[]")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        runtime._load_model_and_encoder("TEST")


def test_non_predictive_model_is_not_promoted(tmp_path):
    model, encoder, report, _ = fit_and_evaluate(dataset(False))
    assert report["status"] == "rejected"
    with pytest.raises(ValueError, match="Refusing to promote"):
        export_model("TEST", model, encoder, report, tmp_path / "models", Path("unused"), {}, "iex")
    assert not (tmp_path / "models").exists()


def test_delayed_sip_requests_respect_free_account_boundary():
    from types import SimpleNamespace
    from src.premarket.bias_training_data import fetch_bias_inference_frames

    requests = []
    frame = pd.DataFrame({"open": [100.], "high": [101.], "low": [99.], "close": [100.], "volume": [10.]}, index=pd.MultiIndex.from_tuples(
        [("SPY", pd.Timestamp("2026-11-03 13:40", tz="UTC"))], names=["symbol", "timestamp"],
    ))
    def fetch(request):
        requests.append(request)
        return SimpleNamespace(df=frame)

    client = SimpleNamespace(get_stock_bars=fetch)
    now = pd.Timestamp("2026-11-03 09:20", tz="America/New_York")
    metadata = {"data_feed": "sip", "feature_cutoff_et": "09:00"}
    result = fetch_bias_inference_frames(client, "SPY", metadata, now=now)
    assert len(result) == 4 and len(requests) == 2
    # alpaca-py normalizes request datetimes to naive UTC.
    assert all(pd.Timestamp(r.end, tz="UTC") <= now - pd.Timedelta(minutes=15) for r in requests)
    assert all(r.feed.value == "sip" and r.adjustment.value == "split" for r in requests)
    with pytest.raises(ValueError, match="before historical feed is available"):
        fetch_bias_inference_frames(client, "SPY", metadata, now=now - pd.Timedelta(minutes=10))
    assert len(requests) == 2


def test_label_targets_daily_close_change_not_a_trade_entry_return(monkeypatch):
    from src.premarket import bias_training

    day, daily, _, hourly, four_hour = frames()
    daily.loc[day.normalize(), :] = 999.  # Today's full daily bar cannot be a feature.
    bars = pd.DataFrame({"close": [103., 102.]}, index=pd.DatetimeIndex([
        "2026-11-03 08:40", "2026-11-03 15:40",
    ], tz=day.tz))
    def features(*args, **kwargs):
        values = pd.DataFrame(0., index=[0], columns=FEATURE_NAMES)
        values["daily_atr14_pct"] = 1.
        return values, None, {"reference_price": 103.}

    monkeypatch.setattr(bias_training, "compute_bias_features", features)
    data, _ = build_dataset("SPY", {"daily": daily, "20min": bars, "hourly": hourly, "4hour": four_hour})
    assert data.iloc[0]["reference_price"] == 101.
    assert data.iloc[0]["close_price"] == 102.
    assert data.iloc[0]["label"] == "bullish"


def test_higher_timeframe_aggregation_stays_utc_aligned_across_dst():
    from src.premarket.bias_training_data import with_higher_timeframes

    index = pd.date_range("2026-10-30", "2026-11-03", freq="20min", tz="America/New_York")
    bars = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 10.}, index=index)
    aggregated = with_higher_timeframes({"20min": bars})
    assert all(hour % 4 == 0 for hour in aggregated["4hour"].index.tz_convert("UTC").hour)
    assert aggregated["4hour"].index.is_monotonic_increasing


@pytest.mark.parametrize("symbol", ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOG", "TSLA"])
def test_bundled_historical_models_are_loadable_and_predict_valid_probabilities(symbol, monkeypatch):
    from src.data_processing.daily_bias_computing import _load_model_and_encoder

    monkeypatch.delenv("ALLOW_PLACEHOLDER_BIAS", raising=False)
    model, encoder, names = _load_model_and_encoder(symbol)
    assert model.bias_metadata_["data_feed"] == "sip"
    assert names == list(FEATURE_NAMES)
    assert list(encoder.classes_) == ["bearish", "bullish", "choppy"]
    day, daily, minute, hourly, four_hour = frames()
    features, error, _ = compute_bias_features(symbol, daily, minute, hourly, four_hour, names, as_of=day)
    assert error is None
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (1, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.)


def test_placeholder_exporter_cannot_overwrite_a_historical_model(tmp_path, monkeypatch):
    from scripts import export_minimal_bias_models as exporter
    import sys

    provenance = tmp_path / "SPY_provenance.json"
    provenance.write_text(json.dumps({"source": "train_daily_bias_models.py"}))
    monkeypatch.setattr(exporter, "MODELS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["export_minimal_bias_models.py", "SPY"])
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        exporter.main()
    assert not (tmp_path / "SPY_daily_bias.pkl").exists()
