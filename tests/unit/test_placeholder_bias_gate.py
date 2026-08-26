"""Test that placeholder bias models are properly gated."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch
import pickle
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_placeholder_bias_model_gate_blocks_by_default(tmp_path):
    """Placeholder models should be rejected unless ALLOW_PLACEHOLDER_BIAS=true."""
    import sys
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Manually create a placeholder model in tmp_path
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    symbol = "SPY"
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    le = LabelEncoder()
    X = np.random.rand(10, 22)
    y = le.fit_transform(["bullish", "bearish", "choppy"] * 3 + ["bullish"])
    clf.fit(X, y)
    
    # Write model with placeholder provenance
    provenance = {
        "source": "export_minimal_bias_models.py",
        "purpose": "CI/dev placeholder prior",
        "warning": "This is a synthetic prior.",
    }
    
    with open(models_dir / f"{symbol}_daily_bias.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(models_dir / f"{symbol}_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    (models_dir / f"{symbol}_feature_names.json").write_text(json.dumps(["f1"] * 22))
    (models_dir / f"{symbol}_provenance.json").write_text(json.dumps(provenance))
    
    from src.data_processing.daily_bias_computing import _load_model_and_encoder
    
    # Ensure the env var is NOT set
    if "ALLOW_PLACEHOLDER_BIAS" in os.environ:
        del os.environ["ALLOW_PLACEHOLDER_BIAS"]
    
    # Patch PROJECT_ROOT to point to tmp_path
    with patch("src.data_processing.daily_bias_computing.PROJECT_ROOT", tmp_path):
        with pytest.raises(RuntimeError, match="placeholder.*CI/dev.*artifact"):
            _load_model_and_encoder("SPY")


def test_placeholder_bias_model_gate_allows_with_override(tmp_path):
    """ALLOW_PLACEHOLDER_BIAS=true should permit placeholder models."""
    import sys
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Manually create a placeholder model in tmp_path
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    symbol = "SPY"
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    le = LabelEncoder()
    X = np.random.rand(10, 22)
    y = le.fit_transform(["bullish", "bearish", "choppy"] * 3 + ["bullish"])
    clf.fit(X, y)
    
    provenance = {
        "source": "export_minimal_bias_models.py",
        "purpose": "CI/dev placeholder prior",
    }
    
    with open(models_dir / f"{symbol}_daily_bias.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(models_dir / f"{symbol}_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    (models_dir / f"{symbol}_feature_names.json").write_text(json.dumps(["f1"] * 22))
    (models_dir / f"{symbol}_provenance.json").write_text(json.dumps(provenance))
    
    from src.data_processing.daily_bias_computing import _load_model_and_encoder
    
    # Set the override
    os.environ["ALLOW_PLACEHOLDER_BIAS"] = "true"
    
    try:
        with patch("src.data_processing.daily_bias_computing.PROJECT_ROOT", tmp_path):
            model, le_loaded, feature_names = _load_model_and_encoder("SPY")
            assert model is not None
            assert le_loaded is not None
            assert isinstance(feature_names, list)
    finally:
        if "ALLOW_PLACEHOLDER_BIAS" in os.environ:
            del os.environ["ALLOW_PLACEHOLDER_BIAS"]


def test_production_model_without_provenance_loads_normally(tmp_path):
    """A model without provenance metadata should load normally (backward compat)."""
    import sys
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    sys.path.insert(0, str(PROJECT_ROOT))
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a model without provenance
    symbol = "TESTPROD"
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    le = LabelEncoder()
    
    X = np.random.rand(10, 5)
    y = le.fit_transform(["bullish", "bearish", "choppy"] * 3 + ["bullish"])
    clf.fit(X, y)
    
    with open(models_dir / f"{symbol}_daily_bias.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(models_dir / f"{symbol}_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    (models_dir / f"{symbol}_feature_names.json").write_text(json.dumps(["f1", "f2", "f3", "f4", "f5"]))
    
    from src.data_processing.daily_bias_computing import _load_model_and_encoder
    
    # Should load without error (no provenance = assume production)
    with patch("src.data_processing.daily_bias_computing.PROJECT_ROOT", tmp_path):
        model, encoder, features = _load_model_and_encoder(symbol)
        assert model is not None
        assert encoder is not None
