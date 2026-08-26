"""Test that placeholder bias models are properly gated."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_placeholder_bias_model_gate_blocks_by_default():
    """Placeholder models should be rejected unless ALLOW_PLACEHOLDER_BIAS=true."""
    # Make sure a placeholder model exists for testing
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.export_minimal_bias_models import main as export_main
    
    # Export SPY placeholder
    with patch("sys.argv", ["export_minimal_bias_models.py", "SPY"]):
        export_main()
    
    # Now try to load it - should fail by default
    from src.data_processing.daily_bias_computing import _load_model_and_encoder
    
    # Ensure the env var is NOT set
    if "ALLOW_PLACEHOLDER_BIAS" in os.environ:
        del os.environ["ALLOW_PLACEHOLDER_BIAS"]
    
    with pytest.raises(RuntimeError, match="placeholder.*CI/dev.*artifact"):
        _load_model_and_encoder("SPY")


def test_placeholder_bias_model_gate_allows_with_override():
    """ALLOW_PLACEHOLDER_BIAS=true should permit placeholder models."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.export_minimal_bias_models import main as export_main
    
    # Export SPY placeholder
    with patch("sys.argv", ["export_minimal_bias_models.py", "SPY"]):
        export_main()
    
    from src.data_processing.daily_bias_computing import _load_model_and_encoder
    
    # Set the override
    os.environ["ALLOW_PLACEHOLDER_BIAS"] = "true"
    
    try:
        model, le, feature_names = _load_model_and_encoder("SPY")
        assert model is not None
        assert le is not None
        assert isinstance(feature_names, list)
    finally:
        if "ALLOW_PLACEHOLDER_BIAS" in os.environ:
            del os.environ["ALLOW_PLACEHOLDER_BIAS"]


def test_production_model_without_provenance_loads_normally():
    """A model without provenance metadata should load normally (backward compat)."""
    import sys
    import pickle
    import json
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    sys.path.insert(0, str(PROJECT_ROOT))
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a model without provenance
    symbol = "TESTPROD"
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    le = LabelEncoder()
    
    X = np.random.rand(10, 5)
    y = le.fit_transform(["bullish", "bearish", "choppy"] * 3 + ["bullish"])
    clf.fit(X, y)
    
    pkl_path = models_dir / f"{symbol}_daily_bias.pkl"
    enc_path = models_dir / f"{symbol}_label_encoder.pkl"
    feat_path = models_dir / f"{symbol}_feature_names.json"
    prov_path = models_dir / f"{symbol}_provenance.json"
    
    # Clean up any existing provenance
    if prov_path.exists():
        prov_path.unlink()
    
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    with open(enc_path, "wb") as f:
        pickle.dump(le, f)
    feat_path.write_text(json.dumps(["f1", "f2", "f3", "f4", "f5"]))
    
    from src.data_processing.daily_bias_computing import _load_model_and_encoder
    
    # Should load without error (no provenance = assume production)
    model, encoder, features = _load_model_and_encoder(symbol)
    assert model is not None
    assert encoder is not None
    
    # Cleanup
    pkl_path.unlink()
    enc_path.unlink()
    feat_path.unlink()
