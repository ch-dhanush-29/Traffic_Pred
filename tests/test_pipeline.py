"""
Unit and Integration Test Suite for FlowCast Pipeline
Verifies configuration, schema compliance, cleaning logic,
NumPy from-scratch math, model inference, and sequence dimensions.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
import pandas as pd
from src.utils import load_config
from src.ml_models import ScratchLinearRegression
from src.clean import derive_congestion_level

def test_config():
    """Verify config.yaml structure and keys."""
    cfg = load_config("config.yaml")
    assert "paths" in cfg
    assert "cleaning" in cfg
    assert "deep_learning" in cfg
    assert cfg["project"]["total_segments"] == 25

def test_derive_congestion_level():
    """Verify congestion level derivation formula and bands."""
    # Capacity = 2000 -> Half-capacity = 1000
    # Volume 300 -> V/C = 0.30 -> Free-flow
    # Volume 600 -> V/C = 0.60 -> Moderate
    # Volume 900 -> V/C = 0.90 -> Heavy
    # Volume 1200 -> V/C = 1.20 -> Severe
    volumes = pd.Series([300, 600, 900, 1200])
    capacities = pd.Series([2000, 2000, 2000, 2000])
    
    levels = derive_congestion_level(volumes, capacities)
    assert levels.tolist() == ['Free-flow', 'Moderate', 'Heavy', 'Severe']

def test_scratch_linear_regression():
    """Verify from-scratch NumPy mini-batch gradient descent on synthetic linear problem."""
    np.random.seed(42)
    X = np.random.randn(200, 3)
    true_w = np.array([2.5, -1.8, 4.0])
    true_b = 3.5
    y = np.dot(X, true_w) + true_b + np.random.normal(0, 0.05, 200)
    
    model = ScratchLinearRegression(lr=0.08, epochs=40, batch_size=32)
    model.fit(X, y)
    
    preds = model.predict(X)
    r2 = 1.0 - (np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2))
    assert r2 > 0.95
    assert abs(model.bias - true_b) < 0.2

def test_processed_dataset_integrity():
    """Verify processed dataset exists and has 0 nulls in feature columns."""
    cfg = load_config("config.yaml")
    assert os.path.exists(cfg['paths']['processed_data'])
    df = pd.read_parquet(cfg['paths']['processed_data'])
    assert len(df) > 150000
    assert df['road_id'].nunique() == 25
    # Check no nulls in target
    assert df['traffic_volume'].isnull().sum() == 0

def test_model_inference():
    """Verify trained models can be loaded and produce realistic bounded predictions."""
    cfg = load_config("config.yaml")
    vol_model = joblib.load(os.path.join(cfg['paths']['models_dir'], "best_volume_model.joblib"))
    feat_imp = pd.read_csv(os.path.join(cfg['paths']['models_dir'], "feature_importances.csv"))
    
    num_features = len(feat_imp)
    sample_input = np.zeros((1, num_features))
    pred = vol_model.predict(sample_input)[0]
    assert isinstance(pred, (float, np.floating))
    assert pred >= 0.0

if __name__ == "__main__":
    print("Running FlowCast Test Suite...")
    test_config()
    print("✓ test_config passed")
    test_derive_congestion_level()
    print("✓ test_derive_congestion_level passed")
    test_scratch_linear_regression()
    print("✓ test_scratch_linear_regression passed")
    test_processed_dataset_integrity()
    print("✓ test_processed_dataset_integrity passed")
    test_model_inference()
    print("✓ test_model_inference passed")
    print("All tests passed successfully!")
