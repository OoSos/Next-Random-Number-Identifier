"""
Integration tests for the complete model pipeline.
These tests verify that the entire forecasting pipeline works together correctly.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.markov_chain import MarkovChain
from src.models.ensemble import EnhancedEnsemble


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-01-31')
    numbers = np.random.randint(1, 11, size=len(dates))
    return pd.DataFrame({'Date': dates, 'Number': numbers})


@pytest.mark.integration
def test_end_to_end_pipeline(sample_data, tmp_path):
    """Test the entire prediction pipeline from data loading to ensemble prediction."""
    # Setup
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_data.to_csv(data_dir / "test_data.csv", index=False)
    
    # Create data loader
    data_loader = DataLoader(str(data_dir))
    
    # Load and preprocess
    df = data_loader.load_csv("test_data.csv")
    df = data_loader.preprocess_data(df)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.transform(df)
    
    # Prepare data for models
    X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
    y = df_features['Number']
    
    # Feature selection
    feature_selector = FeatureSelector(n_features=5)  # Small number for testing
    feature_selector.fit(X, y)
    X_selected = feature_selector.transform(X)
    
    # Initialize models
    rf_model = RandomForestModel(n_estimators=10)  # Small number for testing
    xgb_model = XGBoostModel(n_estimators=10)  # Small number for testing
    markov_model = MarkovChain()
    
    # Train individual models
    rf_model.fit(X_selected, y)
    xgb_model.fit(X_selected, y)
    markov_model.fit(X_selected, y)
    
    # Train ensemble
    ensemble = EnhancedEnsemble(models=[rf_model, xgb_model, markov_model])
    ensemble.fit(X_selected, y)
    
    # Make predictions
    predictions = ensemble.predict(X_selected.head(5))
    
    # Assertions
    assert len(predictions) == 5
    assert all(1 <= pred <= 10 for pred in predictions)
    
    # Test feature importance
    feature_importance = ensemble.feature_importance_
    assert feature_importance is not None
    assert len(feature_importance) > 0


@pytest.mark.benchmark
def test_model_performance_benchmark(sample_data):
    """Benchmark the performance of different models."""
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.transform(sample_data)
    
    # Prepare data for models
    X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
    y = df_features['Number']
    
    # Initialize models
    models = {
        'RandomForest': RandomForestModel(n_estimators=50),
        'XGBoost': XGBoostModel(n_estimators=50),
        'MarkovChain': MarkovChain(),
        'Ensemble': EnhancedEnsemble()
    }
    
    for name, model in models.items():
        # Train model (measure time in a real benchmark)
        model.fit(X, y)
        
        # Make predictions (measure time in a real benchmark)
        _ = model.predict(X.head(10))
        
        # Get feature importance if available
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            assert feature_importance is not None