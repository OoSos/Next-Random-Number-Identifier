"""
Unit tests for model components.
Consolidates all model-specific tests into a single suite.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

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

class TestModels(unittest.TestCase):
    """Test cases for individual model components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2020-01-31')
        numbers = np.random.randint(1, 11, size=len(dates))
        self.df = pd.DataFrame({'Date': dates, 'Number': numbers})
        
        # Prepare features
        self.X = pd.DataFrame({
            'Year': self.df['Date'].dt.year,
            'Month': self.df['Date'].dt.month,
            'Day': self.df['Date'].dt.day,
            'DayOfWeek': self.df['Date'].dt.dayofweek
        })
        self.y = self.df['Number']

    def test_random_forest_model(self):
        """Test RandomForest model functionality."""
        model = RandomForestModel(n_estimators=10)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X.head(5))
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(1 <= pred <= 10 for pred in predictions))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), len(self.X.columns))

    def test_xgboost_model(self):
        """Test XGBoost model functionality."""
        model = XGBoostModel(n_estimators=10)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X.head(5))
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(1 <= pred <= 10 for pred in predictions))

    def test_markov_chain_model(self):
        """Test MarkovChain model functionality."""
        model = MarkovChain(order=1)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X.head(5))
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(1 <= pred <= 10 for pred in predictions))

    def test_enhanced_ensemble(self):
        """Test EnhancedEnsemble functionality."""
        rf_model = RandomForestModel(n_estimators=10)
        xgb_model = XGBoostModel(n_estimators=10)
        markov_model = MarkovChain(order=1)
        
        ensemble = EnhancedEnsemble(models=[rf_model, xgb_model, markov_model])
        ensemble.fit(self.X, self.y)
        predictions = ensemble.predict(self.X.head(5))
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(1 <= pred <= 10 for pred in predictions))
        
        # Test feature importance
        importance = ensemble.feature_importance_
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), len(self.X.columns))

if __name__ == '__main__':
    unittest.main()