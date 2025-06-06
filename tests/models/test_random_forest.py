import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
import pytest

from src.models.random_forest import RandomForestModel


class TestRandomForestModel(unittest.TestCase):
    """Test cases for RandomForestModel class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample data for testing
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.y = pd.Series(np.random.rand(100))
        
        # Create a default model instance
        self.model = RandomForestModel()
        
    def test_initialization(self):
        """Test that the model initializes with correct default parameters"""
        model = RandomForestModel()
        self.assertEqual(model.name, "RandomForest")
        self.assertEqual(model.params['n_estimators'], 100)
        self.assertEqual(model.params['random_state'], 42)
        self.assertEqual(model.params['min_samples_leaf'], 1)
        self.assertIsNone(model.params.get('max_depth'))
        
    def test_custom_initialization(self):
        """Test that the model initializes with custom parameters"""
        custom_model = RandomForestModel(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=123)
        self.assertEqual(custom_model.params['n_estimators'], 200)
        self.assertEqual(custom_model.params['max_depth'], 10)
        self.assertEqual(custom_model.params['min_samples_leaf'], 2)
        self.assertEqual(custom_model.params['random_state'], 123)
        
    def test_fit(self):
        """Test the fit method"""
        # Fit the model
        result = self.model.fit(self.X, self.y)
        
        # Check that fit returns self
        self.assertIs(result, self.model)
        
        # Check that feature importance is computed
        self.assertIsNotNone(self.model.feature_importance_)
        
    def test_predict(self):
        """Test the predict method"""
        # Fit the model first
        self.model.fit(self.X, self.y)
        
        # Test prediction on training data
        predictions = self.model.predict(self.X)
        
        # Check prediction shape and type
        self.assertEqual(len(predictions), len(self.X))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_predict_without_fit(self):
        """Test prediction throws error when model isn't fitted"""
        # Create a new model without fitting
        unfitted_model = RandomForestModel()
        
        # Test that predicting with unfitted model raises an error
        with self.assertRaises(ValueError):
            unfitted_model.predict(self.X)
            
    def test_feature_importance(self):
        """Test the feature importance functionality"""
        # Fit the model
        self.model.fit(self.X, self.y)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Check structure and values
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 3)  # 3 features
        self.assertTrue(all(isinstance(v, float) for v in importance.values()))
        self.assertTrue(all(v >= 0 for v in importance.values()))  # Importance values should be non-negative
        
    def test_feature_importance_without_fit(self):
        """Test feature importance throws error when model isn't fitted"""
        # Create a new model without fitting
        unfitted_model = RandomForestModel()
        
        # Remove the feature_importance_ attribute to simulate unfitted state
        unfitted_model.feature_importance_ = None
        
        # Test that getting feature importance with unfitted model raises an error
        with self.assertRaises(ValueError):
            unfitted_model.get_feature_importance()
            
    def test_get_model_info(self):
        """Test the model info functionality"""
        # Fit the model
        self.model.fit(self.X, self.y)
        
        # Get model info
        info = self.model.get_model_info()
        
        # Check structure
        self.assertIsInstance(info, dict)
        self.assertEqual(info['name'], "RandomForest")
        self.assertEqual(info['n_trees'], 100)
        self.assertEqual(info['n_features'], 3)  # 3 features
        self.assertIsNone(info['max_depth'])
        self.assertIsInstance(info['feature_importance'], dict)
        
    def test_invalid_model_parameters(self):
        """Test that models properly validate input parameters during fit."""
        # Create test data
        X = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
        y = pd.Series([1,2,3])
        
        # Test RandomForest with invalid parameters that should raise ValueError during fit
        model_invalid = RandomForestModel(n_estimators=-1)  # Negative trees
        with pytest.raises(ValueError, match="must be an int in the range"):
            model_invalid.fit(X, y)
        
        model_zero = RandomForestModel(n_estimators=0)  # Zero trees
        with pytest.raises(ValueError, match="must be an int in the range"):
            model_zero.fit(X, y)
        
        # Test that invalid types get converted during initialization
        model1 = RandomForestModel(n_estimators=50.5)  # Float gets converted to int
        assert model1.params['n_estimators'] == 50
        
    def test_parameter_warnings(self):
        """Test that invalid parameters generate appropriate warnings."""
        import warnings
        
        # Test invalid parameter names (should generate warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = RandomForestModel(invalid_param=123)
            assert len(w) > 0
            assert "not valid for RandomForestRegressor" in str(w[0].message)


if __name__ == '__main__':
    unittest.main()
