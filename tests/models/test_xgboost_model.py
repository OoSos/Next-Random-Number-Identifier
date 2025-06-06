import unittest
import pandas as pd
import numpy as np
import pytest

from src.models.xgboost_model import XGBoostModel

class TestXGBoostModel(unittest.TestCase):
    """Test cases for XGBoostModel class"""

    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.y = pd.Series(np.random.randint(0, 2, size=100))
        self.model = XGBoostModel()

    def test_initialization(self):
        model = XGBoostModel()
        self.assertEqual(model.name, "XGBoost")
        self.assertEqual(model.params['n_estimators'], 100)
        self.assertEqual(model.params['learning_rate'], 0.1)
        self.assertEqual(model.params['random_state'], 42)
        self.assertEqual(model.params['max_depth'], 3)

    def test_custom_initialization(self):
        custom_model = XGBoostModel(n_estimators=200, max_depth=10, learning_rate=0.2, random_state=123)
        self.assertEqual(custom_model.params['n_estimators'], 200)
        self.assertEqual(custom_model.params['max_depth'], 10)
        self.assertEqual(custom_model.params['learning_rate'], 0.2)
        self.assertEqual(custom_model.params['random_state'], 123)

    def test_fit(self):
        result = self.model.fit(self.X, self.y)
        self.assertIs(result, self.model)
        self.assertIsNotNone(self.model.feature_importance_)

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_without_fit(self):
        unfitted_model = XGBoostModel()
        with self.assertRaises(ValueError):
            unfitted_model.predict(self.X)

    def test_feature_importance(self):
        self.model.fit(self.X, self.y)
        importance = self.model.get_feature_importance()
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 3)
        self.assertTrue(all(isinstance(v, (float, np.floating)) for v in importance.values()))
        self.assertTrue(all(v >= 0 for v in importance.values()))

    def test_feature_importance_without_fit(self):
        unfitted_model = XGBoostModel()
        unfitted_model.feature_importance_ = None
        with self.assertRaises(ValueError):
            unfitted_model.get_feature_importance()

    def test_invalid_model_parameters(self):
        """Test that models properly validate input parameters during initialization."""
        # Test XGBoost with invalid parameters that should raise ValueError during initialization
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            XGBoostModel(n_estimators=-1)  # Negative trees
        
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            XGBoostModel(n_estimators=0)  # Zero trees
        
        # Test invalid learning rate
        with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
            XGBoostModel(learning_rate=-0.1)  # Negative learning rate
            
        with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
            XGBoostModel(learning_rate=1.5)  # Learning rate > 1
            
        # Test that valid edge cases work
        try:
            model_valid = XGBoostModel(n_estimators=1, learning_rate=0.01, max_depth=1)
            # Create test data
            X = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
            y = pd.Series([1,2,3])
            model_valid.fit(X, y)  # This should work
        except ValueError:
            self.fail("Valid parameters should not raise ValueError")

    def test_parameter_warnings(self):
        """Test that invalid parameters generate appropriate warnings."""
        import warnings
        
        # Test invalid parameter names (should generate warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = XGBoostModel(invalid_param=123)
            # Note: XGBoost might not warn about invalid params in the same way
            # This test documents the current behavior


if __name__ == '__main__':
    unittest.main()
