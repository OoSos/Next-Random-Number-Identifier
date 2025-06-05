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
        self.assertEqual(len(self.model.feature_importance_), 3)
        for feature in self.X.columns:
            self.assertIn(feature, self.model.feature_importance_)

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_without_fit(self):
        unfitted_model = XGBoostModel()
        with self.assertRaises(Exception):
            unfitted_model.predict(self.X)

    def test_feature_importance(self):
        self.model.fit(self.X, self.y)
        importance = self.model.get_feature_importance()
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 3)
        # Accept float or numpy.float32/float64
        self.assertTrue(all(isinstance(v, (float, np.floating)) for v in importance.values()))
        self.assertTrue(all(v >= 0 for v in importance.values()))

    def test_feature_importance_without_fit(self):
        unfitted_model = XGBoostModel()
        unfitted_model.feature_importance_ = None
        with self.assertRaises(ValueError):
            unfitted_model.get_feature_importance()

    def test_invalid_model_parameters(self):
        with pytest.raises(ValueError):
            XGBoostModel(learning_rate=2.0)  # Invalid learning rate
        with pytest.raises(ValueError):
            XGBoostModel(n_estimators=-10)  # Invalid n_estimators

if __name__ == '__main__':
    unittest.main()
