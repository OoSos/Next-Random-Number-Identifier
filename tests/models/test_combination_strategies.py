import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

from src.models.ensemble import EnhancedEnsemble
from src.models.base_model import BaseModel

class MockModel(BaseModel):
    """Mock model for testing confidence-weighted strategies."""
    def __init__(self, predictions, confidences):
        super().__init__(name="MockModel")
        self.predictions = predictions
        self.confidences = confidences
        self.feature_importance_ = {"feature1": 0.5, "feature2": 0.3, "feature3": 0.2}
        self.fitted = False
    
    def fit(self, X, y):
        self.fitted = True
        return self
        
    def predict(self, X):
        return self.predictions
        
    def estimate_confidence(self, X):
        return self.confidences
        
    def get_feature_importance(self):
        return self.feature_importance_


class TestCombinationStrategies(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        X, _ = make_regression(n_samples=5, n_features=3, random_state=42)
        self.X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(3)])
        
        # Create three models with different prediction patterns and confidences
        self.pred1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.pred2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        self.pred3 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        
        # Each model has different confidence patterns
        self.conf1 = np.array([0.9, 0.8, 0.5, 0.3, 0.2])  # High confidence for first samples
        self.conf2 = np.array([0.2, 0.3, 0.9, 0.8, 0.4])  # High confidence for middle samples
        self.conf3 = np.array([0.1, 0.2, 0.3, 0.7, 0.9])  # High confidence for last samples
        
        # Create mock models
        self.models = [
            MockModel(self.pred1, self.conf1),
            MockModel(self.pred2, self.conf2),
            MockModel(self.pred3, self.conf3)
        ]
        
    def test_weighted_average(self):
        """Test weighted average prediction strategy."""
        ensemble = EnhancedEnsemble(
            models=self.models, 
            weights=np.array([0.5, 0.3, 0.2]),
            combination_method='weighted_average'        )
        
        predictions = ensemble.predict(self.X)
        
        # Expected: weighted average of predictions with fixed weights
        expected = 0.5 * self.pred1 + 0.3 * self.pred2 + 0.2 * self.pred3
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_confidence_weighted(self):
        """Test confidence-weighted prediction strategy."""
        ensemble = EnhancedEnsemble(
            models=self.models,
            combination_method='confidence_weighted'
        )
        
        predictions = ensemble.dynamic_confidence_weighting(self.X)
        
        # For sample 0, model1 has highest confidence (0.9) and value 1.0
        # The weighted prediction should be closer to model1's prediction than simple average
        simple_avg_0 = (self.pred1[0] + self.pred2[0] + self.pred3[0]) / 3  # 2.0
        model1_pred_0 = self.pred1[0]  # 1.0
        # Weighted prediction should be between model1's prediction and simple average
        self.assertLess(predictions[0], simple_avg_0)
        self.assertGreater(predictions[0], model1_pred_0)
        
        # For sample 2, model2 has highest confidence (0.9) and value 4.0
        simple_avg_2 = (self.pred1[2] + self.pred2[2] + self.pred3[2]) / 3  # 4.0
        model2_pred_2 = self.pred2[2]  # 4.0
        # Since model2 has highest confidence and its value equals the average, 
        # weighted prediction should be close to the average
        self.assertAlmostEqual(predictions[2], model2_pred_2, places=1)
        
    def test_variance_weighted(self):
        """Test variance-weighted prediction strategy."""
        ensemble = EnhancedEnsemble(
            models=self.models,
            combination_method='variance_weighted'
        )
        
        predictions = ensemble.variance_weighted_combination(self.X)
        
        # For sample 0, model1 should have highest weight due to lowest variance (highest confidence)
        self.assertGreater(predictions[0], (self.pred2[0] + self.pred3[0]) / 2)
        
        # For sample 2, model2 should have highest weight due to lowest variance
        self.assertGreater(predictions[2], (self.pred1[2] + self.pred3[2]) / 2)
        
        # For sample 4, model3 should have highest weight due to lowest variance
        self.assertGreater(predictions[4], (self.pred1[4] + self.pred2[4]) / 2)
        
    def test_ensemble_prediction_method_selection(self):
        """Test that the ensemble uses the correct prediction method based on configuration."""
        # Test with confidence_weighted method
        ensemble1 = EnhancedEnsemble(
            models=self.models,
            combination_method='confidence_weighted'
        )
        
        pred1 = ensemble1.predict(self.X)
        pred1_direct = ensemble1.dynamic_confidence_weighting(self.X)
        np.testing.assert_array_almost_equal(pred1, pred1_direct)
        
        # Test with variance_weighted method
        ensemble2 = EnhancedEnsemble(
            models=self.models,
            combination_method='variance_weighted'
        )
        
        pred2 = ensemble2.predict(self.X)
        pred2_direct = ensemble2.variance_weighted_combination(self.X)
        np.testing.assert_array_almost_equal(pred2, pred2_direct)
        
    def test_confidence_estimation(self):
        """Test confidence estimation for different combination methods."""
        # Test with default method (weighted_average)
        ensemble1 = EnhancedEnsemble(
            models=self.models,
            weights=np.array([0.5, 0.3, 0.2]),
            combination_method='weighted_average'
        )
        
        conf1 = ensemble1.estimate_confidence(self.X)
        expected1 = 0.5 * self.conf1 + 0.3 * self.conf2 + 0.2 * self.conf3
        np.testing.assert_array_almost_equal(conf1, expected1)
        
        # Test with confidence_weighted method (should return max confidence)
        ensemble2 = EnhancedEnsemble(
            models=self.models,
            combination_method='confidence_weighted'
        )
        
        conf2 = ensemble2.estimate_confidence(self.X)
        expected2 = np.maximum.reduce([self.conf1, self.conf2, self.conf3])
        np.testing.assert_array_almost_equal(conf2, expected2)


if __name__ == "__main__":
    unittest.main()