import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from src.models.ensemble import EnhancedEnsemble, AdaptiveEnsemble, ModelPerformanceTracker
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.markov_chain import MarkovChain

class MockModel:
    """Mock model for testing purposes."""
    def __init__(self, name="MockModel", return_value=None, confidence_value=None):
        self.name = name
        # Store both patterns and values for backward compatibility with tests
        self.return_pattern = np.array([1.0, 2.0, 3.0]) if return_value is None else return_value
        self.confidence_pattern = np.array([0.9, 0.8, 0.7]) if confidence_value is None else confidence_value
        # Keep old attribute names for compatibility with existing tests
        self.return_value = self.return_pattern
        self.confidence_value = self.confidence_pattern
        self.feature_importance_ = {"feature1": 0.5, "feature2": 0.3, "feature3": 0.2}
        self.fitted = False
    
    def fit(self, X, y):
        self.fitted = True
        return self
    
    def predict(self, X):
        # Return an array that matches the number of rows in X
        n_samples = len(X)
        # Repeat the pattern as needed
        if n_samples <= len(self.return_pattern):
            return self.return_pattern[:n_samples]
        else:
            repeats = int(np.ceil(n_samples / len(self.return_pattern)))
            extended = np.tile(self.return_pattern, repeats)
            return extended[:n_samples]
    
    def estimate_confidence(self, X):
        # Return confidence values that match the number of rows in X
        n_samples = len(X)
        # Repeat the pattern as needed
        if n_samples <= len(self.confidence_pattern):
            return self.confidence_pattern[:n_samples]
        else:
            repeats = int(np.ceil(n_samples / len(self.confidence_pattern)))
            extended = np.tile(self.confidence_pattern, repeats)
            return extended[:n_samples]
    
    def get_feature_importance(self):
        return self.feature_importance_
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"MockModel('{self.name}')"

class TestEnhancedEnsemble(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(5)])
        self.y = pd.Series(y)
        
        # Create mock models with different prediction patterns
        self.model1 = MockModel("Model1", np.array([1.0, 2.0, 3.0]), np.array([0.9, 0.8, 0.7]))
        self.model2 = MockModel("Model2", np.array([2.0, 3.0, 4.0]), np.array([0.7, 0.8, 0.9]))
        self.model3 = MockModel("Model3", np.array([3.0, 4.0, 5.0]), np.array([0.8, 0.7, 0.6]))
        
        self.models = [self.model1, self.model2, self.model3]
        
        # Create ensemble with equal weights
        self.ensemble = EnhancedEnsemble(models=self.models)
        
    def test_initialization(self):
        """Test ensemble initialization with different parameters."""
        # Test default initialization
        default_ensemble = EnhancedEnsemble()
        self.assertEqual(len(default_ensemble.models), 3)
        self.assertTrue(all(isinstance(m, (RandomForestModel, XGBoostModel, MarkovChain)) 
                           for m in default_ensemble.models))
        
        # Test initialization with custom weights
        weights = np.array([0.5, 0.3, 0.2])
        weighted_ensemble = EnhancedEnsemble(models=self.models, weights=weights)
        np.testing.assert_array_almost_equal(weighted_ensemble.weights, weights)
        
        # Test validation of weights length
        with self.assertRaises(ValueError):
            EnhancedEnsemble(models=self.models, weights=np.array([0.5, 0.5]))
    
    def test_fit(self):
        """Test fitting the ensemble models."""
        self.ensemble.fit(self.X, self.y)
        
        # Check if all models were fitted
        for model in self.models:
            self.assertTrue(model.fitted)
        
        # Check if performance metrics were updated
        for i in range(len(self.models)):
            model_name = f"model_{i}"
            self.assertIn(model_name, self.ensemble.performance_tracker.performance_history)
    
    def test_predict(self):
        """Test prediction functionality."""
        self.ensemble.fit(self.X, self.y)
        predictions = self.ensemble.predict(self.X)
        
        # With equal weights, predictions should be average of individual predictions
        # Get predictions from each model for the actual input data
        expected_predictions = []
        for model in self.models:
            expected_predictions.append(model.predict(self.X))
        
        expected = np.average(expected_predictions, weights=self.ensemble.weights, axis=0)
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_update_weights(self):
        """Test weight updating mechanism."""
        # Create performance metrics
        metrics = {
            "model_0": {"mse": 0.1, "mae": 0.2},
            "model_1": {"mse": 0.2, "mae": 0.3},
            "model_2": {"mse": 0.3, "mae": 0.4}
        }
        
        # Get original weights
        original_weights = self.ensemble.weights.copy()
        
        # Update weights
        self.ensemble.update_weights(metrics)
        
        # Check that weights have changed
        self.assertFalse(np.array_equal(self.ensemble.weights, original_weights))
        
        # Check that weights sum to 1
        self.assertAlmostEqual(np.sum(self.ensemble.weights), 1.0)
    
    def test_bayesian_model_averaging(self):
        """Test Bayesian model averaging prediction."""
        bma_predictions = self.ensemble.bayesian_model_averaging(self.X)
        
        # For equal weights, this should be the same as regular prediction
        regular_predictions = self.ensemble.predict(self.X)
        np.testing.assert_array_almost_equal(bma_predictions, regular_predictions)
    
    def test_confidence_estimation(self):
        """Test prediction confidence estimation."""
        confidences = self.ensemble.estimate_confidence(self.X)
        
        # Expected confidence is weighted average of individual confidences
        expected_confidences = []
        for model in self.models:
            expected_confidences.append(model.estimate_confidence(self.X))
        
        expected = np.average(expected_confidences, weights=self.ensemble.weights, axis=0)
        np.testing.assert_array_almost_equal(confidences, expected)
    
    def test_feature_importance(self):
        """Test feature importance combination."""
        self.ensemble.fit(self.X, self.y)
        
        # Check if feature importance was calculated
        self.assertIsNotNone(self.ensemble.feature_importance_)
        
        # Test if all features from individual models are included
        for model in self.models:
            for feature in model.get_feature_importance().keys():
                self.assertIn(feature, self.ensemble.feature_importance_)
    
    def test_get_model_contributions(self):
        """Test model contribution calculation."""
        contributions = self.ensemble.get_model_contributions()
        
        # Check if all models are included
        self.assertEqual(len(contributions), len(self.models))
        
        # Check if contributions sum to approximately 1
        self.assertAlmostEqual(sum(contributions.values()), 1.0)
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        self.ensemble.fit(self.X, self.y)
        summary = self.ensemble.get_performance_summary()
        
        # Check if summary contains required keys
        self.assertIn("model_weights", summary)
        self.assertIn("performance_trends", summary)
        
        # Check if all models are included in performance trends
        for i in range(len(self.models)):
            model_name = f"model_{i}"
            self.assertIn(model_name, summary["performance_trends"])


class TestAdaptiveEnsemble(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(5)])
        self.y = pd.Series(y)
        
        # Create mock models
        self.models = [MockModel(f"Model{i}") for i in range(3)]
        self.ensemble = AdaptiveEnsemble(models=self.models)
    
    def test_update_weights(self):
        """Test adaptive weight updating mechanism."""
        # Create mock predictions
        predictions = [
            np.array([1.0, 2.0, 3.0]),  # Good performance
            np.array([5.0, 6.0, 7.0]),  # Poor performance
            np.array([2.0, 3.0, 4.0])   # Medium performance
        ]
        
        # Target values
        y_true = pd.Series([1.5, 2.5, 3.5])
        
        # Get original weights
        original_weights = self.ensemble.weights.copy()
        
        # Update weights
        self.ensemble.update_weights(y_true, predictions)
        
        # Check that weights have changed
        self.assertFalse(np.array_equal(self.ensemble.weights, original_weights))
        
        # Check that weights sum to 1
        self.assertAlmostEqual(sum(self.ensemble.weights), 1.0)
        
        # Check that better performing model has higher weight
        self.assertTrue(self.ensemble.weights[0] > self.ensemble.weights[1])

    def test_confidence_estimation(self):
        """Test confidence estimation."""
        confidences = self.ensemble.estimate_confidence(self.X)
        
        # Expected confidence is weighted average of individual confidences
        expected_confidences = []
        for model in self.models:
            expected_confidences.append(model.estimate_confidence(self.X))
        
        expected = np.average(expected_confidences, weights=self.ensemble.weights, axis=0)
        np.testing.assert_array_almost_equal(confidences, expected)


class TestModelPerformanceTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ModelPerformanceTracker(window_size=3)
    
    def test_update_metrics(self):
        """Test updating performance metrics."""
        # Add metrics for a model
        model_name = "test_model"
        metrics = {"mse": 0.1, "mae": 0.2}
        timestamp = pd.Timestamp('2023-01-01')
        
        self.tracker.update_metrics(model_name, metrics, timestamp)
        
        # Check if metrics were added
        self.assertIn(model_name, self.tracker.performance_history)
        self.assertEqual(len(self.tracker.performance_history[model_name]), 1)
        
        # Add more metrics to test window size limit
        for i in range(5):
            self.tracker.update_metrics(model_name, {"mse": 0.1 * i, "mae": 0.2 * i}, 
                                     pd.Timestamp(f'2023-01-{i+2}'))
        
        # Check if window size is maintained
        self.assertEqual(len(self.tracker.performance_history[model_name]), 3)
    
    def test_get_performance_trend(self):
        """Test performance trend calculation."""
        model_name = "test_model"
        
        # Add multiple metrics with improving performance
        for i in range(3):
            self.tracker.update_metrics(model_name, 
                                     {"mse": 0.3 - 0.1 * i, "mae": 0.4 - 0.1 * i},
                                     pd.Timestamp(f'2023-01-{i+1}'))
        
        # Calculate trends
        trends = self.tracker.get_performance_trend(model_name)
        
        # Check if trends are calculated and negative (improving)
        self.assertIn("mse_trend", trends)
        self.assertIn("mae_trend", trends)
        self.assertTrue(trends["mse_trend"] < 0)
        self.assertTrue(trends["mae_trend"] < 0)


if __name__ == "__main__":
    unittest.main()