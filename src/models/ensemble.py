"""
Enhanced Ensemble Model for Next Random Number Identifier

This module implements the ensemble model that combines predictions from multiple
base models. For detailed information about the system architecture and how this
component fits into the larger system, please refer to:

- Architecture Documentation: docs/Next Random Number Identifier-architecture-documentation.md
- Component Interaction: docs/diagrams/NRNI Component Interaction-diagrams.png
- Data Flow: docs/diagrams/NRNI Data-flow-diagram.png

The ensemble model integrates predictions from:
- Random Forest Regression
- XGBoost Classification
- Markov Chain Analysis
"""

from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from .base_model import BaseModel
from .random_forest import RandomForestModel
from .markov_chain import MarkovChain, VariableOrderMarkovChain
try:
    from .xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.metrics import mean_squared_error

class ModelPerformanceTracker:
    """Tracks and analyzes model performance over time."""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize the performance tracker.
        
        Args:
            window_size: Number of performance records to keep per model
        """
        self.window_size = window_size
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
    
    def update_metrics(self, model_name: str, metrics: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """
        Update performance metrics for a specific model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            timestamp: Optional timestamp for the metrics (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Keep only recent history based on window size
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name].pop(0)

    def set_baseline(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        Set baseline metrics for a model for future comparison.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of baseline performance metrics
        """
        self.baseline_metrics[model_name] = metrics.copy()

    def get_performance_trend(self, model_name: str) -> Dict[str, float]:
        """
        Calculate performance trends for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with trend metrics
        """
        if model_name not in self.performance_history:
            return {}
        
        history = self.performance_history[model_name]
        if len(history) < 2:
            return {}
        
        # Calculate trends for each metric
        trends = {}
        metrics = history[0]['metrics'].keys()
        
        for metric in metrics:
            values = [float(h['metrics'][metric]) for h in history if metric in h['metrics']]
            
            if values:
                # Calculate trend as normalized change over time
                first_value = values[0]
                last_value = values[-1]
                if first_value != 0:
                    trend = (last_value - first_value) / first_value * 100  # Percentage change
                else:
                    trend = 0 if last_value == 0 else float('inf')
                
                trends[f'{metric}_trend'] = float(trend)
                
                # Also include absolute change
                trends[f'{metric}_abs_change'] = float(last_value - first_value)
        
        return trends
    
    def detect_drift(self, model_name: str, current_metrics: Dict[str, float], threshold: float = 0.2) -> Dict[str, Any]:
        """
        Detect if model performance has drifted significantly from baseline.
        
        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
            threshold: Threshold for significant drift (proportional change)
            
        Returns:
            Dictionary with drift detection results
        """
        if model_name not in self.baseline_metrics:
            return {'drift_detected': False, 'reason': 'No baseline metrics available'}
        
        baseline = self.baseline_metrics[model_name]
        drift_detected = False
        drifting_metrics = {}
        
        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                
                # Calculate proportional change
                if baseline_value != 0:
                    change = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    change = 0 if current_value == 0 else float('inf')
                
                # Check if change exceeds threshold
                if change > threshold:
                    drift_detected = True
                    drifting_metrics[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change': change
                    }
        
        return {
            'drift_detected': drift_detected,
            'drifting_metrics': drifting_metrics
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked model performance.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {
            'models': list(self.performance_history.keys()),
            'latest_metrics': {},
            'trends': {},
            'drift': {}
        }
        
        for model_name in self.performance_history:
            history = self.performance_history[model_name]
            if history:
                # Get latest metrics
                summary['latest_metrics'][model_name] = history[-1]['metrics']
                
                # Get trends
                summary['trends'][model_name] = self.get_performance_trend(model_name)
                
                # Check for drift
                if model_name in self.baseline_metrics:
                    summary['drift'][model_name] = self.detect_drift(
                        model_name, 
                        history[-1]['metrics']
                    )
        
        return summary

    def save_performance_history(self, filepath: str) -> None:
        """
        Save the performance history to a CSV file.
        
        Args:
            filepath: Path to the CSV file
        """
        records = []
        for model_name, history in self.performance_history.items():
            for record in history:
                record_flat = {
                    'model_name': model_name,
                    'timestamp': record['timestamp'],
                    **record['metrics']
                }
                records.append(record_flat)
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)

class EnhancedEnsemble(BaseEstimator, RegressorMixin):
    """
    Advanced ensemble framework that combines multiple prediction models
    with dynamic weight adjustment and performance monitoring.
    
    The ensemble integrates Random Forest, XGBoost, and Markov Chain models
    to leverage their complementary strengths. It includes performance tracking,
    dynamic weight optimization, and feature importance analysis.
    """
    
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        weights: Optional[np.ndarray] = None,
        performance_window: int = 10,
        min_weight: float = 0.1,
        combination_method: str = 'weighted_average',
        weight_update_strategy: str = 'performance',
        tracking_metrics: List[str] = ['mse', 'mae', 'accuracy']
    ):
        """
        Initialize the enhanced ensemble.
        
        Args:
            models: List of models to include in the ensemble (defaults to RF, XGB, Markov)
            weights: Initial weights for models (defaults to equal weights)
            performance_window: Window size for performance tracking
            min_weight: Minimum weight for any model
            combination_method: Method to combine model predictions
            weight_update_strategy: Strategy for updating weights ('performance', 'accuracy', 'equal')
            tracking_metrics: Metrics to track for performance monitoring
        """
        self.models = models or self._initialize_default_models()
        self.weights = self._initialize_weights(weights, len(self.models))
        self.min_weight = min_weight
        self.performance_tracker = ModelPerformanceTracker(performance_window)
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.combination_method = combination_method
        self.weight_update_strategy = weight_update_strategy
        self.tracking_metrics = tracking_metrics
        self.is_fitted = False
        
        logging.info(f"Initialized EnhancedEnsemble with {len(self.models)} models")

    def _initialize_default_models(self) -> List[BaseModel]:
        """Initialize the default set of models."""
        default_models = [
            RandomForestModel(n_estimators=100),
            MarkovChain(order=2)
        ]
        
        if XGBOOST_AVAILABLE:
            default_models.append(XGBoostModel())
            
        return default_models

    def _initialize_weights(self, weights: Optional[np.ndarray], n_models: int) -> np.ndarray:
        """Initialize model weights."""
        if weights is None:
            return np.ones(n_models) / n_models
        
        if len(weights) != n_models:
            raise ValueError("Number of weights must match number of models")
        
        return weights / np.sum(weights)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnhancedEnsemble':
        """
        Fit all models in the ensemble and initialize performance tracking.
        
        Args:
            X: Training features
            y: Target values
        """
        for i, model in enumerate(self.models):
            try:
                logging.info(f"Fitting model {i} ({model.__class__.__name__})")
                model.fit(X, y)
                
                # Get initial performance metrics
                predictions = model.predict(X)
                metrics = self._calculate_metrics(y, predictions)
                
                self.performance_tracker.update_metrics(
                    f"model_{i}",
                    metrics,
                    datetime.now()
                )
                
                # Set baseline metrics for drift detection
                self.performance_tracker.set_baseline(f"model_{i}", metrics)
                
            except Exception as e:
                logging.error(f"Error fitting model {i}: {str(e)}")
                raise
        
        # Calculate feature importance
        self._update_feature_importance()
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions using the selected combination method.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted values
        """
        if self.combination_method == 'weighted_average':
            return self._weighted_average_prediction(X)
        elif self.combination_method == 'bayesian':
            return self.bayesian_model_averaging(X)
        elif self.combination_method == 'confidence_weighted':
            return self.dynamic_confidence_weighting(X)
        elif self.combination_method == 'variance_weighted':
            return self.variance_weighted_combination(X)
        else:
            logging.warning(f"Unknown combination method: {self.combination_method}. Using weighted average.")
            return self._weighted_average_prediction(X)

    def _weighted_average_prediction(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions using weighted average.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted values
        """
        predictions = []
        
        for model in self.models:
            try:
                model_pred = model.predict(X)
                predictions.append(model_pred)
            except Exception as e:
                logging.error(f"Error in model prediction: {str(e)}")
                continue
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, weights=self.weights, axis=0)
        
        return weighted_pred

    def update_weights(self, performance_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Update model weights based on recent performance.
        
        Args:
            performance_metrics: Dictionary of performance metrics for each model
        """
        # Calculate new weights based on exponential moving average of performance
        new_weights = np.array([
            float(np.mean(list(metrics.values())))  # Explicit float conversion
            for metrics in performance_metrics.values()
        ])
        
        # Apply softmax to convert to probabilities
        new_weights = np.exp(new_weights) / np.sum(np.exp(new_weights))
        
        # Apply minimum weight constraint
        new_weights = np.maximum(new_weights, self.min_weight)
        new_weights = new_weights / np.sum(new_weights)
        
        self.weights = new_weights
        logging.info(f"Updated ensemble weights: {self.weights}")

    def bayesian_model_averaging(self, X: pd.DataFrame) -> np.ndarray:
        """
        Perform Bayesian model averaging for predictions.
        
        Args:
            X: Input features
        
        Returns:
            Array of predicted values
        """
        predictions = np.array([model.predict(X) for model in self.models])
        weights = self.weights / np.sum(self.weights)
        return np.dot(weights, predictions)

    def model_stacking(self, X: pd.DataFrame, meta_learner: BaseEstimator) -> np.ndarray:
        """
        Perform model stacking with a meta-learner.
        
        Args:
            X: Input features
            meta_learner: Meta-learner model
        
        Returns:
            Array of predicted values
        """
        base_predictions = np.column_stack([model.predict(X) for model in self.models])
        meta_learner.fit(base_predictions, X)
        return meta_learner.predict(base_predictions)

    def dynamic_confidence_weighting(self, X: pd.DataFrame) -> np.ndarray:
        """
        Perform dynamic confidence-based weighting for predictions.
        
        This method adjusts weights based on the confidence of each model's predictions.
        Models with higher confidence for a specific sample will have more influence.
        
        Args:
            X: Input features
        
        Returns:
            Array of predicted values
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Get confidence estimates for each model
        confidences = np.array([model.estimate_confidence(X) for model in self.models])
        
        # Normalize confidences across models for each sample
        sum_confidences = np.sum(confidences, axis=0)
        
        # Avoid division by zero
        sum_confidences = np.where(sum_confidences == 0, 1.0, sum_confidences)
        
        # Calculate dynamic weights for each prediction
        dynamic_weights = confidences / sum_confidences[np.newaxis, :]
        
        # Apply model weights to the confidence weights
        for i in range(len(self.weights)):
            dynamic_weights[i] *= self.weights[i]
            
        # Normalize dynamic weights
        dynamic_weights = dynamic_weights / np.sum(dynamic_weights, axis=0)[np.newaxis, :]
        
        # Weighted sum of predictions for each sample
        weighted_predictions = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            weighted_predictions += predictions[i] * dynamic_weights[i]
            
        return weighted_predictions

    def variance_weighted_combination(self, X: pd.DataFrame) -> np.ndarray:
        """
        Weight predictions based on the inverse of their variance (uncertainty).
        
        This approach gives more weight to models with lower prediction variance/uncertainty.
        
        Args:
            X: Input features
        
        Returns:
            Array of predicted values
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Estimate variance as inverse of confidence
        confidences = np.array([model.estimate_confidence(X) for model in self.models])
        
        # Convert confidence to variance (higher confidence = lower variance)
        # Add a small constant to avoid division by zero
        variances = 1.0 / (confidences + 1e-10)
        
        # Calculate precision (inverse of variance)
        precisions = 1.0 / (variances + 1e-10)
        
        # Apply model weights to the precision values
        for i in range(len(self.weights)):
            precisions[i] *= self.weights[i]
        
        # Normalize precisions to get weights
        sum_precisions = np.sum(precisions, axis=0)
        normalized_precisions = precisions / (sum_precisions[np.newaxis, :] + 1e-10)
        
        # Weighted sum of predictions
        weighted_predictions = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            weighted_predictions += predictions[i] * normalized_precisions[i]
        
        return weighted_predictions

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence using the ensemble of models.
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
            
        Returns:
            np.ndarray: Confidence estimates for each prediction
        """
        try:
            confidences = np.array([model.estimate_confidence(X) for model in self.models])
            
            if self.combination_method == 'weighted_average':
                # Weighted average of confidence values
                return np.average(confidences, weights=self.weights, axis=0)
            elif self.combination_method == 'confidence_weighted':
                # Return the maximum confidence for each prediction
                return np.max(confidences, axis=0)
            elif self.combination_method == 'variance_weighted':
                # Calculate the combined confidence based on variance weighting
                variances = 1.0 / (confidences + 1e-10)
                combined_variance = 1.0 / np.sum(1.0 / (variances + 1e-10), axis=0)
                return 1.0 / (combined_variance + 1e-10)
            else:
                # Default: weighted average
                return np.average(confidences, weights=self.weights, axis=0)
                
        except Exception as e:
            logging.error(f"Error estimating confidence: {str(e)}")
            # Return default confidence of 0.5
            return np.ones(X.shape[0]) * 0.5

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {
            'mse': float(np.mean((y_true - y_pred) ** 2)),  # Explicit float conversion
            'mae': float(np.mean(np.abs(y_true - y_pred)))  # Explicit float conversion
        }
        
        # Add accuracy if in tracking metrics
        if 'accuracy' in self.tracking_metrics:
            metrics['accuracy'] = float((y_true.values == y_pred).mean())
            
        return metrics

    def _update_feature_importance(self) -> None:
        """Update ensemble feature importance by combining individual model importances."""
        importance_dict: Dict[str, float] = {}
        
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'get_feature_importance'):
                try:
                    model_importance = model.get_feature_importance()
                    for feature, importance in model_importance.items():
                        if feature not in importance_dict:
                            importance_dict[feature] = 0.0
                        importance_dict[feature] += float(importance * weight)  # Explicit float conversion
                except Exception as e:
                    logging.warning(f"Error getting feature importance from model {model.__class__.__name__}: {str(e)}")
        
        # Normalize importance values
        if importance_dict:
            max_importance = max(importance_dict.values())
            if (max_importance > 0):
                importance_dict = {
                    feature: float(importance / max_importance)
                    for feature, importance in importance_dict.items()
                }
        
        self.feature_importance_ = importance_dict

    def get_model_contributions(self) -> Dict[str, float]:
        """Calculate the contribution of each model to the ensemble."""
        contributions: Dict[str, float] = {}
        
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            # Use name attribute if available (for test mocks), otherwise use class name
            if hasattr(model, 'name'):
                model_name = model.name
            else:
                model_name = model.__class__.__name__
            
            contributions[model_name] = float(weight)  # Explicit float conversion
        
        return contributions

    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of model performance and ensemble metrics."""
        summary = {
            'model_weights': self.get_model_contributions(),
            'performance_trends': {},
            'combination_method': self.combination_method
        }
        
        for i in range(len(self.models)):
            model_name = f"model_{i}"
            summary['performance_trends'][model_name] = (
                self.performance_tracker.get_performance_trend(model_name)
            )
        
        return summary
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the ensemble's combined feature importance.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Ensemble must be fitted before getting feature importance")
            
        return self.feature_importance_
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate ensemble performance on test data.
        
        Args:
            X: Test features
            y: Target values
            
        Returns:
            Dictionary of performance metrics
        """
        predictions = self.predict(X)
        return self._calculate_metrics(y, predictions)

    def check_drift(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Check for model drift in component models and ensemble.
        
        Args:
            X: Current data features
            y: Current target values
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {}
        
        # Get current performance for each component model
        for i, model in enumerate(self.models):
            model_name = f"model_{i}"
            predictions = model.predict(X)
            metrics = self._calculate_metrics(y, predictions)
            
            # Update performance history
            self.performance_tracker.update_metrics(model_name, metrics)
            
            # Check for drift
            drift_results[model_name] = self.performance_tracker.detect_drift(
                model_name, metrics
            )
        
        # Check ensemble drift
        ensemble_metrics = self.evaluate(X, y)
        ensemble_name = "ensemble"
        
        # Update and check ensemble drift
        self.performance_tracker.update_metrics(ensemble_name, ensemble_metrics)
        if ensemble_name not in self.performance_tracker.baseline_metrics:
            self.performance_tracker.set_baseline(ensemble_name, ensemble_metrics)
            drift_results[ensemble_name] = {'drift_detected': False, 'reason': 'First evaluation'}
        else:
            drift_results[ensemble_name] = self.performance_tracker.detect_drift(
                ensemble_name, ensemble_metrics
            )
        
        return drift_results