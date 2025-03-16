from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from .base_model import BaseModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .markov_chain import MarkovChain
from sklearn.metrics import mean_squared_error

class ModelPerformanceTracker:
    """Tracks and analyzes model performance over time."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.baseline_metrics: Dict[str, float] = {}
    
    def update_metrics(self, model_name: str, metrics: Dict[str, float], timestamp: datetime) -> None:
        """Update performance metrics for a specific model."""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Keep only recent history based on window size
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name].pop(0)

    def get_performance_trend(self, model_name: str) -> Dict[str, float]:
        """Calculate performance trends for a model."""
        if model_name not in self.performance_history:
            return {}
        
        history = self.performance_history[model_name]
        if not history:
            return {}
        
        # Calculate trends for each metric
        trends = {}
        metrics = history[0]['metrics'].keys()
        
        for metric in metrics:
            values = [float(h['metrics'][metric]) for h in history]  # Explicit float conversion
            trends[f'{metric}_trend'] = float((values[-1] - values[0]) / len(values))  # Explicit float conversion
        
        return trends

class EnhancedEnsemble(BaseEstimator, RegressorMixin):
    """
    Advanced ensemble framework that combines multiple prediction models
    with dynamic weight adjustment and performance monitoring.
    """
    
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        weights: Optional[np.ndarray] = None,
        performance_window: int = 10,
        min_weight: float = 0.1
    ):
        self.models = models or self._initialize_default_models()
        self.weights = self._initialize_weights(weights, len(self.models))
        self.min_weight = min_weight
        self.performance_tracker = ModelPerformanceTracker(performance_window)
        self.feature_importance_: Optional[Dict[str, float]] = None
        
        logging.info(f"Initialized EnhancedEnsemble with {len(self.models)} models")

    def _initialize_default_models(self) -> List[BaseModel]:
        """Initialize the default set of models."""
        return [
            RandomForestModel(n_estimators=100),
            XGBoostModel(),
            MarkovChain(order=2)
        ]

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
                model.fit(X, y)
                
                # Get initial performance metrics
                predictions = model.predict(X)
                metrics = self._calculate_metrics(y, predictions)
                
                self.performance_tracker.update_metrics(
                    f"model_{i}",
                    metrics,
                    datetime.now()
                )
                
            except Exception as e:
                logging.error(f"Error fitting model {i}: {str(e)}")
                raise
        
        # Calculate feature importance
        self._update_feature_importance()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions with confidence intervals.
        
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

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'mse': float(np.mean((y_true - y_pred) ** 2)),  # Explicit float conversion
            'mae': float(np.mean(np.abs(y_true - y_pred)))  # Explicit float conversion
        }

    def _update_feature_importance(self) -> None:
        """Update ensemble feature importance by combining individual model importances."""
        importance_dict: Dict[str, float] = {}
        
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'get_feature_importance'):
                model_importance = model.get_feature_importance()
                for feature, importance in model_importance.items():
                    if feature not in importance_dict:
                        importance_dict[feature] = 0.0
                    importance_dict[feature] += float(importance * weight)  # Explicit float conversion
        
        self.feature_importance_ = importance_dict

    def get_model_contributions(self) -> Dict[str, float]:
        """Calculate the contribution of each model to the ensemble."""
        contributions: Dict[str, float] = {}
        
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            model_name = model.__class__.__name__
            contributions[model_name] = float(weight)  # Explicit float conversion
        
        return contributions

    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of model performance and ensemble metrics."""
        summary = {
            'model_weights': self.get_model_contributions(),
            'performance_trends': {}
        }
        
        for i in range(len(self.models)):
            model_name = f"model_{i}"
            summary['performance_trends'][model_name] = (
                self.performance_tracker.get_performance_trend(model_name)
            )
        
        return summary

class AdaptiveEnsemble(BaseModel):
    """
    Adaptive ensemble with dynamic model weighting based on recent performance.
    """
    def __init__(self, models: List[BaseModel], window_size: int = 10):
        super().__init__(name="AdaptiveEnsemble")
        self.models = models
        self.window_size = window_size
        self.weights = np.ones(len(models)) / len(models)  # Equal weights initially
        self.recent_performances = []
        
    def update_weights(self, y_true: pd.Series, predictions: List[np.ndarray]):
        """
        Update model weights based on recent performance.
        """
        # Calculate errors for each model
        errors = [mean_squared_error(y_true, pred) for pred in predictions]
        
        # Convert errors to accuracy scores (inverse of error)
        accuracies = [1 / (err + 1e-10) for err in errors]  # Add small constant to avoid division by zero
        
        # Normalize to get weights
        total_accuracy = sum(accuracies)
        new_weights = [acc / total_accuracy for acc in accuracies]
        
        # Apply exponential smoothing to weights
        alpha = 0.3  # Smoothing factor
        self.weights = [alpha * new_w + (1 - alpha) * old_w 
                      for new_w, old_w in zip(new_weights, self.weights)]