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
        min_weight: float = 0.1,
        combination_method: str = 'weighted_average'
    ):
        self.models = models or self._initialize_default_models()
        self.weights = self._initialize_weights(weights, len(self.models))
        self.min_weight = min_weight
        self.performance_tracker = ModelPerformanceTracker(performance_window)
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.combination_method = combination_method
        
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
        return {
            'mse': float(np.mean((y_true - y_pred) ** 2)),  # Explicit float conversion
            'mae': float(np.mean(np.abs(y_true - y_pred)))  # Explicit float conversion
        }

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
            if max_importance > 0:
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
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaptiveEnsemble':
        """
        Fit all component models.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            self: Fitted model
        """
        for model in self.models:
            model.fit(X, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted values
        """
        predictions = [model.predict(X) for model in self.models]
        return np.average(predictions, weights=self.weights, axis=0)
        
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
                      
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get combined feature importance from all models.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        importance_dict = {}
        
        for model, weight in zip(self.models, self.weights):
            model_importance = model.get_feature_importance()
            for feature, importance in model_importance.items():
                if feature not in importance_dict:
                    importance_dict[feature] = 0.0
                importance_dict[feature] += importance * weight
                
        # Normalize importance values
        if importance_dict:
            max_importance = max(importance_dict.values())
            if max_importance > 0:
                importance_dict = {
                    feature: importance / max_importance
                    for feature, importance in importance_dict.items()
                }
                
        return importance_dict

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence using the ensemble of models.
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
            
        Returns:
            np.ndarray: Confidence estimates for each prediction
        """
        confidences = np.array([model.estimate_confidence(X) for model in self.models])
        return np.average(confidences, weights=self.weights, axis=0)