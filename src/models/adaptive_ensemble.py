from typing import List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator

from .base_model import BaseModel

# Add new AdaptiveEnsemble implementation
class AdaptiveEnsemble(BaseModel):
    """Adaptive ensemble model that dynamically adjusts weights based on performance."""
    
    def __init__(self, base_models: List[BaseEstimator], window_size: int = 10):
        """Initialize the adaptive ensemble.
        
        Args:
            base_models: List of base models to ensemble
            window_size: Size of the window for weight adaptation
        """
        super().__init__()
        self.base_models = base_models
        self.window_size = window_size
        self.model_weights = np.ones(len(base_models)) / len(base_models)
        self.prediction_history: List[np.ndarray] = []
        self.error_history: List[np.ndarray] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit each base model and initialize weights.
        
        Args:
            X: Training features
            y: Target values
        """
        for model in self.base_models:
            model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted combination of base models.
        
        Args:
            X: Input features
            
        Returns:
            Weighted predictions
        """
        predictions = np.array([model.predict(X) for model in self.base_models])
        return np.average(predictions, axis=0, weights=self.model_weights)
        
    def update_weights(self, y_true: np.ndarray, predictions: np.ndarray) -> None:
        """Update model weights based on recent performance.
        
        Args:
            y_true: True values
            predictions: Model predictions
        """
        # Calculate errors for each model
        errors = np.abs(predictions - y_true.reshape(-1, 1))
        self.error_history.append(errors)
        
        # Keep only recent history
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            
        # Calculate average errors over window
        avg_errors = np.mean(self.error_history, axis=0)
        
        # Update weights inversely proportional to errors
        weights = 1.0 / (avg_errors + 1e-10)  # Add small constant to avoid division by zero
        self.model_weights = weights / np.sum(weights)
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'base_models': self.base_models,
            'window_size': self.window_size,
            'model_weights': self.model_weights.tolist()
        }
        
    def set_params(self, **params: Any) -> None:
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
