from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator
from .base_model import BaseModel

class OptimizedEnsemble(BaseModel):
    """Ensemble model with optimized weights through grid search."""
    
    def __init__(self, base_models: List[BaseEstimator], cv_folds: int = 5):
        """Initialize the optimized ensemble.
        
        Args:
            base_models: List of base models to ensemble
            cv_folds: Number of cross-validation folds
        """
        super().__init__()
        self.base_models = base_models
        self.cv_folds = cv_folds
        self.model_weights = np.ones(len(base_models)) / len(base_models)
        self.best_weights: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit base models and optimize weights.
        
        Args:
            X: Training features
            y: Target values
        """
        # First fit all base models
        for model in self.base_models:
            model.fit(X, y)
            
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.base_models])
        
        # Define weight grid
        weight_grid = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
        best_score = float('inf')
        best_weights = None
        
        # Grid search for optimal weights
        for weights in self._generate_weight_combinations(weight_grid, len(self.base_models)):
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            score = np.mean((ensemble_pred - y) ** 2)
            
            if score < best_score:
                best_score = score
                best_weights = weights
                
        self.best_weights = best_weights
        if best_weights is not None:
            self.model_weights = best_weights
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using optimized weights.
        
        Args:
            X: Input features
            
        Returns:
            Weighted predictions
        """
        predictions = np.array([model.predict(X) for model in self.base_models])
        return np.average(predictions, axis=0, weights=self.model_weights)
        
    def _generate_weight_combinations(self, grid: np.ndarray, n_models: int) -> np.ndarray:
        """Generate valid weight combinations that sum to 1."""
        if n_models == 1:
            return np.array([[1.0]])
        elif n_models == 2:
            weights = []
            for w1 in grid:
                w2 = 1 - w1
                if 0 <= w2 <= 1:
                    weights.append([w1, w2])
            return np.array(weights)
        else:
            weights = []
            for w1 in grid:
                sub_weights = self._generate_weight_combinations(
                    grid, n_models - 1)
                for sw in sub_weights:
                    if sum(sw) <= 1 - w1:
                        scale = (1 - w1) / sum(sw)
                        weights.append([w1] + list(sw * scale))
            return np.array(weights)
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'base_models': self.base_models,
            'cv_folds': self.cv_folds,
            'model_weights': self.model_weights.tolist(),
            'best_weights': self.best_weights.tolist() if self.best_weights is not None else None
        }
        
    def set_params(self, **params: Any) -> None:
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
