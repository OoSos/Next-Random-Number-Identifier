# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Local application imports
from models.base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest implementation for number prediction. This class extends the BaseModel
    to provide a concrete implementation using scikit-learn's RandomForestRegressor.
    The model is specifically tuned for time series prediction tasks with additional
    functionality for feature importance analysis.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,  # Using Optional[int] to allow None
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the Random Forest model with specific parameters for time series prediction.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_leaf: Minimum samples required at leaf nodes
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to RandomForestRegressor
        """
        super().__init__(name="RandomForest")
        
        # Filter out None values to avoid type issues
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            **kwargs
        }
        if max_depth is not None:
            params['max_depth'] = max_depth
        
        self.params = params
        self.model = RandomForestRegressor(**self.params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """
        Fit the Random Forest model to the training data. This method handles the training
        process and stores feature importance information.
        
        Args:
            X: Training features as a DataFrame
            y: Target values as a Series
            
        Returns:
            self: The fitted model instance
        """
        # Ensure data types are correct
        X = X.astype(float)
        y = y.astype(float)
        
        # Fit the model
        self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance_ = dict(zip(X.columns, 
                                          self.model.feature_importances_))
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted Random Forest model. This method includes
        input validation and type checking.
        
        Args:
            X: Features to make predictions for
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        # Ensure data types are correct
        X = X.astype(float)
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance scores for each feature used in the model. This helps
        identify which features are most influential in making predictions.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to their importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
            
        return self.feature_importance_
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model's current state and configuration.
        This is useful for model analysis and debugging.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information
        """
        return {
            'name': self.name,
            'parameters': self.get_params(),
            'n_features': len(self.feature_importance_) if self.feature_importance_ else None,
            'n_trees': self.params['n_estimators'],
            'max_depth': self.params.get('max_depth', None),
            'feature_importance': self.feature_importance_
        }