# Standard library imports
from typing import Dict, Any, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        self.performance_metrics: Dict[str, float] = {}
        
    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess input data for Random Forest model.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Preprocessed features and target
        """
        # Ensure numerical features
        X = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y
        
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
        # Preprocess data
        X_processed, y_processed = self.preprocess(X, y)
        
        # Ensure data types are correct
        X_processed = X_processed.astype(float)
        y_processed = y_processed.astype(float)
        
        # Fit the model
        self.model.fit(X_processed, y_processed)
        
        # Store feature importance
        self.feature_importance_ = dict(zip(X_processed.columns, 
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
        
        # Preprocess data
        X_processed, _ = self.preprocess(X)
            
        # Ensure data types are correct
        X_processed = X_processed.astype(float)
        
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get uncertainty estimates for predictions using the fitted Random Forest model.
        For regression, this returns the standard deviation of individual tree predictions.
        
        Args:
            X: Features to make predictions for
            
        Returns:
            np.ndarray: Standard deviation of predictions across trees (uncertainty measure)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        # Preprocess data
        X_processed, _ = self.preprocess(X)
        X_processed = X_processed.astype(float)
        
        # Get predictions from all estimators
        predictions = np.array([tree.predict(X_processed) for tree in self.model.estimators_])
        
        # Calculate standard deviation across trees
        return np.std(predictions, axis=0)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using multiple regression metrics.
        
        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True target values
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        # Preprocess data
        X_processed, y_processed = self.preprocess(X, y)
        
        # Make predictions
        predictions = self.predict(X_processed)
        
        # Calculate metrics
        mse = mean_squared_error(y_processed, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_processed, predictions)
        r2 = r2_score(y_processed, predictions)
        
        # Store metrics
        self.performance_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        return self.performance_metrics
    
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
            'feature_importance': self.feature_importance_,
            'performance': self.performance_metrics
        }
        
    def get_oob_score(self) -> float:
        """
        Get the out-of-bag (OOB) score for the model, if available.
        The OOB score is an unbiased estimate of the model's performance
        without requiring a separate test set.
        
        Returns:
            float: OOB score
        
        Raises:
            ValueError: If oob_score was not enabled during initialization
        """
        if not hasattr(self.model, 'oob_score_'):
            raise ValueError("OOB score not available. Enable oob_score=True during model initialization.")
            
        return float(self.model.oob_score_)