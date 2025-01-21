from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Dict, Any

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    """
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.params = kwargs
        self.feature_importance_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target values
            
        Returns:
            self: The fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X (pd.DataFrame): Features to make predictions for
            
        Returns:
            np.ndarray: Predicted values
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return self.params
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        self.params.update(params)
        return self
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} Model"