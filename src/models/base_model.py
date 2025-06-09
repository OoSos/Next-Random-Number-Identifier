# Standard library imports
from abc import ABC, abstractmethod
from typing import Dict, Any

# Third-party imports
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    All models must implement the core interface for fit, predict, feature importance, and confidence estimation.
    """
    def __init__(self, name: str, **kwargs) -> None:
        """
        Initialize the base model.
        
        Args:
            name (str): Name of the model
            **kwargs: Additional model parameters
        """
        self.name = name
        self.model = None
        self.params = kwargs
        self.feature_importance_: Any = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target values
        Returns:
            BaseModel: The fitted model instance
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
    
    @abstractmethod
    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence.
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
        Returns:
            np.ndarray: Confidence estimates
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
            BaseModel: The model instance with updated parameters
        """
        self.params.update(params)
        return self
    
    def __str__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: Model name and type
        """
        return f"{self.name} Model"