from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from src.models.base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost model for classification and feature importance analysis.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1, random_state: int = 42, **kwargs) -> None:
        """
        Initialize the XGBoostModel.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Boosting learning rate
            random_state (int): Random seed
            **kwargs: Additional model parameters
        """
        # Parameter validation
        if not (0 < learning_rate <= 1):
            raise ValueError("learning_rate must be between 0 and 1")
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError("max_depth must be a positive integer or None")
        
        super().__init__(name="XGBoost")
        from xgboost import XGBClassifier
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state
        }
        self.params.update({k: v for k, v in kwargs.items() if v is not None})
        self.model = XGBClassifier(**self.params)
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.label_encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """
        Fit the XGBoost model to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target values
        Returns:
            XGBoostModel: The fitted model instance
        """
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.feature_importance_ = dict(zip(X.columns, self.model.feature_importances_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted XGBoost model.
        
        Args:
            X (pd.DataFrame): Features to make predictions for
        Returns:
            np.ndarray: Predicted values (decoded to original labels)
        """
        y_pred = self.model.predict(X)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_pred)
        return y_pred

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the XGBoost model.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.feature_importance_

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence using the XGBoost model.
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
        Returns:
            np.ndarray: Confidence estimates for each prediction
        """
        proba = self.model.predict_proba(X)
        return np.max(proba, axis=1)