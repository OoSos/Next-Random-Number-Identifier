from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost implementation for random number prediction.
    Inherits from BaseModel and implements classification-specific methods.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model with custom or default parameters.
        """
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'random_state': 42,
            'use_label_encoder': False
        }
        
        self.params = params if params is not None else self.default_params
        self.model = XGBClassifier(**self.params)
        self.feature_importance: Optional[pd.DataFrame] = None
        # Explicitly declare the type of performance_metrics
        self.performance_metrics: Dict[str, float] = {}

    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess input data for XGBoost model.
        
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
        
        # Adjust target for classification (0-based)
        if y is not None:
            y = y - 1
            
        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the XGBoost model to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
        """
        X_processed, y_processed = self.preprocess(X, y)
        self.model.fit(X_processed, y_processed)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values (adjusted back to original scale)
        """
        X_processed, _ = self.preprocess(X)
        predictions = self.model.predict(X_processed)
        return predictions + 1  # Adjust back to original scale

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions for each class.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Probability predictions for each class
        """
        X_processed, _ = self.preprocess(X)
        return self.model.predict_proba(X_processed)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True target values
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        X_processed, y_processed = self.preprocess(X, y)
        predictions = self.predict(X_processed)
        y_pred = predictions - 1  # Adjust predictions for metric calculation
        
        if y_processed is None:
            raise ValueError("Cannot evaluate model without target values")
    
        # Convert to numpy arrays and ensure correct types
        y_true = np.asarray(y_processed, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
    
        # Create new dictionary with explicit float conversion
        metrics_dict = {
            'accuracy': accuracy_score(y_processed, y_pred),
            'precision': precision_score(y_processed, y_pred, average='weighted'),
            'recall': recall_score(y_processed, y_pred, average='weighted'),
            'f1': f1_score(y_processed, y_pred, average='weighted')
        }
        
        return self.performance_metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained before getting feature importance")
        return self.feature_importance

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dict[str, Any]: Current model parameters
        """
        return self.model.get_params()