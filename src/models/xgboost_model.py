from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost implementation for random number prediction.
    Inherits from BaseModel and implements classification-specific methods.
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        """
        Initialize the XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Boosting learning rate
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for XGBClassifier
        """
        super().__init__(name="XGBoost")
        self.params.update({
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'random_state': random_state,
            'use_label_encoder': False,
            **kwargs
        })
        self.model = XGBClassifier(**self.params)
        self.feature_importance_ = None
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
            y = y - 1 if y.min() > 0 else y.copy()
            
        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """
        Fit the XGBoost model to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            self: The fitted model instance
        """
        X_processed, y_processed = self.preprocess(X, y)
        self.model.fit(X_processed, y_processed)
        
        # Calculate feature importance
        self.feature_importance_ = dict(zip(X_processed.columns, self.model.feature_importances_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values (adjusted back to original scale)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
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
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
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
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
            
        X_processed, y_processed = self.preprocess(X, y)
        y_pred = self.model.predict(X_processed)
        
        # Create metrics dictionary
        metrics_dict = {
            'accuracy': accuracy_score(y_processed, y_pred),
            'precision': precision_score(y_processed, y_pred, average='weighted'),
            'recall': recall_score(y_processed, y_pred, average='weighted'),
            'f1': f1_score(y_processed, y_pred, average='weighted')
        }
        
        # Update performance metrics
        self.performance_metrics = metrics_dict
        return metrics_dict

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.feature_importance_

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dict[str, Any]: Current model parameters
        """
        return self.model.get_params()