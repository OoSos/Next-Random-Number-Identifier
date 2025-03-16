from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression

class ModelStacking(BaseModel):
    def __init__(self, base_models: List[BaseModel], meta_model: BaseModel = LogisticRegression(), **kwargs):
        super().__init__(name='ModelStacking', **kwargs)
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelStacking':
        # Fit base models
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            meta_features[:, i] = model.predict(X)
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X)
        return self.meta_model.predict(meta_features)

    def get_feature_importance(self) -> Dict[str, float]:
        # Feature importance is not straightforward for stacking
        return {}

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence using the ensemble of base models.
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
            
        Returns:
            np.ndarray: Confidence estimates for each prediction
        """
        confidences = np.array([model.estimate_confidence(X) for model in self.base_models])
        return np.mean(confidences, axis=0)