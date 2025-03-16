from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class BayesianModelAveraging(BaseModel):
    def __init__(self, models: List[BaseModel], **kwargs):
        super().__init__(name='BayesianModelAveraging', **kwargs)
        self.models = models

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BayesianModelAveraging':
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

    def get_feature_importance(self) -> Dict[str, float]:
        feature_importance = {}
        for model in self.models:
            model_importance = model.get_feature_importance()
            for feature, importance in model_importance.items():
                if feature not in feature_importance:
                    feature_importance[feature] = 0
                feature_importance[feature] += importance
        # Normalize importance
        total_importance = sum(feature_importance.values())
        for feature in feature_importance:
            feature_importance[feature] /= total_importance
        return feature_importance

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])
        return np.std(predictions, axis=0)