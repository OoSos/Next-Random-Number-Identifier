"""
Feature selection module for identifying and selecting the most relevant features.
"""

from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection class that combines multiple selection methods.
    """
    def __init__(
        self,
        n_features: int = 20,
        selection_method: str = 'combined',
        random_state: int = 42
    ):
        """
        Initialize the FeatureSelector.

        Args:
            n_features: Number of features to select
            selection_method: One of ['combined', 'mutual_info', 'random_forest']
            random_state: Random state for reproducibility
        """
        self.n_features = n_features
        self.selection_method = selection_method
        self.random_state = random_state
        self.selected_features_: Optional[List[str]] = None
        self.feature_importances_: Optional[Dict[str, float]] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector to the data.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            self: The fitted selector
        """
        if self.selection_method not in ['combined', 'mutual_info', 'random_forest']:
            raise ValueError("Invalid selection method")
            
        importances = {}
        
        # Mutual information scores
        mi_selector = SelectKBest(mutual_info_regression, k='all')
        mi_selector.fit(X, y)
        mi_scores = mi_selector.scores_
        
        # Random Forest importance scores
        rf_selector = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        rf_selector.fit(X, y)
        rf_scores = rf_selector.feature_importances_
        
        # Combine scores based on method
        if self.selection_method == 'mutual_info':
            scores = mi_scores
        elif self.selection_method == 'random_forest':
            scores = rf_scores
        else:  # combined
            # Normalize and combine scores
            mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
            rf_scores = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min())
            scores = (mi_scores + rf_scores) / 2
            
        # Create importance dictionary
        for feature, score in zip(X.columns, scores):
            importances[feature] = score
            
        # Sort features by importance
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        self.feature_importances_ = dict(sorted_features)
        self.selected_features_ = [f[0] for f in sorted_features[:self.n_features]]
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting only the chosen features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with only selected features
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        return X[self.selected_features_]
        
    def get_feature_importance_plot(self) -> dict:
        """
        Get feature importance data for plotting.
        
        Returns:
            Dict with feature names and their importance scores
        """
        if self.feature_importances_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        return {
            'features': list(self.feature_importances_.keys()),
            'importances': list(self.feature_importances_.values())
        }