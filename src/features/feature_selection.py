"""
Feature selection module for identifying and selecting the most relevant features.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Advanced feature selection for random number prediction.
    
    This class implements multiple selection methods and can combine their results
    to identify the most valuable features for prediction tasks.
    """
    def __init__(
        self,
        n_features: int = 50,
        selection_method: str = 'ensemble',
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        Initialize the FeatureSelector.
        
        Args:
            n_features: Number of features to select
            selection_method: Selection method to use ('ensemble', 'mutual_info', 
                              'random_forest', 'lasso', 'permutation')
            random_state: Random state for reproducibility
            verbose: Whether to print verbose output
        """
        self.n_features = n_features
        self.selection_method = selection_method
        self.random_state = random_state
        self.verbose = verbose
        
        # Will be filled during fit
        self.selected_features_: Optional[List[str]] = None
        self.feature_importances_: Optional[Dict[str, float]] = None
        self.method_importances_: Dict[str, Dict[str, float]] = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector to the data.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            self: The fitted selector
        """
        if self.verbose:
            print(f"Fitting FeatureSelector with method: {self.selection_method}")
            print(f"Input shape: {X.shape}")
        
        # Validate selection method
        valid_methods = ['ensemble', 'mutual_info', 'random_forest', 'lasso', 'permutation', 'combined']
        if self.selection_method not in valid_methods:
            raise ValueError(f"Invalid selection method. Choose from: {valid_methods}")
        
        # For backward compatibility
        if self.selection_method == 'combined':
            self.selection_method = 'ensemble'
            
        # Calculate feature importances with each method
        if self.selection_method == 'ensemble' or self.verbose:
            # Calculate importances using all methods for ensemble or verbose mode
            self._calculate_mutual_info_importance(X, y)
            self._calculate_random_forest_importance(X, y)
            self._calculate_lasso_importance(X, y)
            self._calculate_permutation_importance(X, y)
            
            if self.selection_method == 'ensemble':
                # Combine importances from all methods
                self._combine_importances()
        else:
            # Calculate importances using only the specified method
            if self.selection_method == 'mutual_info':
                self._calculate_mutual_info_importance(X, y)
            elif self.selection_method == 'random_forest':
                self._calculate_random_forest_importance(X, y)
            elif self.selection_method == 'lasso':
                self._calculate_lasso_importance(X, y)
            elif self.selection_method == 'permutation':
                self._calculate_permutation_importance(X, y)
                
            # Use the single method's importances
            self.feature_importances_ = self.method_importances_[self.selection_method]
        
        # Select top features
        self._select_top_features()
        
        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features")
            print(f"Top 10 features: {self.selected_features_[:10]}")
        
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
            
        # Handle missing columns gracefully
        available_features = [f for f in self.selected_features_ if f in X.columns]
        
        if len(available_features) < len(self.selected_features_):
            missing_count = len(self.selected_features_) - len(available_features)
            if self.verbose:
                print(f"Warning: {missing_count} selected features not found in input data")
        
        if not available_features:
            raise ValueError("None of the selected features are present in the input data")
            
        return X[available_features]
    
    def _calculate_mutual_info_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculate feature importance using mutual information.
        
        Mutual information measures the dependency between features and the target,
        without assuming linearity.
        """
        if self.verbose:
            print("Calculating mutual information importances...")
        
        # Handle potential errors in mutual information calculation
        try:
            mi_selector = SelectKBest(mutual_info_regression, k='all')
            mi_selector.fit(X, y)
            mi_scores = mi_selector.scores_
            
            # Create importance dictionary
            importances = {}
            for feature, score in zip(X.columns, mi_scores):
                importances[feature] = score
                
            self.method_importances_['mutual_info'] = importances
            
        except Exception as e:
            if self.verbose:
                print(f"Error in mutual information calculation: {str(e)}")
            # Initialize with zero importances if calculation fails
            self.method_importances_['mutual_info'] = {feature: 0.0 for feature in X.columns}
    
    def _calculate_random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculate feature importance using Random Forest.
        
        Uses the built-in feature importance from Random Forest, which considers
        the contribution of each feature to decreasing impurity across all trees.
        """
        if self.verbose:
            print("Calculating Random Forest importances...")
            
        try:
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X, y)
            rf_scores = rf.feature_importances_
            
            # Create importance dictionary
            importances = {}
            for feature, score in zip(X.columns, rf_scores):
                importances[feature] = score
                
            self.method_importances_['random_forest'] = importances
            
        except Exception as e:
            if self.verbose:
                print(f"Error in Random Forest importance calculation: {str(e)}")
            self.method_importances_['random_forest'] = {feature: 0.0 for feature in X.columns}
    
    def _calculate_lasso_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculate feature importance using Lasso regression.
        
        Lasso performs feature selection by shrinking less important feature 
        coefficients toward zero.
        """
        if self.verbose:
            print("Calculating Lasso importances...")
            
        try:
            # Normalize data for Lasso
            X_normalized = (X - X.mean()) / (X.std() + 1e-10)
            
            # Fit Lasso with automatic alpha selection
            lasso = Lasso(alpha=0.01, random_state=self.random_state, max_iter=10000)
            lasso.fit(X_normalized, y)
            
            # Use absolute coefficient values as importance
            lasso_scores = np.abs(lasso.coef_)
            
            # Create importance dictionary
            importances = {}
            for feature, score in zip(X.columns, lasso_scores):
                importances[feature] = score
                
            self.method_importances_['lasso'] = importances
            
        except Exception as e:
            if self.verbose:
                print(f"Error in Lasso importance calculation: {str(e)}")
            self.method_importances_['lasso'] = {feature: 0.0 for feature in X.columns}
    
    def _calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculate feature importance using permutation importance.
        
        Permutation importance measures the decrease in model performance when
        a single feature's values are randomly shuffled.
        """
        if self.verbose:
            print("Calculating permutation importances...")
            
        try:
            # Train a random forest model for permutation importance
            rf = RandomForestRegressor(
                n_estimators=50,  # Smaller ensemble for speed
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X, y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                rf, X, y, 
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Use mean importance as the score
            perm_scores = perm_importance.importances_mean
            
            # Create importance dictionary
            importances = {}
            for feature, score in zip(X.columns, perm_scores):
                importances[feature] = score
                
            self.method_importances_['permutation'] = importances
            
        except Exception as e:
            if self.verbose:
                print(f"Error in permutation importance calculation: {str(e)}")
            self.method_importances_['permutation'] = {feature: 0.0 for feature in X.columns}
    
    def _combine_importances(self) -> None:
        """
        Combine feature importances from all methods.
        
        For the ensemble approach, this normalizes and combines the importance
        scores from all methods.
        """
        if self.verbose:
            print("Combining importances from all methods...")
            
        all_features = set()
        for method_imps in self.method_importances_.values():
            all_features.update(method_imps.keys())
            
        # Initialize combined importance dict
        combined_importances = {feature: 0.0 for feature in all_features}
        
        # Normalize and add each method's importances
        for method, importances in self.method_importances_.items():
            # Skip empty importance sets
            if not importances:
                continue
                
            # Get values for normalization
            values = np.array(list(importances.values()))
            min_val = values.min()
            max_val = values.max()
            max_val = max_val if max_val > min_val else min_val + 1.0
            
            # Normalize and add to combined importances
            for feature, score in importances.items():
                normalized_score = (score - min_val) / (max_val - min_val)
                combined_importances[feature] += normalized_score
                
        # Divide by number of methods for final score
        n_methods = len(self.method_importances_)
        if n_methods > 0:
            for feature in combined_importances:
                combined_importances[feature] /= n_methods
                
        self.feature_importances_ = combined_importances
    
    def _select_top_features(self) -> None:
        """
        Select the top N features based on importance scores.
        """
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top N features
        n = min(self.n_features, len(sorted_features))
        self.selected_features_ = [f[0] for f in sorted_features[:n]]
        
        # Update feature_importances_ to contain only selected features
        self.feature_importances_ = {
            k: v for k, v in sorted_features[:n]
        }
    
    def get_feature_importance_plot(self) -> dict:
        """
        Get feature importance data for plotting.
        
        Returns:
            Dict with feature names and their importance scores
        """
        if self.feature_importances_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        
        # Sort features by importance for plotting
        sorted_items = sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'features': [item[0] for item in sorted_items],
            'importances': [item[1] for item in sorted_items]
        }
        
    def get_method_comparison(self) -> dict:
        """
        Get comparison data for different feature selection methods.
        
        Returns:
            Dict with comparison data for different methods
        """
        if not self.method_importances_:
            raise RuntimeError("No method importances calculated")
            
        # Get the top 10 features from each method
        method_top_features = {}
        
        for method, importances in self.method_importances_.items():
            sorted_features = sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            method_top_features[method] = [f[0] for f in sorted_features[:10]]
            
        return {
            'method_top_features': method_top_features,
            'overlap_with_ensemble': {
                method: len(set(features) & set(self.selected_features_[:10]))
                for method, features in method_top_features.items()
            }
        }
    
    """
    Feature selection utility for NRNI.
    Provides methods for selecting the most relevant features.
    """
    def __init__(self, method: str = 'correlation', threshold: float = 0.1) -> None:
        """
        Initialize the FeatureSelector.
        
        Args:
            method (str): Feature selection method ('correlation', 'model', etc.)
            threshold (float): Threshold for feature selection
        """
        self.method = method
        self.threshold = threshold

    def select(self, df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """
        Select features from the DataFrame based on the chosen method.
        
        Args:
            df (pd.DataFrame): Input data
            target (Optional[str]): Target column for feature selection
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        if self.method == 'correlation' and target and target in df.columns:
            corr = df.corr(numeric_only=True)[target].abs()
            selected = corr[corr > self.threshold].index.tolist()
            return df[selected]
        # Add other selection methods as needed
        return df