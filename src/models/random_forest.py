# Standard library imports
from typing import Dict, Any, Optional, Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.validation import check_is_fitted
import warnings

# Local application imports
from src.models.base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest model for regression and feature importance analysis.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs) -> None:
        """
        Initialize the RandomForestModel.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (Optional[int]): Maximum depth of the tree
            min_samples_leaf (int): Minimum samples required at a leaf node
            random_state (int): Random seed
            **kwargs: Additional model parameters (must be valid for RandomForestRegressor)
        
        Raises:
            ValueError: If any provided kwarg is not a valid RandomForestRegressor parameter or has invalid type/value.
        """
        super().__init__(name="RandomForest")
        rf_defaults = RandomForestRegressor()
        valid_params = rf_defaults.get_params().keys()
        # Known param types from sklearn
        bool_params = {'bootstrap', 'oob_score', 'warm_start'}
        str_params = {'criterion'}
        int_params = {'n_estimators', 'min_samples_leaf', 'random_state', 'max_depth', 'min_samples_split', 'min_weight_fraction_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'max_samples'}
        allowed_criteria = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
        params = {}
        params['n_estimators'] = int(n_estimators)
        params['min_samples_leaf'] = int(min_samples_leaf)
        params['random_state'] = int(random_state)
        if max_depth is not None:
            params['max_depth'] = int(max_depth)
        # Handle kwargs with type checks
        for k, v in kwargs.items():
            if k not in valid_params:
                warnings.warn(f"Parameter '{k}' is not valid for RandomForestRegressor and will be ignored.")
                continue
            if k in bool_params:
                if not isinstance(v, bool):
                    warnings.warn(f"Parameter '{k}' must be bool, got {type(v).__name__}. Ignoring.")
                    continue
                params[k] = v
            elif k in str_params:
                if not isinstance(v, str):
                    warnings.warn(f"Parameter '{k}' must be str, got {type(v).__name__}. Ignoring.")
                    continue
                if k == 'criterion' and v not in allowed_criteria:
                    warnings.warn(f"Invalid value for criterion: {v}. Must be one of {allowed_criteria}. Ignoring.")
                    continue
                params[k] = v
            elif k in int_params:
                try:
                    params[k] = int(v)
                except Exception:
                    warnings.warn(f"Parameter '{k}' must be int-compatible, got {type(v).__name__}. Ignoring.")
            else:
                # Use the type from the default RandomForestRegressor instance
                default_type = type(rf_defaults.get_params()[k])
                if not isinstance(v, default_type) and v is not None:
                    warnings.warn(f"Parameter '{k}' must be {default_type.__name__}, got {type(v).__name__}. Ignoring.")
                    continue
                params[k] = v
        self.params = params
        self.model = RandomForestRegressor(**self.params)
        self.performance_metrics: Dict[str, float] = {}
        self.feature_importance_: Optional[Dict[str, float]] = None

    def preprocess(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocess input data for Random Forest model.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Preprocessed features and target
        
        Warns:
            UserWarning: If non-numeric columns are dropped from X.
        """
        orig_cols = set(X.columns)
        X_num = X.select_dtypes(include=[np.number])
        dropped = orig_cols - set(X_num.columns)
        if dropped:
            warnings.warn(f"Non-numeric columns dropped in preprocessing: {dropped}")
        X_num = X_num.fillna(X_num.mean())
        return X_num, y

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """
        Fit the Random Forest model to the training data.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target values
            
        Returns:
            RandomForestModel: The fitted model instance
        
        Raises:
            ValueError: If y is None.
        """
        X_processed, y_processed = self.preprocess(X, y)
        X_processed = X_processed.astype(float)
        if y_processed is not None:
            y_processed = y_processed.astype(float)
        else:
            raise ValueError("Target variable y must not be None during fit.")
        self.model.fit(X_processed, y_processed)
        self.feature_importance_ = dict(zip(X_processed.columns, self.model.feature_importances_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted Random Forest model.
        
        Args:
            X (pd.DataFrame): Features to make predictions for
            
        Returns:
            np.ndarray: Predicted values
        
        Raises:
            ValueError: If model is not fitted.
        """
        check_is_fitted(self.model)
        X_processed, _ = self.preprocess(X)
        X_processed = X_processed.astype(float)
        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get uncertainty estimates for predictions using the fitted Random Forest model.
        For regression, this returns the standard deviation of individual tree predictions.
        
        Args:
            X (pd.DataFrame): Features to make predictions for
            
        Returns:
            np.ndarray: Standard deviation of predictions across trees (uncertainty measure)
        
        Raises:
            ValueError: If model is not fitted.
        """
        check_is_fitted(self.model)
        X_processed, _ = self.preprocess(X)
        X_processed = X_processed.astype(float)
        predictions = np.array([tree.predict(X_processed) for tree in self.model.estimators_])
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
        X_processed, y_processed = self.preprocess(X, y)
        predictions = self.predict(X_processed)
        mse = mean_squared_error(y_processed, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_processed, predictions)
        r2 = r2_score(y_processed, predictions)
        self.performance_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        return self.performance_metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the Random Forest model.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        
        Raises:
            ValueError: If model is not fitted.
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.feature_importance_

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model's current state and configuration.
        
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
        check_is_fitted(self.model)
        if not hasattr(self.model, 'oob_score_'):
            raise ValueError("OOB score not available. Enable oob_score=True during model initialization.")
        return float(self.model.oob_score_)

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            param_grid (Dict[str, List[Any]]): Grid of hyperparameters to search
            
        Returns:
            Dict[str, Any]: Best hyperparameters
        """
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        self.params.update(grid_search.best_params_)
        self.model = RandomForestRegressor(**self.params)
        self.fit(X, y)
        return grid_search.best_params_

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence using the Random Forest model.
        For regression, returns the inverse of prediction std deviation (normalized).
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
            
        Returns:
            np.ndarray: Confidence estimates for each prediction
        
        Raises:
            ValueError: If model is not fitted.
        """
        check_is_fitted(self.model)
        X_processed, _ = self.preprocess(X)
        X_processed = X_processed.astype(float)
        predictions = np.array([tree.predict(X_processed) for tree in self.model.estimators_])
        stds = np.std(predictions, axis=0)
        max_std = np.max(stds) if np.max(stds) > 0 else 1.0
        confidence = 1.0 - (stds / max_std)
        return confidence