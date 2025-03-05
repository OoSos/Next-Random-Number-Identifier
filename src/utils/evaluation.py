import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    """
    Utility class for evaluating model performance with various metrics.
    Supports both regression and classification tasks.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        pass
        
    def evaluate_regression(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dict with regression metrics
        """
        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
        }
        
        return metrics
        
    def evaluate_classification(self, 
                              y_true: pd.Series, 
                              y_pred: np.ndarray,
                              average: str = 'weighted') -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            average: Method for averaging metrics ('micro', 'macro', 'weighted')
            
        Returns:
            Dict with classification metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }
        
        return metrics
        
    def compare_models(self, 
                      y_true: pd.Series, 
                      predictions: Dict[str, np.ndarray],
                      task_type: str = 'regression') -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models' performance.
        
        Args:
            y_true: True target values
            predictions: Dict mapping model names to predictions
            task_type: Either 'regression' or 'classification'
            
        Returns:
            Dict with metrics for each model
        """
        comparison = {}
        
        for model_name, y_pred in predictions.items():
            if task_type.lower() == 'regression':
                metrics = self.evaluate_regression(y_true, y_pred)
            elif task_type.lower() == 'classification':
                metrics = self.evaluate_classification(y_true, y_pred)
            else:
                raise ValueError("task_type must be 'regression' or 'classification'")
                
            comparison[model_name] = metrics
            
        return comparison
    
    def generate_report(self, 
                       comparison: Dict[str, Dict[str, float]],
                       sort_by: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a formatted comparison report.
        
        Args:
            comparison: Dict with metrics for each model
            sort_by: Optional metric name to sort results by
            
        Returns:
            DataFrame with formatted comparison
        """
        report = pd.DataFrame(comparison).T
        
        if sort_by is not None and sort_by in report.columns:
            report = report.sort_values(sort_by, ascending=False)
            
        return report