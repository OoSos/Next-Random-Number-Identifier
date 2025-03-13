# src/utils/__init__.py

from .data_loader import DataLoader
from .evaluation import ModelEvaluator
from .monitoring import ModelMonitor
from .file_utils import debug_file_path
import pandas as pd

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across the codebase.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    column_mapping = {
        'Super Ball': 'Number',
        'super ball': 'Number',
        'super_ball': 'Number',
        'superball': 'Number',
        'SUPER BALL': 'Number',
        'Ball': 'Number'
    }
    
    # Apply mapping to rename columns
    return df.rename(columns=column_mapping)

__all__ = [
    'DataLoader',
    'ModelEvaluator',
    'ModelMonitor',
    'debug_file_path',
    'standardize_column_names',
]