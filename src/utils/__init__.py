# src/utils/__init__.py

from .enhanced_data_loader import EnhancedDataLoader, DataSchemaValidator
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
    # Use the new EnhancedDataLoader's implementation
    loader = EnhancedDataLoader(".")  # Directory doesn't matter for this use
    return loader.standardize_column_names(df)

__all__ = [
    'EnhancedDataLoader',
    'DataSchemaValidator',
    'ModelEvaluator',
    'ModelMonitor',
    'debug_file_path',
    'standardize_column_names',
]