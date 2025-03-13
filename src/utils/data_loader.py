import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
from .. import utils  # Import the utils module to access standardize_column_names

class DataLoader:
    """
    Utility class for loading and preprocessing data.
    """
    def __init__(self, data_dir: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoader with data directory path and optional configuration.
        
        Args:
            data_dir (str): Path to the data directory
            config (Optional[Dict[str, Any]]): Configuration parameters
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across the codebase.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        return utils.standardize_column_names(df)
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load CSV data from the data directory.
        
        Args:
            filename (str): Name of the CSV file to load
            
        Returns:
            pd.DataFrame: Loaded and standardized DataFrame
        """
        file_path = self.data_dir / filename
        self.logger.info(f"Loading CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            # Standardize column names
            df = self.standardize_column_names(df)
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for analysis.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Convert 'Number' column to numeric if it exists
        if 'Number' in df.columns:
            df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
            
        # Convert 'Date' column to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df