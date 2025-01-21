import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

class DataLoader:
    """
    Utility class for loading and preprocessing data.
    """
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename (str): Name of the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.data_dir / filename
        return pd.read_csv(file_path, **kwargs)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data preprocessing.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Convert date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        return df.sort_values('Date') if 'Date' in df.columns else df
    
    def split_data(self, 
                  df: pd.DataFrame, 
                  target_col: str,
                  test_size: float = 0.2,
                  shuffle: bool = False,
                  random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_col (str): Name of the target column
            test_size (float): Proportion of data to use for testing
            shuffle (bool): Whether to shuffle the data before splitting
            random_state (Optional[int]): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
                X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return train_test_split(X, y, 
                              test_size=test_size, 
                              shuffle=shuffle, 
                              random_state=random_state)
    
    def load_and_prepare_data(self, 
                            filename: str, 
                            target_col: str,
                            test_size: float = 0.2,
                            **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load, preprocess, and split data in one step.
        
        Args:
            filename (str): Name of the CSV file
            target_col (str): Name of the target column
            test_size (float): Proportion of data to use for testing
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
                X_train, X_test, y_train, y_test
        """
        df = self.load_csv(filename, **kwargs)
        df = self.preprocess_data(df)
        return self.split_data(df, target_col, test_size)