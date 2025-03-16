import pandas as pd
from pathlib import Path
import logging

class SimpleDataLoader:
    """
    Simplified data loader to ensure functionality.
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
    def load_csv(self, filename):
        """Load data from CSV file"""
        file_path = self.data_dir / filename
        
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded {file_path} with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            return pd.DataFrame()
        
    def preprocess_data(self, df):
        """Basic data preprocessing"""
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        # Standardize column names
        column_mapping = {
            'Super Ball': 'Number',
            'super ball': 'Number',
            'super_ball': 'Number',
            'superball': 'Number',
            'SUPER BALL': 'Number',
            'Ball': 'Number'
        }
        
        return df.rename(columns=column_mapping).sort_values('Date') if 'Date' in df.columns else df
