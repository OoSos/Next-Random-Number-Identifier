import pandas as pd
import numpy as np
from pathlib import Path
import logging

class SimpleDataLoader:
    """
    Simplified data loader with robust error handling for CSV files.
    """
    def __init__(self, data_dir):
        """
        Initialize the data loader with a directory path.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure the directory exists
        if not self.data_dir.exists():
            self.logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, filename):
        """
        Load data from a CSV file with robust error handling.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            DataFrame containing the loaded data
        """
        file_path = self.data_dir / filename
        self.logger.info(f"Attempting to load CSV from: {file_path}")
        
        if not file_path.exists():
            self.logger.error(f"File {file_path} does not exist.")
            return pd.DataFrame()
        
        try:
            # Try standard pandas read_csv
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded CSV with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV with pandas: {str(e)}")
            self.logger.info("Attempting manual CSV loading...")
            
            # Fallback to manual CSV loading
            rows = []
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip().split(',')
                    for line in f:
                        values = line.strip().split(',')
                        if len(values) == len(header):
                            rows.append(dict(zip(header, values)))
                        else:
                            self.logger.warning(f"Skipping malformed line: {line}")
                
                df = pd.DataFrame(rows)
                self.logger.info(f"Manual loading successful. Shape: {df.shape}")
                return df
            except Exception as inner_e:
                self.logger.error(f"Manual loading also failed: {str(inner_e)}")
                # Return empty DataFrame as last resort
                return pd.DataFrame()
    
    def preprocess_data(self, df):
        """
        Preprocess the loaded data for analysis.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for preprocessing.")
            return df
        
        # Standardize column names
        column_mapping = {
            'Super Ball': 'Number',
            'super ball': 'Number', 
            'super_ball': 'Number',
            'superball': 'Number',
            'SUPER BALL': 'Number',
            'Ball': 'Number'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert Date to datetime if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        # Convert Number to numeric if it exists    
        if 'Number' in df.columns:
            df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
            
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
            
        return df.sort_values('Date') if 'Date' in df.columns else df

# Example usage
if __name__ == "__main__":
    loader = SimpleDataLoader("data")
    df = loader.load_csv("historical_random_numbers.csv")
    processed_df = loader.preprocess_data(df)
    print(processed_df.head())