import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any, Union, List

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data for correctness and completeness."""
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the structure of a CSV DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_columns": []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            results["valid"] = False
            results["errors"].append("DataFrame is empty")
            return results
            
        # Check for required columns
        required_columns = ["Date", "Number"]
        for col in required_columns:
            if col not in df.columns:
                results["missing_columns"].append(col)
                results["warnings"].append(f"Missing required column: {col}")
                
        if results["missing_columns"]:
            results["valid"] = False
            
        # Check for data types
        if "Date" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
                results["warnings"].append("Date column is not datetime type")
                
        if "Number" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["Number"]):
                results["warnings"].append("Number column is not numeric type")
                
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            for col, count in missing_counts.items():
                if count > 0:
                    results["warnings"].append(f"Column {col} has {count} missing values")
                    
        return results
    
    @staticmethod
    def validate_date_sequence(dates: pd.Series) -> Dict[str, Any]:
        """
        Validate the sequence of dates for consistency.
        
        Args:
            dates: Series of dates to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check if dates are sorted
        if not dates.equals(dates.sort_values()):
            results["warnings"].append("Dates are not in chronological order")
            
        # Check for duplicates
        duplicates = dates.duplicated()
        if duplicates.any():
            results["warnings"].append(f"Found {duplicates.sum()} duplicate dates")
            
        # Analyze date differences
        if len(dates) > 1:
            date_diffs = dates.diff().dropna()
            
            # Calculate statistics
            results["stats"]["min_diff"] = date_diffs.min().days
            results["stats"]["max_diff"] = date_diffs.max().days
            results["stats"]["mean_diff"] = date_diffs.mean().days
            
            # Count occurrences of each difference
            diff_counts = date_diffs.dt.days.value_counts().sort_index()
            results["stats"]["diff_counts"] = diff_counts.to_dict()
            
            # Identify most common intervals
            most_common = diff_counts.nlargest(3)
            results["stats"]["most_common_intervals"] = most_common.to_dict()
            
        return results


class DataLoader:
    """
    Utility class for loading and preprocessing data with robust error handling.
    """
    def __init__(self, data_dir: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataLoader with data directory path and optional configuration.
        
        Args:
            data_dir: Path to the data directory
            config: Optional configuration parameters
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize validator
        self.validator = DataValidator()
        
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across the codebase.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
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
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV data from the data directory with robust error handling.
        
        Args:
            filename: Name of the CSV file to load
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame containing the loaded data
        """
        file_path = self.data_dir / filename
        logger.info(f"Loading CSV file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File {file_path} does not exist.")
            return pd.DataFrame()
        
        try:
            # Try standard pandas read_csv
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded CSV with shape: {df.shape}")
            
            # Standardize column names
            df = self.standardize_column_names(df)
            return df
        except Exception as e:
            logger.error(f"Error loading CSV with pandas: {str(e)}")
            logger.info("Attempting manual CSV loading...")
            
            # Try manual loading as fallback
            rows = []
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip().split(',')
                    for line in f:
                        try:
                            values = line.strip().split(',')
                            if len(values) == len(header):
                                rows.append(dict(zip(header, values)))
                            else:
                                logger.warning(f"Skipping malformed line: {line}")
                        except Exception as line_e:
                            logger.warning(f"Error processing line: {str(line_e)}")
                
                df = pd.DataFrame(rows)
                df = self.standardize_column_names(df)
                logger.info(f"Manual loading successful. Shape: {df.shape}")
                return df
            except Exception as inner_e:
                logger.error(f"Manual loading also failed: {str(inner_e)}")
                # Return empty DataFrame as last resort
                return pd.DataFrame()
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for analysis with robust error handling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for preprocessing.")
            return df
        
        logger.info("Preprocessing data...")
        
        # Standardize column names
        df = self.standardize_column_names(df)
        
        try:
            # Convert 'Date' column to datetime if it exists
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
            # Convert 'Number' column to numeric if it exists
            if 'Number' in df.columns:
                df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
                
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Validate the preprocessed data
            validation_results = self.validator.validate_csv_structure(df)
            if not validation_results["valid"]:
                for warning in validation_results["warnings"]:
                    logger.warning(f"Data validation warning: {warning}")
                for error in validation_results["errors"]:
                    logger.error(f"Data validation error: {error}")
            
            # Return sorted DataFrame if Date column exists
            if 'Date' in df.columns:
                return df.sort_values('Date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            # Return the original DataFrame if preprocessing fails
            return df
            
    def load_and_preprocess(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load and preprocess a CSV file in one step.
        
        Args:
            filename: Name of the CSV file to load
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.load_csv(filename, **kwargs)
        return self.preprocess_data(df)
    
    def get_date_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the date sequence for patterns and intervals.
        
        Args:
            df: DataFrame containing a 'Date' column
            
        Returns:
            Dictionary with date statistics
        """
        if 'Date' not in df.columns:
            logger.warning("No 'Date' column found in DataFrame")
            return {}
            
        dates = df['Date'].sort_values()
        return self.validator.validate_date_sequence(dates)
    
    def generate_synthetic_data(self, num_rows: int = 100, start_date: str = '2020-01-01', 
                              max_number: int = 10, min_number: int = 1) -> pd.DataFrame:
        """
        Generate synthetic data for testing.
        
        Args:
            num_rows: Number of rows to generate
            start_date: Starting date for the sequence
            max_number: Maximum random number
            min_number: Minimum random number
            
        Returns:
            DataFrame with synthetic data
        """
        dates = pd.date_range(start=start_date, periods=num_rows, freq='D')
        numbers = np.random.randint(min_number, max_number + 1, size=num_rows)
        
        return pd.DataFrame({
            'Date': dates,
            'Number': numbers
        })

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader("data")
    df = loader.load_csv("historical_random_numbers.csv")
    processed_df = loader.preprocess_data(df)
    print(processed_df.head())