import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger(__name__)

class DataSchemaValidator:
    """Validates data schema, structure, and content integrity."""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame, 
        required_columns: List[str] = ["Date", "Number"],
        date_column: str = "Date",
        value_column: str = "Number"
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of columns that must be present
            date_column: Name of the date column
            value_column: Name of the value column
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_columns": [],
            "data_quality": {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            results["valid"] = False
            results["errors"].append("DataFrame is empty")
            return results
            
        # Check for required columns
        for col in required_columns:
            if col not in df.columns:
                results["missing_columns"].append(col)
                results["warnings"].append(f"Missing required column: {col}")
                
        if results["missing_columns"]:
            results["valid"] = False
            
        # Check data types if columns exist
        if date_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                results["warnings"].append(f"{date_column} column is not datetime type")
                
        if value_column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                results["warnings"].append(f"{value_column} column is not numeric type")
                
        # Check for missing values and calculate data quality metrics
        if not df.empty:
            total_cells = df.size
            missing_cells = df.isna().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            results["data_quality"] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "total_cells": total_cells,
                "missing_cells": missing_cells,
                "missing_percentage": missing_percentage,
                "column_stats": {}
            }
            
            # Column-specific statistics
            for col in df.columns:
                col_missing = df[col].isna().sum()
                col_missing_pct = (col_missing / len(df)) * 100
                
                results["data_quality"]["column_stats"][col] = {
                    "missing_count": int(col_missing),
                    "missing_percentage": float(col_missing_pct),
                    "dtype": str(df[col].dtype)
                }
                
                if col_missing > 0:
                    if col_missing_pct > 50:
                        results["errors"].append(f"Column {col} has {col_missing_pct:.1f}% missing values")
                    else:
                        results["warnings"].append(f"Column {col} has {col_missing_pct:.1f}% missing values")
        
        return results
    
    @staticmethod
    def validate_date_sequence(
        dates: pd.Series, 
        min_date: Optional[pd.Timestamp] = None,
        max_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Validate a sequence of dates for chronological order and consistency.
        
        Args:
            dates: Series of dates to validate
            min_date: Optional minimum acceptable date
            max_date: Optional maximum acceptable date
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        if dates.empty:
            results["valid"] = False
            results["errors"].append("Date series is empty")
            return results
            
        # Check if dates are sorted
        is_sorted = dates.equals(dates.sort_values())
        results["stats"]["is_chronological"] = is_sorted
        if not is_sorted:
            results["warnings"].append("Dates are not in chronological order")
            
        # Check for duplicates
        duplicates = dates.duplicated()
        duplicate_count = duplicates.sum()
        results["stats"]["duplicate_dates"] = int(duplicate_count)
        if duplicate_count > 0:
            results["warnings"].append(f"Found {duplicate_count} duplicate dates")
            
        # Check date range
        min_observed = dates.min()
        max_observed = dates.max()
        results["stats"]["min_date"] = min_observed
        results["stats"]["max_date"] = max_observed
        results["stats"]["date_range_days"] = (max_observed - min_observed).days
        
        # Validate against provided bounds
        if min_date is not None and min_observed < min_date:
            results["errors"].append(f"Dates before minimum allowed date ({min_date})")
            results["valid"] = False
            
        if max_date is not None and max_observed > max_date:
            results["errors"].append(f"Dates after maximum allowed date ({max_date})")
            results["valid"] = False
            
        # Analyze date differences
        if len(dates) > 1:
            date_diffs = dates.sort_values().diff().dropna()
            
            # Calculate statistics
            results["stats"]["min_diff_days"] = int(date_diffs.dt.days.min())
            results["stats"]["max_diff_days"] = int(date_diffs.dt.days.max())
            results["stats"]["mean_diff_days"] = float(date_diffs.dt.days.mean())
            results["stats"]["median_diff_days"] = int(date_diffs.dt.days.median())
            
            # Find gaps in the sequence
            large_gaps = date_diffs[date_diffs.dt.days > 7]
            if not large_gaps.empty:
                results["stats"]["large_gaps"] = len(large_gaps)
                results["warnings"].append(f"Found {len(large_gaps)} large gaps (>7 days) in date sequence")
            
            # Count occurrences of each difference
            diff_counts = date_diffs.dt.days.value_counts().sort_index()
            results["stats"]["interval_distribution"] = diff_counts.to_dict()
            
            # Identify most common intervals
            most_common = diff_counts.nlargest(3)
            results["stats"]["most_common_intervals"] = most_common.to_dict()
            
        return results
    
    @staticmethod
    def validate_number_sequence(
        numbers: pd.Series,
        expected_min: Optional[int] = None,
        expected_max: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate a sequence of numbers for range, distribution, and potential anomalies.
        
        Args:
            numbers: Series of numbers to validate
            expected_min: Optional minimum expected value
            expected_max: Optional maximum expected value
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        if numbers.empty:
            results["valid"] = False
            results["errors"].append("Number series is empty")
            return results
            
        # Basic statistics
        observed_min = numbers.min()
        observed_max = numbers.max()
        results["stats"]["min"] = float(observed_min)
        results["stats"]["max"] = float(observed_max)
        results["stats"]["mean"] = float(numbers.mean())
        results["stats"]["median"] = float(numbers.median())
        results["stats"]["std"] = float(numbers.std())
        
        # Validate against expected range
        if expected_min is not None and observed_min < expected_min:
            results["errors"].append(f"Values below minimum expected value ({expected_min})")
            results["valid"] = False
            
        if expected_max is not None and observed_max > expected_max:
            results["errors"].append(f"Values above maximum expected value ({expected_max})")
            results["valid"] = False
            
        # Check for outliers using IQR method
        q1 = numbers.quantile(0.25)
        q3 = numbers.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        outliers = numbers[(numbers < lower_bound) | (numbers > upper_bound)]
        outlier_count = len(outliers)
        
        results["stats"]["outliers"] = int(outlier_count)
        if outlier_count > 0:
            results["warnings"].append(f"Found {outlier_count} outliers using IQR method")
            
        # Distribution analysis
        value_counts = numbers.value_counts().sort_index()
        results["stats"]["value_distribution"] = value_counts.to_dict()
        
        # Chi-square test for uniformity (if applicable for random numbers)
        # Only apply if we have expected bounds
        if expected_min is not None and expected_max is not None:
            unique_values = sorted(numbers.unique())
            observed = [value_counts.get(val, 0) for val in unique_values]
            n = len(numbers)
            k = len(unique_values)
            expected = [n/k] * k  # Uniform distribution
            
            # Compute chi-square statistic
            chi2 = sum([(o - e)**2 / e for o, e in zip(observed, expected)])
            results["stats"]["chi2_uniformity"] = float(chi2)
            results["stats"]["chi2_p_value"] = None  # Would need scipy to compute p-value
            
        return results


class EnhancedDataLoader:
    """
    Enhanced data loader with comprehensive error handling, validation, and preprocessing capabilities.
    """
    def __init__(
        self, 
        data_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        create_if_missing: bool = True
    ):
        """
        Initialize the EnhancedDataLoader with configuration options.
        
        Args:
            data_dir: Directory containing data files
            config: Optional configuration dictionary
            create_if_missing: Whether to create the data directory if it doesn't exist
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize validator
        self.validator = DataSchemaValidator()
        
        # Create directory if it doesn't exist and create_if_missing is True
        if not self.data_dir.exists():
            if create_if_missing:
                self.logger.info(f"Creating data directory: {self.data_dir}")
                os.makedirs(self.data_dir, exist_ok=True)
            else:
                self.logger.warning(f"Data directory does not exist: {self.data_dir}")
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to ensure consistency across different data sources.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Standard mappings for common column names
        column_mapping = {
            'Super Ball': 'Number',
            'super ball': 'Number',
            'super_ball': 'Number',
            'superball': 'Number',
            'SUPER BALL': 'Number',
            'Ball': 'Number',
            'ball': 'Number',
            'NUM': 'Number',
            'Value': 'Number',
            'value': 'Number',
            
            'DATE': 'Date',
            'date': 'Date',
            'DateTime': 'Date',
            'Timestamp': 'Date',
            'Time': 'Date',
            'time': 'Date'
        }
        
        # Apply mappings
        return df.rename(columns=column_mapping)
    
    def load_file(
        self, 
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data from a file with format auto-detection and error handling.
        
        Args:
            file_path: Path to the file
            file_format: Optional file format to override auto-detection
            **kwargs: Additional arguments to pass to the loader function
            
        Returns:
            Tuple of (DataFrame, metadata dictionary)
        """
        file_path = Path(file_path)
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
            'file_info': {
                'path': str(file_path),
                'exists': file_path.exists(),
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'format': file_format or file_path.suffix.lower()[1:]
            }
        }
        
        # Check if file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            return pd.DataFrame(), results
        
        # Auto-detect format if not provided
        if file_format is None:
            file_format = file_path.suffix.lower()[1:]
            if not file_format:
                error_msg = f"Could not determine file format for: {file_path}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                return pd.DataFrame(), results
                
        # Load based on format
        try:
            if file_format in ['csv', 'txt']:
                df = self._load_csv(file_path, **kwargs)
            elif file_format in ['xlsx', 'xls']:
                df = self._load_excel(file_path, **kwargs)
            elif file_format == 'json':
                df = self._load_json(file_path, **kwargs)
            else:
                error_msg = f"Unsupported file format: {file_format}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                return pd.DataFrame(), results
                
            # Update results
            results['success'] = True
            results['file_info']['row_count'] = len(df)
            results['file_info']['column_count'] = len(df.columns)
            
            # Standardize column names
            df = self.standardize_column_names(df)
            
            return df, results
            
        except Exception as e:
            error_msg = f"Error loading file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            
            # Try fallback loading if it's a CSV
            if file_format in ['csv', 'txt']:
                self.logger.info(f"Attempting fallback CSV loading for {file_path}")
                try:
                    df = self._load_csv_fallback(file_path)
                    if not df.empty:
                        results['success'] = True
                        results['warnings'].append("Used fallback CSV loader")
                        df = self.standardize_column_names(df)
                        return df, results
                except Exception as fallback_e:
                    results['errors'].append(f"Fallback loading failed: {str(fallback_e)}")
            
            return pd.DataFrame(), results
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file with comprehensive error handling.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame containing the CSV data
        """
        # Set default parameters that can be overridden
        params = {
            'skipinitialspace': True,
            'on_bad_lines': 'warn',
            'low_memory': False
        }
        params.update(kwargs)
        
        try:
            df = pd.read_csv(file_path, **params)
            self.logger.info(f"Successfully loaded CSV from {file_path} with shape {df.shape}")
            return df
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error: {str(e)}")
            # Try with different parameters
            self.logger.info("Attempting to load with error_bad_lines=False")
            params['on_bad_lines'] = 'skip'
            return pd.read_csv(file_path, **params)
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error: {str(e)}")
            # Try with different encoding
            self.logger.info("Attempting to load with encoding='latin1'")
            params['encoding'] = 'latin1'
            return pd.read_csv(file_path, **params)
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def _load_csv_fallback(self, file_path: Path) -> pd.DataFrame:
        """
        Manual fallback loader for CSV files when pandas parser fails.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame constructed from the CSV data
        """
        self.logger.info(f"Using manual CSV parser for {file_path}")
        rows = []
        
        with open(file_path, 'r', errors='replace') as f:
            # Read header
            header_line = f.readline().strip()
            
            # Try different delimiters
            for delimiter in [',', ';', '\t', '|']:
                header = header_line.split(delimiter)
                if len(header) > 1:
                    break
            
            # Process data rows
            for i, line in enumerate(f):
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    values = line.split(delimiter)
                    
                    # Pad or trim values to match header length
                    if len(values) < len(header):
                        values.extend([None] * (len(header) - len(values)))
                    elif len(values) > len(header):
                        values = values[:len(header)]
                        
                    rows.append(dict(zip(header, values)))
                except Exception as e:
                    self.logger.warning(f"Error processing line {i+2}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Convert columns to appropriate types
        for col in df.columns:
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # Try to convert to datetime if 'date' in column name
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
        
        self.logger.info(f"Manual CSV parsing successful with shape {df.shape}")
        return df
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments to pass to pd.read_excel
            
        Returns:
            DataFrame containing the Excel data
        """
        try:
            df = pd.read_excel(file_path, **kwargs)
            self.logger.info(f"Successfully loaded Excel from {file_path} with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            raise
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            **kwargs: Additional arguments to pass to pd.read_json
            
        Returns:
            DataFrame containing the JSON data
        """
        try:
            df = pd.read_json(file_path, **kwargs)
            self.logger.info(f"Successfully loaded JSON from {file_path} with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {str(e)}")
            raise
    
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file from the data directory.
        
        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            DataFrame containing the CSV data
        """
        file_path = self.data_dir / filename
        df, _ = self.load_file(file_path, file_format='csv', **kwargs)
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for analysis with robust error handling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for preprocessing")
            return df
        
        self.logger.info("Preprocessing data...")
        processed_df = df.copy()
        
        try:
            # Standardize column names
            processed_df = self.standardize_column_names(processed_df)
            
            # Convert 'Date' column to datetime if it exists
            if 'Date' in processed_df.columns:
                processed_df['Date'] = pd.to_datetime(processed_df['Date'], errors='coerce')
                
            # Convert 'Number' column to numeric if it exists
            if 'Number' in processed_df.columns:
                processed_df['Number'] = pd.to_numeric(processed_df['Number'], errors='coerce')
                
            # Handle missing values
            # First try forward-fill then backward-fill
            processed_df = processed_df.ffill().bfill()
            
            # For any remaining NaNs in the Number column, replace with median
            if 'Number' in processed_df.columns and processed_df['Number'].isna().any():
                median_value = processed_df['Number'].median()
                processed_df['Number'].fillna(median_value, inplace=True)
            
            # Sort by date if available
            if 'Date' in processed_df.columns:
                processed_df = processed_df.sort_values('Date').reset_index(drop=True)
            
            # Validate the preprocessed data
            validation_results = self.validator.validate_dataframe(processed_df)
            
            if not validation_results["valid"]:
                for warning in validation_results["warnings"]:
                    self.logger.warning(f"Data validation warning: {warning}")
                for error in validation_results["errors"]:
                    self.logger.error(f"Data validation error: {error}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            # Return the original DataFrame if preprocessing fails
            return df
    
    def load_and_preprocess(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load and preprocess a file in one step.
        
        Args:
            filename: Name of the file in the data directory
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.load_csv(filename, **kwargs)
        return self.preprocess_data(df)
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive validation on a DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'dataframe': self.validator.validate_dataframe(df)
        }
        
        # Add date validation if applicable
        if 'Date' in df.columns:
            date_results = self.validator.validate_date_sequence(df['Date'])
            results['date_sequence'] = date_results
        
        # Add number validation if applicable
        if 'Number' in df.columns:
            number_results = self.validator.validate_number_sequence(
                df['Number'],
                expected_min=1,  # Assuming numbers are 1-10
                expected_max=10
            )
            results['number_sequence'] = number_results
            
        # Overall validity
        results['valid'] = all(r.get('valid', True) for r in results.values())
            
        return results
    
    def generate_synthetic_data(
        self, 
        num_rows: int = 100, 
        start_date: str = '2020-01-01',
        max_number: int = 10, 
        min_number: int = 1,
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data for testing.
        
        Args:
            num_rows: Number of rows to generate
            start_date: Starting date for the sequence
            max_number: Maximum random number
            min_number: Minimum random number
            filename: Optional filename to save the generated data
            
        Returns:
            DataFrame with synthetic data
        """
        self.logger.info(f"Generating synthetic data with {num_rows} rows")
        
        # Generate dates with a realistic pattern (not all consecutive days)
        base_dates = pd.date_range(start=start_date, periods=num_rows * 2, freq='D')
        dates = base_dates[np.sort(np.random.choice(len(base_dates), num_rows, replace=False))]
        dates = dates.sort_values()
        
        # Generate random numbers
        numbers = np.random.randint(min_number, max_number + 1, size=num_rows)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Number': numbers
        })
        
        # Save to file if filename provided
        if filename:
            file_path = self.data_dir / filename
            df.to_csv(file_path, index=False)
            self.logger.info(f"Saved synthetic data to {file_path}")
        
        return df
    
    def get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile for exploratory analysis.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Dictionary containing profile information
        """
        profile = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
                'missing_cells': df.isna().sum().sum(),
                'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'column_profiles': {}
        }
        
        # Profile each column
        for col in df.columns:
            col_data = df[col]
            col_profile = {
                'dtype': str(col_data.dtype),
                'missing_count': int(col_data.isna().sum()),
                'missing_percentage': float((col_data.isna().sum() / len(df)) * 100)
            }
            
            # Numeric column stats
            if pd.api.types.is_numeric_dtype(col_data):
                col_profile.update({
                    'min': float(col_data.min()) if not col_data.empty else None,
                    'max': float(col_data.max()) if not col_data.empty else None,
                    'mean': float(col_data.mean()) if not col_data.empty else None,
                    'median': float(col_data.median()) if not col_data.empty else None,
                    'std': float(col_data.std()) if not col_data.empty else None,
                    'unique_values': int(col_data.nunique())
                })
                
                # Distribution
                if col_data.nunique() < 50:  # Only for columns with fewer unique values
                    col_profile['value_counts'] = col_data.value_counts().to_dict()
            
            # Date column stats
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_profile.update({
                    'min': col_data.min() if not col_data.empty else None,
                    'max': col_data.max() if not col_data.empty else None,
                    'range_days': (col_data.max() - col_data.min()).days if not col_data.empty else None
                })
            
            # String/categorical column stats
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                col_profile.update({
                    'unique_values': int(col_data.nunique()),
                    'most_common': col_data.value_counts().head(5).to_dict() if not col_data.empty else None
                })
            
            profile['column_profiles'][col] = col_profile
        
        return profile


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create loader
    loader = EnhancedDataLoader("data")
    
    # Load and process data
    df = loader.load_csv("historical_random_numbers.csv")
    
    if not df.empty:
        # Preprocess
        processed_df = loader.preprocess_data(df)
        
        # Validate
        validation_results = loader.validate_data(processed_df)
        
        # Generate profile
        profile = loader.get_data_profile(processed_df)
        
        print(f"Loaded data with shape: {processed_df.shape}")
        print(f"Data is valid: {validation_results['valid']}")
    else:
        print("Failed to load data")