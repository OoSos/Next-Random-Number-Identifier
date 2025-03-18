from typing import Dict, Any, Optional, Union, List, Tuple, TypedDict, Literal, TypeVar, cast
from typing_extensions import NotRequired
import pandas as pd
import numpy as np
from pathlib import Path
import os
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype
)
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')
V = TypeVar('V')

def safe_dict_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Safely merge two dictionaries, handling None values."""
    result = base.copy()
    for k, v in update.items():
        if v is not None:
            result[k] = v
    return result

class ErrorList(List[str]):
    """Type-safe list for error messages."""
    pass

class WarningList(List[str]):
    """Type-safe list for warning messages."""
    pass

class FileInfo(TypedDict):
    path: str
    exists: bool
    size: int
    format: str
    row_count: int
    column_count: int

class FileLoadResult(TypedDict):
    success: bool
    errors: ErrorList
    warnings: WarningList
    file_info: FileInfo

class ColumnStats(TypedDict, total=False):
    value_counts: Dict[str, int]
    most_common: Dict[str, int]
    min: Optional[Union[float, pd.Timestamp]]
    max: Optional[Union[float, pd.Timestamp]]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    unique_values: int
    range_days: Optional[int]

class ColumnProfile(TypedDict):
    dtype: str
    missing_count: int
    missing_percentage: float
    stats: ColumnStats

class ValidationStats(TypedDict):
    """Statistics from validation operations."""
    min_date: Optional[pd.Timestamp]
    max_date: Optional[pd.Timestamp]
    is_chronological: bool
    duplicate_dates: int
    large_gaps: Optional[int]
    quality: Optional[Dict[str, Union[int, float]]]
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    outliers: Optional[int]
    value_distribution: Optional[Dict[str, Any]]
    chi2_uniformity: Optional[float]
    chi2_p_value: Optional[float]

class ValidationResult(TypedDict):
    """Result of a validation operation."""
    valid: bool
    errors: ErrorList
    warnings: WarningList
    stats: ValidationStats

def create_validation_result() -> ValidationResult:
    """Create a properly initialized validation result."""
    return {
        'valid': True,
        'errors': ErrorList(),
        'warnings': WarningList(),
        'stats': {
            'min_date': None,
            'max_date': None,
            'is_chronological': False,
            'duplicate_dates': 0,
            'large_gaps': None,
            'quality': None,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'outliers': None,
            'value_distribution': None,
            'chi2_uniformity': None,
            'chi2_p_value': None
        }
    }

class ValidationProfile(TypedDict):
    dataframe: ValidationResult
    date_sequence: NotRequired[ValidationResult]
    number_sequence: NotRequired[ValidationResult]
    valid: bool


class CSVReadParams(TypedDict, total=False):
    """Type hints for pandas read_csv parameters"""
    sep: Optional[str]
    delimiter: Optional[str]
    header: Union[int, List[int], Literal['infer'], None]
    names: Optional[List[str]]
    dtype: Optional[Dict[str, Any]]
    engine: Optional[Literal['c', 'python', 'pyarrow']]
    converters: Optional[Dict[str, Any]]
    true_values: Optional[List[str]]
    false_values: Optional[List[str]]
    skipinitialspace: bool
    skiprows: Optional[Union[int, List[int]]]
    skipfooter: int
    nrows: Optional[int]
    na_values: Optional[Union[List[str], Dict[str, List[str]]]]
    keep_default_na: bool
    na_filter: bool
    verbose: bool
    skip_blank_lines: bool
    parse_dates: Union[bool, List[int], List[str], Dict[str, List[str]]]
    thousands: Optional[str]
    decimal: str
    encoding: Optional[str]
    on_bad_lines: Literal['error', 'warn', 'skip']


class DataSchemaValidator:
    """Validates data schema, structure, and content integrity."""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List[str] = ["Date", "Number"],
        date_column: str = "Date",
        value_column: str = "Number"
    ) -> ValidationResult:
        """Validate DataFrame structure and content."""
        result = create_validation_result()

        if df.empty:
            result["valid"] = False
            result["errors"].append("DataFrame is empty")
            return result

        for col in required_columns:
            if col not in df.columns:
                result["errors"].append(f"Missing required column: {col}")

        if result["errors"]:
            result["valid"] = False
            return result

        if date_column in df.columns:
            if not is_datetime64_any_dtype(df[date_column]):
                result["warnings"].append(f"{date_column} column is not datetime type")

        if value_column in df.columns:
            if not is_numeric_dtype(df[value_column]):
                result["warnings"].append(f"{value_column} column is not numeric type")

        # Calculate data quality metrics
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100

        result["stats"]["quality"] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "total_cells": total_cells,
            "missing_cells": int(missing_cells),
            "missing_percentage": float(missing_percentage)
        }

        # Column-specific validation
        for col in df.columns:
            col_missing = df[col].isna().sum()
            col_missing_pct = (col_missing / len(df)) * 100

            if col_missing > 0:
                msg = f"Column {col} has {col_missing_pct:.1f}% missing"
                if col_missing_pct > 50:
                    result["errors"].append(msg)
                else:
                    result["warnings"].append(msg)

        return result

    @staticmethod
    def validate_date_sequence(
        dates: pd.Series,
        min_date: Optional[pd.Timestamp] = None,
        max_date: Optional[pd.Timestamp] = None
    ) -> ValidationResult:
        """Validate date sequence for chronological order and consistency."""
        results = create_validation_result()
        
        if dates.empty:
            results["valid"] = False
            results["errors"].append("Date series is empty")
            return results

        # Ensure dates are datetime type
        try:
            if not pd.api.types.is_datetime64_any_dtype(dates):
                dates = pd.to_datetime(dates, errors='coerce')
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Failed to convert dates to datetime: {str(e)}")
            return results
            
        # Store min and max dates
        results["stats"]["min_date"] = dates.min()
        results["stats"]["max_date"] = dates.max()
            
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
            results["warnings"].append(
                f"Found {duplicate_count} duplicate dates"
            )
            
        # Additional date analysis
        if len(dates.dropna()) > 1:
            date_diffs = dates.sort_values().diff().dropna()
            try:
                large_gaps = date_diffs[date_diffs.dt.days > 7]
                
                if not large_gaps.empty:
                    msg = f"Found {len(large_gaps)} gaps >7 days in sequence"
                    results["stats"]["large_gaps"] = len(large_gaps)
                    results["warnings"].append(msg)
            except Exception as e:
                results["warnings"].append(f"Could not analyze date gaps: {str(e)}")
        
        return results

    @staticmethod
    def validate_number_sequence(
        numbers: pd.Series,
        expected_min: Optional[int] = None,
        expected_max: Optional[int] = None
    ) -> ValidationResult:
        """Validate a sequence of numbers."""
        result = create_validation_result()

        if numbers.empty:
            result["valid"] = False
            result["errors"].append("Number series is empty")
            return result

        # Basic statistics
        observed_min = numbers.min()
        observed_max = numbers.max()
        result["stats"]["min"] = float(observed_min)
        result["stats"]["max"] = float(observed_max)
        result["stats"]["mean"] = float(numbers.mean())
        result["stats"]["median"] = float(numbers.median())
        result["stats"]["std"] = float(numbers.std())

        # Range validation
        if expected_min is not None and observed_min < expected_min:
            result["errors"].append(f"Values below minimum expected value ({expected_min})")
            result["valid"] = False

        if expected_max is not None and observed_max > expected_max:
            result["errors"].append(f"Values above maximum expected value ({expected_max})")
            result["valid"] = False

        # Outlier detection using IQR method
        q1, q3 = numbers.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((numbers < lower_bound) | (numbers > upper_bound)).sum()
        result["stats"]["outliers"] = int(outlier_count)

        if outlier_count > 0:
            result["warnings"].append(f"Found {outlier_count} outliers using IQR method")

        # Distribution analysis
        value_counts = numbers.value_counts()
        result["stats"]["value_distribution"] = value_counts.to_dict()

        # Chi-square test for uniformity
        if expected_min is not None and expected_max is not None:
            k = expected_max - expected_min + 1
            expected = [len(numbers)/k] * k
            observed = [value_counts.get(i, 0) for i in range(expected_min, expected_max + 1)]
            chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
            result["stats"]["chi2_uniformity"] = float(chi2)
            result["stats"]["chi2_p_value"] = None

        return result


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
            'errors': ErrorList(),
            'warnings': WarningList(),
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
        """Safe CSV loader with robust error handling."""
        typed_params: Dict[str, Any] = {
            'skipinitialspace': True,
            'on_bad_lines': 'warn',
            'header': 'infer',
            'encoding': None,
            'engine': 'python',  # Use python engine for better error handling
            'dtype': None,
            'parse_dates': False,
            'keep_default_na': True,
            'na_filter': True,
            'skip_blank_lines': True
        }
        
        # Override defaults with provided kwargs, maintaining type safety
        safe_params = {
            k: v for k, v in {**typed_params, **kwargs}.items() 
            if v is not None
        }
        
        try:
            df = pd.read_csv(str(file_path), **safe_params)
            self.logger.info(f"Successfully loaded CSV from {file_path} with shape {df.shape}")
            return df
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error: {str(e)}")
            # Try with different parameters
            safe_params['on_bad_lines'] = 'skip'
            return pd.read_csv(str(file_path), **safe_params)
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error: {str(e)}")
            # Try with different encoding
            safe_params['encoding'] = 'latin1'
            return pd.read_csv(str(file_path), **safe_params)
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def _load_csv_fallback(self, file_path: Path) -> pd.DataFrame:
        """
        Manual fallback loader for CSV files when pandas parser fails.
        """
        self.logger.info(f"Using manual CSV parser for {file_path}")
        rows: List[Dict[str, Any]] = []
        
        with open(file_path, 'r', errors='replace') as f:
            header_line = f.readline().strip()
            delimiter = ','  # default delimiter
            
            # Try different delimiters
            for test_delimiter in [',', ';', '\t', '|']:
                header = header_line.split(test_delimiter)
                if len(header) > 1:
                    delimiter = test_delimiter
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
                        values.extend([''] * (len(header) - len(values)))
                    elif len(values) > len(header):
                        values = values[:len(header)]
                        
                    rows.append(dict(zip(header, values)))
                except Exception as e:
                    self.logger.warning(f"Error processing line {i + 2}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Convert columns to appropriate types
        for col in df.columns:
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Try to convert to datetime if 'date' in column name
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
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
    
    def validate_data(self, df: pd.DataFrame) -> ValidationProfile:
        """
        Run comprehensive validation on a DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        base_validation: ValidationStats = {
            'min_date': None,
            'max_date': None,
            'is_chronological': False,
            'duplicate_dates': 0,
            'large_gaps': None,
            'quality': None,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'outliers': None,
            'value_distribution': None,
            'chi2_uniformity': None,
            'chi2_p_value': None
        }
        
        results: ValidationProfile = {
            'dataframe': {
                'valid': True,
                'errors': ErrorList(),
                'warnings': WarningList(),
                'stats': base_validation.copy()
            },
            'valid': True
        }
        
        # Add date validation if applicable
        if 'Date' in df.columns:
            date_results = self.validator.validate_date_sequence(df['Date'])
            results['date_sequence'] = date_results
        
        # Add number validation if applicable
        if 'Number' in df.columns:
            number_results = self.validator.validate_number_sequence(
                df['Number'],
                expected_min=1,
                expected_max=10
            )
            # Safe merging with type checking
            number_stats = safe_dict_merge(base_validation.copy(), number_results['stats'])
            results['number_sequence'] = cast(ValidationResult, {
                'valid': number_results['valid'],
                'errors': number_results['errors'].copy(),
                'warnings': number_results['warnings'].copy(),
                'stats': cast(ValidationStats, number_stats)
            })
            
        # Overall validity - safely handle optional fields
        all_results = [results['dataframe']]
        if 'date_sequence' in results:
            all_results.append(results['date_sequence'])
        if 'number_sequence' in results:
            all_results.append(results['number_sequence'])
            
        has_errors = any(
            bool(r.get('errors', [])) for r in all_results if r is not None
        )
        results['valid'] = not has_errors
            
        return results

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
            elif pd.api.types.is_string_dtype(col_data):
                col_profile.update({
                    'unique_values': int(col_data.nunique()),
                    'most_common': col_data.value_counts().head(5).to_dict() if not col_data.empty else None
                })
            
            profile['column_profiles'][col] = col_profile
        
        return profile

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