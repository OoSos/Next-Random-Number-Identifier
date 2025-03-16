import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

logger.info(f"Project root: {project_root}")
logger.info(f"Python path: {sys.path}")

# Import the DataLoader
try:
    from src.utils.enhanced_data_loader import EnhancedDataLoader, DataValidator
    logger.info("Successfully imported DataLoader and DataValidator")
except ImportError as e:
    logger.error(f"Failed to import DataLoader: {e}")
    logger.info("Trying alternative import path...")
    
    try:
        # Try to import from system path
        sys.path.insert(0, str(project_root / "src"))
        from src.utils.enhanced_data_loader import EnhancedDataLoader, DataValidator
        logger.info("Successfully imported DataLoader from alternative path")
    except ImportError as e2:
        logger.error(f"Alternative import also failed: {e2}")
        sys.exit(1)


def setup_test_environment():
    """Set up test environment with synthetic data."""
    logger.info("Setting up test environment")
    
    # Create data directory if it doesn't exist
    data_dir = project_root / "data"
    if not data_dir.exists():
        logger.info(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data file
    test_path = data_dir / "test_data.csv"
    logger.info(f"Creating test data file: {test_path}")
    
    # Generate synthetic data
    dates = pd.date_range(start='2020-01-01', periods=50)
    numbers = np.random.randint(1, 11, size=50)
    test_df = pd.DataFrame({'Date': dates, 'Number': numbers})
    test_df.to_csv(test_path, index=False)
    logger.info(f"Created test data with {len(test_df)} rows")
    
    # Create file with different column names
    alt_columns_path = data_dir / "alt_columns_data.csv"
    alt_df = pd.DataFrame({'Date': dates, 'Super Ball': numbers})
    alt_df.to_csv(alt_columns_path, index=False)
    logger.info(f"Created test data with alternative column names")
    
    # Create file with missing values
    missing_values_path = data_dir / "missing_values_data.csv"
    missing_df = test_df.copy()
    missing_df.loc[10:15, 'Number'] = np.nan
    missing_df.to_csv(missing_values_path, index=False)
    logger.info(f"Created test data with missing values")
    
    # Create a malformed file
    malformed_path = data_dir / "malformed_data.csv"
    with open(malformed_path, 'w') as f:
        f.write("Date,Number\n")
        f.write("2020-01-01,1\n")
        f.write("malformed-date,not-a-number\n")
        f.write("2020-01-03,3\n")
    logger.info(f"Created malformed test data file")
    
    return {
        'data_dir': data_dir,
        'test_path': test_path,
        'alt_columns_path': alt_columns_path,
        'missing_values_path': missing_values_path,
        'malformed_path': malformed_path
    }


def test_loading_functionality(test_env):
    """Test all loading functionality."""
    logger.info("\n=== Testing basic loading functionality ===")
    
    data_dir = test_env['data_dir']
    loader = EnhancedDataLoader(str(data_dir))
    
    # Test loading normal file
    df = loader.load_csv("test_data.csv")
    if not df.empty:
        logger.info(f"Successfully loaded test data with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"First few rows:\n{df.head()}")
    else:
        logger.error("Failed to load test data")
        
    # Test loading file with alternative column names
    alt_df = loader.load_csv("alt_columns_data.csv")
    if 'Number' in alt_df.columns:
        logger.info("Successfully standardized column names")
        logger.info(f"Columns after standardization: {alt_df.columns.tolist()}")
    else:
        logger.error("Failed to standardize column names")
        
    # Test loading file with missing values
    missing_df = loader.load_csv("missing_values_data.csv")
    if missing_df['Number'].isna().any():
        logger.info(f"Successfully loaded file with missing values")
        logger.info(f"Missing value count: {missing_df['Number'].isna().sum()}")
    
    # Test loading malformed file
    malformed_df = loader.load_csv("malformed_data.csv")
    if not malformed_df.empty:
        logger.info(f"Successfully loaded malformed file with shape: {malformed_df.shape}")
        logger.info(f"Data types: {malformed_df.dtypes}")
    
    # Test loading non-existent file
    nonexistent_df = loader.load_csv("nonexistent_file.csv")
    if nonexistent_df.empty:
        logger.info("Correctly handled non-existent file")
    else:
        logger.error("Unexpectedly loaded non-existent file")
        
    return {
        'normal': df,
        'alt_columns': alt_df,
        'missing_values': missing_df,
        'malformed': malformed_df
    }


def test_preprocessing_functionality(test_env, dfs):
    """Test all preprocessing functionality."""
    logger.info("\n=== Testing preprocessing functionality ===")
    
    data_dir = test_env['data_dir']
    loader = EnhancedDataLoader(str(data_dir))
    
    # Test preprocessing normal data
    processed_df = loader.preprocess_data(dfs['normal'])
    logger.info(f"Preprocessed normal data with shape: {processed_df.shape}")
    
    # Check if Date is datetime
    is_datetime = pd.api.types.is_datetime64_any_dtype(processed_df['Date'])
    logger.info(f"Date column is datetime: {is_datetime}")
    
    # Check if Number is numeric
    is_numeric = pd.api.types.is_numeric_dtype(processed_df['Number'])
    logger.info(f"Number column is numeric: {is_numeric}")
    
    # Test preprocessing data with missing values
    processed_missing = loader.preprocess_data(dfs['missing_values'])
    missing_after = processed_missing['Number'].isna().sum()
    logger.info(f"Missing values after preprocessing: {missing_after}")
    
    # Test preprocessing malformed data
    processed_malformed = loader.preprocess_data(dfs['malformed'])
    logger.info(f"Processed malformed data with shape: {processed_malformed.shape}")
    logger.info(f"Data types after processing: {processed_malformed.dtypes}")
    
    # Test combined load and preprocess
    combined_df = loader.load_and_preprocess("test_data.csv")
    logger.info(f"Combined load and preprocess result shape: {combined_df.shape}")
    
    return {
        'processed_normal': processed_df,
        'processed_missing': processed_missing,
        'processed_malformed': processed_malformed,
        'combined': combined_df
    }


def test_data_validation(processed_dfs):
    """Test data validation functionality."""
    logger.info("\n=== Testing data validation functionality ===")
    
    validator = DataValidator()
    
    # Validate normal data structure
    normal_validation = validator.validate_csv_structure(processed_dfs['processed_normal'])
    logger.info(f"Normal data validation result: {normal_validation['valid']}")
    if normal_validation['warnings']:
        logger.info(f"Validation warnings: {normal_validation['warnings']}")
    
    # Validate malformed data structure
    malformed_validation = validator.validate_csv_structure(processed_dfs['processed_malformed'])
    logger.info(f"Malformed data validation result: {malformed_validation['valid']}")
    logger.info(f"Validation warnings: {malformed_validation['warnings']}")
    
    # Validate date sequence
    date_validation = validator.validate_date_sequence(processed_dfs['processed_normal']['Date'])
    logger.info(f"Date sequence validation result: {date_validation['valid']}")
    logger.info(f"Date statistics: {date_validation['stats']}")
    
    return {
        'normal_validation': normal_validation,
        'malformed_validation': malformed_validation,
        'date_validation': date_validation
    }


def test_synthetic_data_generation():
    """Test the synthetic data generation functionality."""
    logger.info("\n=== Testing synthetic data generation ===")
    
    loader = EnhancedDataLoader("data")
    
    # Generate synthetic data
    synthetic_df = loader.generate_synthetic_data(num_rows=100)
    logger.info(f"Generated synthetic data with shape: {synthetic_df.shape}")
    logger.info(f"Date range: {synthetic_df['Date'].min()} to {synthetic_df['Date'].max()}")
    logger.info(f"Number range: {synthetic_df['Number'].min()} to {synthetic_df['Number'].max()}")
    
    return synthetic_df


def print_test_summary(results):
    """Print a summary of all test results."""
    logger.info("\n=== Test Summary ===")
    
    # Check if required functionality works
    basic_loading = results['loading']['normal'] is not None and not results['loading']['normal'].empty
    preprocessing = results['preprocessing']['processed_normal'] is not None and not results['preprocessing']['processed_normal'].empty
    validation = results['validation']['normal_validation']['valid']
    
    logger.info(f"Basic loading functionality: {'✅' if basic_loading else '❌'}")
    logger.info(f"Preprocessing functionality: {'✅' if preprocessing else '❌'}")
    logger.info(f"Data validation functionality: {'✅' if validation else '❌'}")
    logger.info(f"Synthetic data generation: {'✅' if results['synthetic'] is not None else '❌'}")
    
    overall_success = all([basic_loading, preprocessing, validation, results['synthetic'] is not None])
    logger.info(f"\nOverall test result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
    
    return overall_success


def main():
    """Run all DataLoader tests."""
    logger.info("Starting DataLoader tests")
    
    # Setup test environment
    test_env = setup_test_environment()
    
    # Run tests
    loading_results = test_loading_functionality(test_env)
    preprocessing_results = test_preprocessing_functionality(test_env, loading_results)
    validation_results = test_data_validation(preprocessing_results)
    synthetic_df = test_synthetic_data_generation()
    
    # Compile results
    results = {
        'loading': loading_results,
        'preprocessing': preprocessing_results,
        'validation': validation_results,
        'synthetic': synthetic_df
    }
    
    # Print summary
    success = print_test_summary(results)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
