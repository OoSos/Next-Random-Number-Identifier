import unittest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

# Import the enhanced data loader
from src.utils.enhanced_data_loader import EnhancedDataLoader, DataSchemaValidator

class TestEnhancedDataLoader(unittest.TestCase):
    """Test cases for the EnhancedDataLoader class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.temp_dir = TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Create a loader instance
        self.loader = EnhancedDataLoader(self.data_dir)
        
        # Create test files
        self.create_test_data()
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create test data files."""
        # Create a normal CSV file
        normal_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Number': range(1, 11)
        })
        normal_data.to_csv(self.data_dir / 'normal.csv', index=False)
        
        # Create a CSV with different column names
        alt_columns_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Super Ball': range(1, 11)
        })
        alt_columns_data.to_csv(self.data_dir / 'alt_columns.csv', index=False)
        
        # Create a CSV with missing values
        missing_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Number': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        })
        missing_data.to_csv(self.data_dir / 'missing_values.csv', index=False)
        
        # Create a malformed CSV file
        with open(self.data_dir / 'malformed.csv', 'w') as f:
            f.write("Date,Number\n")
            f.write("2020-01-01,1\n")
            f.write("malformed-date,not-a-number\n")
            f.write("2020-01-03,3\n")
    
    def test_load_csv_normal(self):
        """Test loading a normal CSV file."""
        df = self.loader.load_csv('normal.csv')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 10)
        self.assertIn('Date', df.columns)
        self.assertIn('Number', df.columns)
    
    def test_load_csv_alt_columns(self):
        """Test loading a CSV with alternative column names."""
        df = self.loader.load_csv('alt_columns.csv')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 10)
        self.assertIn('Date', df.columns)
        self.assertIn('Number', df.columns)  # Should be standardized
    
    def test_load_csv_missing_values(self):
        """Test loading a CSV with missing values."""
        df = self.loader.load_csv('missing_values.csv')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 10)
        # Should have NaN values
        self.assertTrue(df['Number'].isna().any())
    
    def test_load_csv_malformed(self):
        """Test loading a malformed CSV file."""
        df = self.loader.load_csv('malformed.csv')
        self.assertFalse(df.empty)  # Should still load
        self.assertEqual(len(df), 3)  # All three rows
    
    def test_load_nonexistent_csv(self):
        """Test loading a non-existent CSV file."""
        df = self.loader.load_csv('nonexistent.csv')
        self.assertTrue(df.empty)  # Should return empty DataFrame
    
    def test_preprocess_data_normal(self):
        """Test preprocessing normal data."""
        df = self.loader.load_csv('normal.csv')
        processed = self.loader.preprocess_data(df)
        self.assertEqual(len(processed), 10)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed['Number']))
    
    def test_preprocess_data_missing_values(self):
        """Test preprocessing data with missing values."""
        df = self.loader.load_csv('missing_values.csv')
        processed = self.loader.preprocess_data(df)
        self.assertEqual(len(processed), 10)
        # Missing values should be filled
        self.assertFalse(processed['Number'].isna().any())
    
    def test_preprocess_data_malformed(self):
        """Test preprocessing malformed data."""
        df = self.loader.load_csv('malformed.csv')
        processed = self.loader.preprocess_data(df)
        self.assertFalse(processed.empty)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed['Date']))
        # Non-numeric values should be converted to NaN and then filled
        self.assertTrue(pd.api.types.is_numeric_dtype(processed['Number']))
    
    def test_load_and_preprocess(self):
        """Test combined load and preprocess operation."""
        processed = self.loader.load_and_preprocess('normal.csv')
        self.assertEqual(len(processed), 10)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed['Number']))
    
    def test_validate_data(self):
        """Test data validation."""
        df = self.loader.load_and_preprocess('normal.csv')
        validation = self.loader.validate_data(df)
        self.assertTrue(validation['valid'])
        self.assertTrue(validation['dataframe']['valid'])
        self.assertTrue('date_sequence' in validation)
        self.assertTrue('number_sequence' in validation)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        synthetic = self.loader.generate_synthetic_data(num_rows=20)
        self.assertEqual(len(synthetic), 20)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(synthetic['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(synthetic['Number']))
        self.assertTrue(all(1 <= x <= 10 for x in synthetic['Number']))
    
    def test_get_data_profile(self):
        """Test data profiling."""
        df = self.loader.load_and_preprocess('normal.csv')
        profile = self.loader.get_data_profile(df)
        self.assertEqual(profile['basic_info']['rows'], 10)
        self.assertEqual(profile['basic_info']['columns'], 2)
        self.assertIn('Date', profile['column_profiles'])
        self.assertIn('Number', profile['column_profiles'])


class TestDataSchemaValidator(unittest.TestCase):
    """Test cases for the DataSchemaValidator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.validator = DataSchemaValidator()
        
        # Create test data
        self.normal_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Number': range(1, 11)
        })
        
        self.missing_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Number': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        })
        
        self.invalid_df = pd.DataFrame({
            'Date': ['2020-01-01', 'invalid', '2020-01-03'],
            'Number': [1, 'a', 3]
        })
        
        self.empty_df = pd.DataFrame()
    
    def test_validate_dataframe_normal(self):
        """Test validating a normal DataFrame."""
        result = self.validator.validate_dataframe(self.normal_df)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
        self.assertEqual(len(result['warnings']), 0)
    
    def test_validate_dataframe_missing(self):
        """Test validating a DataFrame with missing values."""
        result = self.validator.validate_dataframe(self.missing_df)
        self.assertTrue(result['valid'])  # Still valid
        self.assertGreater(len(result['warnings']), 0)  # Should have warnings
    
    def test_validate_dataframe_invalid(self):
        """Test validating an invalid DataFrame."""
        result = self.validator.validate_dataframe(self.invalid_df)
        self.assertTrue(result['valid'])  # Still valid with warnings
        self.assertGreater(len(result['warnings']), 0)
    
    def test_validate_dataframe_empty(self):
        """Test validating an empty DataFrame."""
        result = self.validator.validate_dataframe(self.empty_df)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_validate_date_sequence(self):
        """Test validating a date sequence."""
        dates = self.normal_df['Date']
        result = self.validator.validate_date_sequence(dates)
        self.assertTrue(result['valid'])
        self.assertTrue(result['stats']['is_chronological'])
        self.assertEqual(result['stats']['duplicate_dates'], 0)
    
    def test_validate_number_sequence(self):
        """Test validating a number sequence."""
        numbers = self.normal_df['Number']
        result = self.validator.validate_number_sequence(
            numbers, expected_min=1, expected_max=10
        )
        self.assertTrue(result['valid'])
        self.assertEqual(result['stats']['min'], 1.0)
        self.assertEqual(result['stats']['max'], 10.0)


if __name__ == '__main__':
    unittest.main()