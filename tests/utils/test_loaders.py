# Standard library imports
import unittest
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from src.utils.simple_data_loader import SimpleDataLoader
from src.utils.enhanced_data_loader import EnhancedDataLoader
from src.utils.file_utils import debug_file_path

# Define project root for test data access
project_root = Path(__file__).parent.parent.parent

class TestBasicLoaderFunctionality(unittest.TestCase):
    """Test cases for basic data loading functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_dir = project_root / "data"
        self.simple_loader = SimpleDataLoader(str(self.data_dir))
        self.enhanced_loader = EnhancedDataLoader(str(self.data_dir))
        
    def test_simple_loader_basic(self):
        """Test basic functionality of SimpleDataLoader."""
        # Test CSV loading
        df = self.simple_loader.load_csv("historical_random_numbers.csv")
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn('Date', df.columns)
        self.assertIn('Number', df.columns)
        
        # Test preprocessing
        processed_df = self.simple_loader.preprocess_data(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['Number']))
    
    def test_file_paths(self):
        """Test file path handling."""
        path_info = debug_file_path()
        self.assertIsNotNone(path_info)
        self.assertIsInstance(path_info, dict)
    
    def test_enhanced_loader_basic(self):
        """Test basic functionality of EnhancedDataLoader."""
        df = self.enhanced_loader.load_csv("historical_random_numbers.csv")
        self.assertFalse(df.empty, "DataFrame should not be empty")
        
        # Test combined load and preprocess
        processed_df = self.enhanced_loader.load_and_preprocess("historical_random_numbers.csv")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['Number']))
    
    def test_error_handling(self):
        """Test error handling in both loaders."""
        # Test non-existent file
        df_simple = self.simple_loader.load_csv("nonexistent.csv")
        self.assertTrue(df_simple.empty)
        
        df_enhanced = self.enhanced_loader.load_csv("nonexistent.csv")
        self.assertTrue(df_enhanced.empty)

if __name__ == '__main__':
    unittest.main()