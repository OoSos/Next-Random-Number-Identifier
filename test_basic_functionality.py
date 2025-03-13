"""
Basic test script to verify core functionality
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import with error handling
try:
    # First try to import our simplified data loader
    from src.utils.simple_data_loader import SimpleDataLoader
    from src.utils.file_utils import debug_file_path
    
    print("SimpleDataLoader imported successfully")
    
    # Test data loading
    print("\nTesting data loading...")
    data_dir = project_root / "data"
    loader = SimpleDataLoader(str(data_dir))
    
    # Debug file paths
    print("\nDebugging file paths...")
    path_info = debug_file_path()
    print(f"Path info: {path_info}")
    
    # Try loading the CSV using the guaranteed load_csv method
    try:
        print("\nLoading CSV data...")
        df = loader.load_csv("historical_random_numbers.csv")
        print(f"Successfully loaded CSV with shape: {df.shape}")
        
        # Preprocess the data
        print("\nPreprocessing data...")
        processed_df = loader.preprocess_data(df)
        print("Data after preprocessing:")
        print(processed_df.head())
        
    except Exception as e:
        print(f"Error loading or processing CSV: {str(e)}")
    
except ImportError as e:
    print(f"Import error: {str(e)}")
    print("Please ensure the project structure is correct")
except Exception as e:
    print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    print("Running basic functionality test...")