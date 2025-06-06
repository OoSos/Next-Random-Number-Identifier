# scripts/profile_feature_engineering.py
import cProfile
import pstats
import io
import os
import pandas as pd
import time
import sys

# Add src to Python path to allow direct imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer

# Define the path to the data file
DATA_FILE_PATH = "data/historical_random_numbers.csv"
# Define the output directory for profiling results
PROFILE_RESULTS_DIR = "benchmark_results"

# Ensure the output directory exists
os.makedirs(os.path.join(PROJECT_ROOT, PROFILE_RESULTS_DIR), exist_ok=True)

def profile_feature_engineering(num_rows_to_load, dataset_name):
    """
    Profiles the feature engineering process for a given number of rows.

    Args:
        num_rows_to_load (int or None): Number of rows to load from the dataset. None for all rows.
        dataset_name (str): A name for the dataset size (e.g., 'small', 'medium') for naming output files.
    """
    print(f"\n--- Profiling Feature Engineering for {dataset_name} dataset ({num_rows_to_load if num_rows_to_load else 'all'} rows) ---")

    # 1. Load data
    absolute_data_file_path = os.path.join(PROJECT_ROOT, DATA_FILE_PATH)
    print(f"Loading data: {absolute_data_file_path} (nrows={num_rows_to_load})")
    try:
        data_directory = os.path.dirname(absolute_data_file_path)
        data_filename = os.path.basename(absolute_data_file_path)
        
        data_loader = DataLoader(data_dir=data_directory)
        
        df = data_loader.load_csv(filename=data_filename, nrows=num_rows_to_load if num_rows_to_load is not None else None)

        if df.empty:
            print(f"Data loading (load_csv) returned an empty DataFrame for {dataset_name} dataset. Skipping profiling.")
            return None, None
        print(f"Loaded {len(df)} rows. Initial columns: {df.columns.tolist()}")

        # Verify that 'Date' and 'Number' columns are present after loading
        if 'Date' not in df.columns:
            print(f"Critical error: 'Date' column not found in loaded CSV. Columns: {df.columns.tolist()}")
            return None, None
        if 'Number' not in df.columns:
            print(f"Critical error: 'Number' column not found in loaded CSV. Columns: {df.columns.tolist()}")
            return None, None

        # Preprocess data (converts 'Date' to datetime, 'Number' to numeric, sorts by 'Date', handles NaNs)
        # preprocess_data expects 'Date' and 'Number'
        df = data_loader.preprocess_data(df.copy()) # Pass a copy to avoid SettingWithCopyWarning
        if df.empty:
            print(f"Data preprocessing returned an empty DataFrame for {dataset_name} dataset. Skipping profiling.")
            return None, None
        print(f"Columns after preprocess_data: {df.columns.tolist()}")

        # Rename columns to what FeatureEngineer expects ('date', 'number')
        rename_map_for_fe = {}
        if 'Date' in df.columns: # Should be present after preprocessing
            rename_map_for_fe['Date'] = 'date'
        else:
            print(f"Critical error: 'Date' column not found after preprocessing. FeatureEngineer might require it as 'date'. Columns: {df.columns.tolist()}")
            return None, None
        
        if 'Number' in df.columns: # Should be present after preprocessing
            rename_map_for_fe['Number'] = 'number'
        else:
            print(f"Critical error: 'Number' column not found after preprocessing. FeatureEngineer requires 'number'. Columns: {df.columns.tolist()}")
            return None, None 

        df.rename(columns=rename_map_for_fe, inplace=True)
        print(f"Columns after renaming for FeatureEngineer: {df.columns.tolist()}")
        
        # Final checks for columns required by FeatureEngineer
        if 'number' not in df.columns:
            print(f"Critical error: 'number' column is definitively missing before passing to FeatureEngineer. Columns: {df.columns.tolist()}")
            return None, None
        if 'date' not in df.columns:
             print(f"Critical error: 'date' column is missing before passing to FeatureEngineer. Columns: {df.columns.tolist()}")
             return None, None

        print(f"Successfully prepared {len(df)} rows for feature engineering.")

    except Exception as e:
        print(f"Error during data loading for {dataset_name} dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 2. Instantiate FeatureEngineer
    try:
        feature_engineer = FeatureEngineer()
    except Exception as e:
        print(f"Error during FeatureEngineer instantiation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 3. Profile the transform method
    profiler = cProfile.Profile()
    
    start_time = time.time()
    profiler.enable()
    processed_df = None # Initialize to ensure it's defined
    try:
        # Pass a copy to avoid unintended side effects if transform modifies df in-place
        processed_df = feature_engineer.transform(df.copy()) 
    except Exception as e:
        print(f"Error during feature_engineer.transform for {dataset_name} dataset: {e}")
        import traceback
        traceback.print_exc()
        # Fallthrough to disable profiler and return
    finally:
        profiler.disable()
    
    if processed_df is None: # Error occurred in transform
        return None, None

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Feature engineering for {dataset_name} dataset took: {execution_time:.4f} seconds.")

    # 4. Print and save profiling stats
    stats_file_path = os.path.join(PROJECT_ROOT, PROFILE_RESULTS_DIR, f"profile_feature_engineering_{dataset_name}.prof")
    profiler.dump_stats(stats_file_path)
    print(f"Profiling statistics saved to: {stats_file_path}")

    s = io.StringIO()
    # Sort by cumulative time, then total time
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative', 'tottime')
    ps.print_stats(30) # Print top 30 functions
    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30) # Print top 30 functions by total time
    print("\nTop 30 functions by total time (tottime):")
    print(s.getvalue())

    return execution_time, processed_df

if __name__ == "__main__":
    # Performance targets from NRNI_Copilot_Strategic_Guidance.md
    # FEATURE_ENGINEERING_TIME = {
    #     'small_dataset': 2.0,    # < 2 seconds for <1000 rows
    #     'medium_dataset': 10.0,  # < 10 seconds for <10000 rows
    # }
    # historical_random_numbers.csv has 1554 lines (1 header + 1553 data rows).

    results = {}

    # Profile for 'small' dataset (500 rows)
    # Target: < 2.0 seconds
    time_small, df_small = profile_feature_engineering(num_rows_to_load=500, dataset_name="small_500_rows")
    results["small_500_rows"] = {'time': time_small, 'target': 2.0}
    
    # Profile for 'medium' dataset (all 1553 rows from the file)
    # Target: < 10.0 seconds
    time_medium, df_medium = profile_feature_engineering(num_rows_to_load=1553, dataset_name="medium_1553_rows")
    results["medium_1553_rows"] = {'time': time_medium, 'target': 10.0}

    print("\n--- Profiling Summary ---")
    for name, data in results.items():
        if data['time'] is not None:
            print(f"Dataset: {name}")
            print(f"  Processing time: {data['time']:.4f}s (Target: < {data['target']:.1f}s)")
            if data['time'] < data['target']:
                print("  Performance target: MET")
            else:
                print("  Performance target: NOT MET")
        else:
            print(f"Dataset: {name} - Profiling did not complete successfully.")
            
    print("\n--- Profiling Complete ---")
    print("Next steps: Analyze the .prof files if necessary (e.g., using snakeviz or pstats_viewer).")
    print("If targets are not met, identify bottlenecks from the stats and discuss optimization strategies.")
    print("If data loading failed, verify CSV column names ('DrawDate', 'Number') and DataLoader behavior.")
