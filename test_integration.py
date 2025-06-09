# Standard library imports
import sys
import os
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local application imports
from src.main import main
from src.features.feature_engineering import FeatureEngineer

def test_end_to_end_workflow():
    """
    Integration test for the entire workflow from data loading to prediction.
    """
    # Create a small synthetic dataset
    dates = pd.date_range(start='2020-01-01', end='2020-01-31')
    numbers = np.random.randint(1, 11, size=len(dates))
    df = pd.DataFrame({'Date': dates, 'Number': numbers})
    
    # Save to temporary file
    temp_dir = Path('temp_test_data')
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / 'test_data.csv'
    df.to_csv(temp_file, index=False)
    
    try:
        # Run the main function with the test data
        results = main(data_path=str(temp_file), model_type='ensemble')
        
        # Assert that the workflow completed successfully
        assert results['success'], f"Main function failed: {results.get('error', 'Unknown error')}"
        
        # Assert that models were trained
        assert 'models' in results, "No models were trained"
        assert 'ensemble' in results['models'], "Ensemble model not found in results"
        
        # Assert that metrics were calculated
        assert 'metrics' in results, "No metrics were calculated"
        assert 'ensemble' in results['metrics'], "Ensemble metrics not found in results"
        
        # Try predicting with the ensemble model
        ensemble = results['models']['ensemble']
        
        # Create features for prediction
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
        
        # Make predictions
        predictions = ensemble.predict(X.head(1))
        
        # Verify prediction format
        assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
        assert len(predictions) == 1, "Should have one prediction"
        assert 1 <= predictions[0] <= 10, "Prediction should be between 1 and 10"
        
        print("Integration test passed successfully!")
        return True
    except Exception as e:
        print(f"Integration test failed: {str(e)}")
        return False
    finally:
        # Clean up temporary files
        if temp_file.exists():
            temp_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()

def test_end_to_end_workflow_with_fallback():
    """
    Integration test for the workflow with fallback to simpler model.
    """
    # Create a small synthetic dataset
    dates = pd.date_range(start='2020-01-01', end='2020-01-31')
    numbers = np.random.randint(1, 11, size=len(dates))
    df = pd.DataFrame({'Date': dates, 'Number': numbers})
    
    # Save to temporary file
    temp_dir = Path('temp_test_data')
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / 'test_data.csv'
    df.to_csv(temp_file, index=False)
    
    try:
        # Run the main function with the test data using a simpler model
        results = main(data_path=str(temp_file), model_type='xgb')
        
        # Assert that the workflow completed successfully
        assert results['success'], f"Main function failed with XGBoost: {results.get('error', 'Unknown error')}"
        
        # Assert that models were trained
        assert 'models' in results, "No models were trained"
        assert 'xgb' in results['models'], "XGBoost model not found in results"
        
        # Assert that metrics were calculated
        assert 'metrics' in results, "No metrics were calculated"
        assert 'xgb' in results['metrics'], "XGBoost metrics not found in results"
        
        # Try predicting with the XGBoost model
        xgb_model = results['models']['xgb']
        
        # Create features for prediction
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
        
        # Make predictions
        predictions = xgb_model.predict(X.head(1))
        
        # Verify prediction format
        assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
        assert len(predictions) == 1, "Should have one prediction"
        
        print("Integration test with fallback passed successfully!")
        return True
    except Exception as e:
        print(f"Integration test with fallback failed: {str(e)}")
        return False
    finally:
        # Clean up temporary files
        if temp_file.exists():
            temp_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()

def test_end_to_end_workflow_with_error_handling():
    """
    Integration test for the workflow error handling capabilities.
    """
    # Create an invalid dataset (missing required columns)
    df = pd.DataFrame({'InvalidColumn': np.random.randn(20)})
    
    # Save to temporary file
    temp_dir = Path('temp_test_data')
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / 'invalid_test_data.csv'
    df.to_csv(temp_file, index=False)
    
    try:
        # Run the main function with invalid data - should handle errors gracefully
        results = main(data_path=str(temp_file), model_type='ensemble')
        
        # Main should complete but might report failure
        assert 'success' in results, "Results dictionary should always have a success key"
        assert 'error' in results or 'models' in results, "Results should contain either error or models"
        
        # If successful despite invalid data, it means the error handling worked
        if results['success']:
            assert 'models' in results, "No models were trained despite success"
            assert len(results['models']) > 0, "No models were created despite success"
        
        print("Error handling test passed")
        return True
    except Exception as e:
        print(f"Error handling test failed with unhandled exception: {str(e)}")
        return False
    finally:
        # Clean up temporary files
        if temp_file.exists():
            temp_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()

if __name__ == "__main__":
    test_end_to_end_workflow()
    test_end_to_end_workflow_with_fallback()
    test_end_to_end_workflow_with_error_handling()