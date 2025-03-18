import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Adjust path to import from project
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced_data_loader import EnhancedDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.random_forest import RandomForestModel
from src.models.ensemble import EnhancedEnsemble
from src.models.markov_chain import MarkovChain


class TestEnhancedDataLoaderIntegration(unittest.TestCase):
    """
    Integration tests for EnhancedDataLoader with other system components.
    
    Tests the enhanced data loader's integration with:
    - Feature engineering pipeline
    - Model training workflow
    - Data validation system
    - Full end-to-end pipeline
    - Error handling scenarios
    
    Each test verifies a different aspect of system integration to ensure
    components work together correctly.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.data_dir = project_root / "data"
        self.test_file = "historical_random_numbers.csv"
        self.loader = EnhancedDataLoader(self.data_dir)
        
        # Create test data if it doesn't exist
        if not (self.data_dir / self.test_file).exists():
            self.create_test_data()

    def create_test_data(self):
        """Create sample test data for integration tests."""
        print(f"Creating sample test data in {self.data_dir}")
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
        # Create synthetic data
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        numbers = np.random.randint(1, 11, size=len(dates))
        df = pd.DataFrame({'Date': dates, 'Number': numbers})
        
        # Save test data
        df.to_csv(self.data_dir / self.test_file, index=False)

    def test_loader_with_feature_engineering(self):
        """
        Test that EnhancedDataLoader works with feature engineering.
        
        Verifies:
        - Data loading and preprocessing work with feature engineering
        - Features are correctly generated from processed data
        - Column types and values are preserved
        """
        # Load and preprocess data
        df = self.loader.load_csv(self.test_file)
        self.assertFalse(df.empty, "DataFrame should not be empty after loading")
        
        # Process data
        processed_df = self.loader.preprocess_data(df)
        self.assertIn('Date', processed_df.columns, "'Date' column should be present")
        self.assertIn('Number', processed_df.columns, "'Number' column should be present")
        
        # Verify data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['Date']), 
                      "'Date' column should be datetime type")
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['Number']), 
                      "'Number' column should be numeric type")
        
        # Test integration with feature engineering
        try:
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.transform(processed_df)
            
            # Verify feature engineering produced additional columns
            self.assertGreater(len(df_features.columns), 2, 
                             "Feature engineering should create additional columns")
            
            print(f"Feature engineering created columns: {df_features.columns.tolist()}")
        except Exception as e:
            self.fail(f"Feature engineering integration failed: {str(e)}")

    def test_loader_with_model_training(self):
        """
        Test that data from EnhancedDataLoader can be used for model training.
        
        Verifies:
        - Data can be used for model training
        - Models can consume processed features
        - Predictions can be generated
        """
        # Load and prepare the data
        df = self.loader.load_and_preprocess(self.test_file)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        
        # Create features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        
        # Prepare data for models
        feature_cols = [col for col in df_features.columns if col not in ['Date', 'Number']]
        X = df_features[feature_cols].fillna(0)
        y = df_features['Number']
        
        # Test with RandomForest model
        try:
            rf_model = RandomForestModel(n_estimators=10)  # Small for testing
            rf_model.fit(X, y)
            predictions = rf_model.predict(X.head(5))
            
            self.assertEqual(len(predictions), 5, "Model should make predictions")
            print(f"RandomForest model predictions: {predictions}")
        except Exception as e:
            self.fail(f"Model training integration failed: {str(e)}")

    def test_data_validation_integration(self):
        """
        Test that data validation works with the data pipeline.
        
        Verifies:
        - Data validation works in pipeline
        - Results contain expected validation info
        - Data profiling generates correct stats
        """
        # Load data
        df = self.loader.load_csv(self.test_file)
        
        # Run validation
        validation_results = self.loader.validate_data(df)
        
        # Check validation results
        self.assertIsNotNone(validation_results, "Validation should return results")
        self.assertIn('dataframe', validation_results, "Should have dataframe validation")
        self.assertIn('number_sequence', validation_results, "Should have number sequence validation")
        self.assertIn('date_sequence', validation_results, "Should have date sequence validation")
        
        print(f"Validation found data valid: {validation_results.get('valid', False)}")
        
        # Test with data profile
        profile = self.loader.get_data_profile(df)
        self.assertIn('basic_info', profile, "Profile should have basic info")
        self.assertIn('column_profiles', profile, "Profile should have column profiles")

    def test_full_pipeline_integration(self):
        """
        Test the enhanced data loader in the full model pipeline.
        
        Verifies:
        - Data loading and preprocessing work in full pipeline
        - Features are correctly generated and consumed by models
        - Models can be trained and make predictions
        """
        # Load and preprocess data
        df = self.loader.load_and_preprocess(self.test_file)
        
        # Create features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        
        # Prepare data
        feature_cols = [col for col in df_features.columns if col not in ['Date', 'Number']]
        X = df_features[feature_cols].fillna(0)
        y = df_features['Number']
        
        # Train random forest
        rf_model = RandomForestModel(n_estimators=10)
        rf_model.fit(X, y)
        
        # Train markov chain
        markov_model = MarkovChain(order=1)
        markov_model.fit(X, y)
        
        # Create and train ensemble
        try:
            ensemble = EnhancedEnsemble(models=[rf_model, markov_model])
            ensemble.fit(X, y)
            
            # Make predictions
            predictions = ensemble.predict(X.head(5))
            self.assertEqual(len(predictions), 5, "Ensemble should make predictions")
            
            print(f"Full pipeline test successful. Ensemble predictions: {predictions}")
        except Exception as e:
            self.fail(f"Full pipeline integration failed: {str(e)}")

    def test_error_handling_integration(self):
        """
        Test error handling in integration scenarios.
        
        Verifies:
        - Missing files are handled gracefully
        - Malformed data is preprocessed correctly
        - Data type conversion errors are managed
        """
        # Test with non-existent file
        try:
            df = self.loader.load_csv("nonexistent_file.csv")
            self.assertTrue(df.empty, "Should return empty DataFrame for nonexistent file")
            
            # Test preprocessing with manually created malformed data
            malformed_df = pd.DataFrame({
                'Date': ['2022-01-01', 'not-a-date', '2022-01-03'],
                'Number': ['10', 'abc', '5']
            })
            
            processed_df = self.loader.preprocess_data(malformed_df)
            self.assertFalse(processed_df.empty, "Should handle malformed data")
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['Date']), 
                          "Should convert dates despite errors")
            self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['Number']), 
                          "Should convert numbers despite errors")
            
            print("Error handling integration test passed")
        except Exception as e:
            self.fail(f"Error handling integration failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()