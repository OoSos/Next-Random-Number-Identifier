"""
Consolidated tests for EnhancedDataLoader.
Combines unit tests and integration tests for the data loader.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.enhanced_data_loader import EnhancedDataLoader, DataSchemaValidator
from src.features.feature_engineering import FeatureEngineer
from src.models.random_forest import RandomForestModel
from src.models.markov_chain import MarkovChain
from src.models.ensemble import EnhancedEnsemble

class TestEnhancedDataLoader(unittest.TestCase):
    """Combined test cases for EnhancedDataLoader."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.loader = EnhancedDataLoader(self.data_dir)
        self.create_test_data()
        
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
        
    def create_test_data(self):
        """Create test data files."""
        # Normal data
        normal_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Number': range(1, 11)
        })
        normal_data.to_csv(self.data_dir / 'normal.csv', index=False)
        
        # Data with missing values
        missing_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Number': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        })
        missing_data.to_csv(self.data_dir / 'missing.csv', index=False)
        
        # Malformed data
        with open(self.data_dir / 'malformed.csv', 'w') as f:
            f.write("Date,Number\n")
            f.write("2020-01-01,1\n")
            f.write("invalid-date,not-a-number\n")
            f.write("2020-01-03,3\n")
    
    # Unit Tests
    
    def test_load_csv(self):
        """Test CSV loading functionality."""
        df = self.loader.load_csv('normal.csv')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 10)
        self.assertIn('Date', df.columns)
        self.assertIn('Number', df.columns)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        df = self.loader.load_csv('normal.csv')
        processed = self.loader.preprocess_data(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed['Number']))
    
    def test_missing_values(self):
        """Test handling of missing values."""
        df = self.loader.load_csv('missing.csv')
        processed = self.loader.preprocess_data(df)
        self.assertFalse(processed['Number'].isna().any())
    
    def test_malformed_data(self):
        """Test handling of malformed data."""
        df = self.loader.load_csv('malformed.csv')
        processed = self.loader.preprocess_data(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed['Number']))
    
    # Integration Tests
    
    def test_feature_engineering_integration(self):
        """Test integration with feature engineering."""
        df = self.loader.load_and_preprocess('normal.csv')
        
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        
        self.assertGreater(len(df_features.columns), 2)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(df_features[col]) 
                          for col in df_features.columns if col != 'Date'))
    
    def test_model_training_integration(self):
        """Test integration with model training."""
        df = self.loader.load_and_preprocess('normal.csv')
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        
        feature_cols = [col for col in df_features.columns if col not in ['Date', 'Number']]
        X = df_features[feature_cols].fillna(0)
        y = df_features['Number']
        
        # Test with RandomForest
        rf_model = RandomForestModel(n_estimators=10)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X.head(5))
        self.assertEqual(len(rf_pred), 5)
        
        # Test with MarkovChain
        markov_model = MarkovChain(order=1)
        markov_model.fit(X, y)
        markov_pred = markov_model.predict(X.head(5))
        self.assertEqual(len(markov_pred), 5)
    
    def test_full_pipeline_integration(self):
        """Test the complete data pipeline."""
        # Load and process data
        df = self.loader.load_and_preprocess('normal.csv')
        
        # Create features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        
        # Prepare data
        feature_cols = [col for col in df_features.columns if col not in ['Date', 'Number']]
        X = df_features[feature_cols].fillna(0)
        y = df_features['Number']
        
        # Train models
        rf_model = RandomForestModel(n_estimators=10)
        rf_model.fit(X, y)
        
        markov_model = MarkovChain(order=1)
        markov_model.fit(X, y)
        
        # Create and test ensemble
        ensemble = EnhancedEnsemble(models=[rf_model, markov_model])
        ensemble.fit(X, y)
        predictions = ensemble.predict(X.head(5))
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(1 <= pred <= 10 for pred in predictions))

if __name__ == '__main__':
    unittest.main()