"""
Integration tests for the complete pipeline.
Consolidates all end-to-end and component integration tests.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from src.main import main
from src.utils.enhanced_data_loader import EnhancedDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.markov_chain import MarkovChain
from src.models.ensemble import EnhancedEnsemble

class TestPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.create_test_data()
        self.loader = EnhancedDataLoader(self.data_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
        
    def create_test_data(self):
        """Create test data for pipeline testing."""
        dates = pd.date_range(start='2020-01-01', end='2020-01-31')
        numbers = np.random.randint(1, 11, size=len(dates))
        df = pd.DataFrame({'Date': dates, 'Number': numbers})
        df.to_csv(self.data_dir / 'test_data.csv', index=False)

    def test_end_to_end_workflow(self):
        """Test the complete end-to-end workflow."""
        # Run main pipeline
        results = main(data_path=str(self.data_dir / 'test_data.csv'), 
                      model_type='ensemble')
        
        # Verify results
        self.assertTrue(results['success'])
        self.assertIn('models', results)
        self.assertIn('metrics', results)
        
        # Test prediction capability
        ensemble = results['models']['ensemble']
        feature_engineer = FeatureEngineer()
        
        # Create test input
        test_date = pd.DataFrame({'Date': [pd.Timestamp('2020-02-01')]})
        features = feature_engineer.transform(test_date)
        X = features.drop(['Date'], axis=1, errors='ignore').fillna(0)
        
        predictions = ensemble.predict(X)
        self.assertEqual(len(predictions), 1)
        self.assertTrue(1 <= predictions[0] <= 10)

    def test_data_pipeline_integration(self):
        """Test data loading and feature engineering pipeline."""
        # Load and process data
        df = self.loader.load_csv('test_data.csv')
        self.assertFalse(df.empty)
        
        processed_df = self.loader.preprocess_data(df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_df['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['Number']))
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(processed_df)
        self.assertGreater(len(df_features.columns), 2)
        
        # Feature selection
        feature_selector = FeatureSelector(n_features=5)
        X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
        y = df_features['Number']
        
        X_selected = feature_selector.fit_transform(X, y)
        self.assertEqual(X_selected.shape[1], 5)

    def test_model_pipeline_integration(self):
        """Test model training and prediction pipeline."""
        # Prepare data
        df = self.loader.load_and_preprocess('test_data.csv')
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.transform(df)
        
        X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
        y = df_features['Number']
        
        # Train individual models
        rf_model = RandomForestModel(n_estimators=10)
        xgb_model = XGBoostModel(n_estimators=10)
        markov_model = MarkovChain(order=1)
        
        rf_model.fit(X, y)
        xgb_model.fit(X, y)
        markov_model.fit(X, y)
        
        # Create and train ensemble
        ensemble = EnhancedEnsemble(models=[rf_model, xgb_model, markov_model])
        ensemble.fit(X, y)
        
        # Test predictions
        predictions = ensemble.predict(X.head(5))
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(1 <= pred <= 10 for pred in predictions))

    def test_error_handling_integration(self):
        """Test error handling in the pipeline."""
        # Test with malformed data
        malformed_df = pd.DataFrame({
            'Date': ['2020-01-01', 'invalid', '2020-01-03'],
            'Number': [1, 'bad', 3]
        })
        malformed_df.to_csv(self.data_dir / 'malformed.csv', index=False)
        
        # Test pipeline with malformed data
        results = main(data_path=str(self.data_dir / 'malformed.csv'),
                      model_type='rf')
        
        # Should complete with warnings but not fail
        self.assertIn('success', results)
        if results['success']:
            self.assertIn('models', results)
            self.assertIn('rf', results['models'])

if __name__ == '__main__':
    unittest.main()