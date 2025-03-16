import unittest
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from unittest.mock import patch
import pytest

from src.models.markov_chain import MarkovChain


class TestMarkovChain(unittest.TestCase):
    """Test cases for MarkovChain class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample data for testing
        self.sample_data = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 4])
        self.X = pd.DataFrame({'Number': self.sample_data})
        self.y = self.sample_data
        
        # Create a default model instance
        self.model = MarkovChain(order=2)
        
    def test_initialization(self):
        """Test that the model initializes with correct default parameters"""
        model = MarkovChain()
        self.assertEqual(model.name, "MarkovChain")
        self.assertEqual(model.order, 2)
        self.assertEqual(model.smoothing, 0.1)
        self.assertIsNotNone(model.transition_matrix)
        self.assertIsNone(model.unique_numbers)
        
    def test_custom_initialization(self):
        """Test that the model initializes with custom parameters"""
        custom_model = MarkovChain(name="CustomMarkov", order=3, smoothing=0.2)
        self.assertEqual(custom_model.name, "CustomMarkov")
        self.assertEqual(custom_model.order, 3)
        self.assertEqual(custom_model.smoothing, 0.2)
        
    def test_create_sequence(self):
        """Test the sequence creation helper method"""
        model = MarkovChain(order=2)
        sequences = model._create_sequence(self.y)
        
        # For order=2, we should have len(data)-order pairs
        self.assertEqual(len(sequences), len(self.y) - model.order)
        
        # Check the first sequence pair
        self.assertEqual(sequences[0], ((1, 2), 3))
        
    def test_fit(self):
        """Test the fit method"""
        # Fit the model with smoothing disabled for testing
        result = self.model.fit(self.X, self.y, disable_smoothing=True)
        
        # Check that fit returns self
        self.assertIs(result, self.model)
        
        # Check that unique numbers are stored
        self.assertIsNotNone(self.model.unique_numbers)
        self.assertEqual(set(self.model.unique_numbers), set([1, 2, 3, 4]))
        
        # Check that transition matrix is populated
        self.assertGreater(len(self.model.transition_matrix), 0)
        
        # Check a specific transition probability
        # After seeing (1,2), the next number is always 3 in our sample
        self.assertAlmostEqual(self.model.transition_matrix[(1, 2)][3], 1.0, places=2)
        
    def test_predict(self):
        """Test the predict method"""
        # Fit the model first
        self.model.fit(self.X, self.y)
        
        # Test prediction
        test_df = pd.DataFrame({'Number': [3, 1]})  # Last two numbers to form state (3,1)
        predictions = self.model.predict(test_df)
        
        # Check prediction shape
        self.assertEqual(len(predictions), len(test_df))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_predict_without_fit(self):
        """Test prediction throws error when model isn't fitted"""
        # Create a new model without fitting
        unfitted_model = MarkovChain()
        
        # Test that predicting with unfitted model raises an error
        with self.assertRaises(ValueError):
            test_df = pd.DataFrame({'Number': [3, 1]})
            unfitted_model.predict(test_df)
            
    def test_predict_proba(self):
        """Test the probability prediction method"""
        # Fit the model
        self.model.fit(self.X, self.y)
        
        # Test probability prediction
        test_df = pd.DataFrame({'Number': [3, 1]})
        probas = self.model.predict_proba(test_df)
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(sum(probas), 1.0)
        self.assertEqual(len(probas), len(set(self.y)))  # One probability for each unique number
        
    def test_predict_proba_without_fit(self):
        """Test probability prediction throws error when model isn't fitted"""
        # Create a new model without fitting
        unfitted_model = MarkovChain()
        
        # Test that predicting probabilities with unfitted model raises an error
        with self.assertRaises(ValueError):
            test_df = pd.DataFrame({'Number': [3, 1]})
            unfitted_model.predict_proba(test_df)
            
    def test_get_transition_matrix(self):
        """Test getting the transition matrix"""
        # Fit the model
        self.model.fit(self.X, self.y)
        
        # Get transition matrix
        matrix = self.model.get_transition_matrix()
        
        # Check that it's a DataFrame with correct structure
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(set(matrix.columns), set(self.model.unique_numbers))
        
    def test_get_transition_matrix_without_fit(self):
        """Test getting transition matrix throws error when model isn't fitted"""
        # Create a new model without fitting
        unfitted_model = MarkovChain()
        
        # Test that getting transition matrix with unfitted model raises an error
        with self.assertRaises(ValueError):
            unfitted_model.get_transition_matrix()
            
    def test_most_probable_transitions(self):
        """Test getting most probable transitions"""
        # Fit the model
        self.model.fit(self.X, self.y)
        
        # Get most probable transitions
        transitions = self.model.get_most_probable_transitions(top_n=3)
        
        # Check structure
        self.assertIsInstance(transitions, list)
        self.assertLessEqual(len(transitions), 3)  # Could be fewer than top_n if there aren't enough transitions
        
        # Check content of first transition
        self.assertIsInstance(transitions[0], tuple)
        self.assertEqual(len(transitions[0]), 3)  # (state, next_state, probability)
        
    def test_feature_importance(self):
        """Test the feature importance functionality"""
        # Fit the model
        self.model.fit(self.X, self.y)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Check structure and values
        self.assertIsInstance(importance, dict)
        self.assertGreaterEqual(len(importance), 0)  # Could be empty if all transitions have equal probability
        
        if importance:
            # Check that values are normalized (max should be 1.0)
            self.assertAlmostEqual(max(importance.values()), 1.0)
        
    def test_feature_importance_without_fit(self):
        """Test feature importance throws error when model isn't fitted"""
        # Create a new model without fitting
        unfitted_model = MarkovChain()
        
        # Reset the transition matrix to simulate unfitted state
        unfitted_model.transition_matrix = {}
        
        # Test that getting feature importance with unfitted model raises an error
        with self.assertRaises(ValueError):
            unfitted_model.get_feature_importance()
            
    def test_runs_test(self):
        """Test the runs test for randomness"""
        result = self.model.runs_test(self.y)
        self.assertIn('z', result)
        self.assertIn('p_value', result)
        
    def test_serial_test(self):
        """Test the serial test for randomness"""
        result = self.model.serial_test(self.y)
        self.assertIn('z', result)
        self.assertIn('p_value', result)
        
    def test_visualize_transition_matrix(self):
        """Test the visualization of the transition matrix"""
        self.model.fit(self.X, self.y)
        with patch('matplotlib.pyplot.show') as mock_show:
            self.model.visualize_transition_matrix()
            mock_show.assert_called_once()
        
    def test_generate_report(self):
        """Test the generation of the model report"""
        self.model.fit(self.X, self.y)
        report = self.model.generate_report()
        self.assertIsInstance(report, str)
        self.assertIn('Model:', report)
        self.assertIn('Order:', report)
        self.assertIn('Smoothing:', report)
        self.assertIn('Total Transitions:', report)
        self.assertIn('Unique Numbers:', report)
        self.assertIn('Performance Metrics:', report)
        self.assertIn('Feature Importance:', report)


if __name__ == '__main__':
    unittest.main()