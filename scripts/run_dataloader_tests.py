#!/usr/bin/env python3
"""
Script to run enhanced dataloader tests.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the test module
from tests.utils.test_enhanced_data_loader import TestEnhancedDataLoader, TestDataSchemaValidator

def run_tests():
    """Run the enhanced dataloader tests."""
    print("Running EnhancedDataLoader tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestEnhancedDataLoader))
    suite.addTest(loader.loadTestsFromTestCase(TestDataSchemaValidator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)