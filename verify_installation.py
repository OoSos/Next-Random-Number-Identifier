#!/usr/bin/env python3
"""
Environment verification script for NRNI project.
Tests the installation and verifies that optional dependencies work as documented.
"""

import sys
import os

def check_environment():
    """Check the environment and verify installation."""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test core functionality
    try:
        from src.models.random_forest import RandomForestModel
        from src.models.markov_chain import MarkovChain
        print("✅ Core models available")
    except ImportError as e:
        print(f"❌ Core models failed: {e}")
        return False
    
    # Test feature engineering
    try:
        from src.features.feature_engineering import FeatureEngineer
        print("✅ Feature engineering available")
    except ImportError as e:
        print(f"❌ Feature engineering failed: {e}")
        return False
    
    # Test XGBoost (optional)
    try:
        from src.models.xgboost_model import XGBoostModel
        from src.models.ensemble import EnhancedEnsemble
        print("✅ XGBoost features available")
    except ImportError:
        print("⚠️  XGBoost not available - install with: pip install -e .[xgboost]")
    
    # Test PyTorch (optional)
    try:
        import torch
        print("✅ PyTorch features available")
    except ImportError:
        print("⚠️  PyTorch not available - install with: pip install -e .[torch]")
    
    return True

def test_basic_functionality():
    """Test basic functionality to ensure everything works."""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        import pandas as pd
        import numpy as np
        from src.features.feature_engineering import FeatureEngineer
        
        # Create sample data
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Number': np.random.randint(1, 100, 10)
        })
        
        # Test feature engineering
        fe = FeatureEngineer()
        result = fe.transform(df)
        
        print(f"✅ Feature engineering test passed: {len(result.columns)} features created")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== NRNI Environment Verification ===")
    
    env_ok = check_environment()
    func_ok = test_basic_functionality()
    
    if env_ok and func_ok:
        print("\n🎉 Environment verification completed successfully!")
        print("Your NRNI installation is ready to use.")
    else:
        print("\n❌ Environment verification failed.")
        print("Please check the installation instructions in README.md")
        sys.exit(1)
