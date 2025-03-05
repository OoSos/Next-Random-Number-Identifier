"""
Next Random Number Identifier Package

This package provides tools and models for analyzing and predicting random number sequences
using various machine learning and statistical approaches.

Modules:
    - models: Contains implementations of different prediction models
    - features: Handles feature engineering and selection
    - utils: Provides utility functions and data handling
    - visualization: Contains visualization tools and plotting functions
"""

# Import main components for easier access
from .models import BaseModel, RandomForestModel
from utils import DataLoader
from features import FeatureEngineer
from visualization import plot_predictions, plot_feature_importance

# Define what should be available when someone imports your package
__all__ = [
    'BaseModel',
    'RandomForestModel',
    'DataLoader',
    'FeatureEngineer',
    'plot_predictions',
    'plot_feature_importance',
]

# Package metadata
__version__ = '2.0.0'
__author__ = 'AIQube Centaur Systems Team'