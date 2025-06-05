"""
Test script for feature engineering and selection components.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector

def test_feature_engineering():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    numbers = np.random.randint(1, 11, size=len(dates))
    df = pd.DataFrame({
        'Date': dates,
        'Number': numbers
    })
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(
        windows=[5, 10],
        lags=[1, 2, 3],
        enable_seasonal=True,
        enable_cyclical=True
    )
    # Transform data
    features_df = feature_engineer.transform(df)
    assert not features_df.empty, "Feature engineering should return non-empty DataFrame"
    assert len(features_df.columns) > len(df.columns), "Should generate additional features"
    assert len(features_df) == len(df), "Should maintain same number of rows"

def test_feature_selection():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    numbers = np.random.randint(1, 11, size=len(dates))
    df = pd.DataFrame({
        'Date': dates,
        'Number': numbers
    })
    feature_engineer = FeatureEngineer(
        windows=[5, 10],
        lags=[1, 2, 3],
        enable_seasonal=True,
        enable_cyclical=True
    )
    features_df = feature_engineer.transform(df)
    selector = FeatureSelector(n_features=10, selection_method='ensemble')
    selector.fit(features_df.drop(['Date', 'Number'], axis=1).fillna(0), df['Number'])
    selected_features = selector.transform(features_df.drop(['Date', 'Number'], axis=1).fillna(0))
    assert not selected_features.empty, "Feature selection should return non-empty DataFrame"
    assert selected_features.shape[1] == 10, "Should select 10 features"