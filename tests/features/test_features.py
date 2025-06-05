"""
Test script for feature engineering and selection components.
"""

import pandas as pd
import numpy as np
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
    try:
        features_df = feature_engineer.transform(df)
        print("Feature engineering successful!")
        print(f"Generated {len(features_df.columns)} features")
        print("\nSample features:")
        print(features_df.columns[:10].tolist())
    except Exception as e:
        print(f"Feature engineering failed: {str(e)}")
        
    # Test feature selection
    if 'features_df' in locals():
        selector = FeatureSelector(n_features=10, selection_method='combined')
        try:
            selector.fit(features_df.drop(['Date', 'Number'], axis=1).fillna(0), df['Number'])
            selected_features = selector.transform(features_df.drop(['Date', 'Number'], axis=1).fillna(0))
            print("\nFeature selection successful!")
            print("Selected features:")
            print(selected_features.columns.tolist())
        except Exception as e:
            print(f"Feature selection failed: {str(e)}")

if __name__ == "__main__":
    test_feature_engineering()