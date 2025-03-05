# test_feature_engineer.py
from src.features.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range(start='2020-01-01', end='2020-01-31')
numbers = np.random.randint(1, 11, size=len(dates))
df = pd.DataFrame({'Date': dates, 'Number': numbers})

# Test default initialization
fe = FeatureEngineer()
print("Successfully initialized FeatureEngineer with default parameters")

# Test with custom parameters
fe_custom = FeatureEngineer(
    windows=[3, 7],
    lags=[1, 3, 5],
    create_time_features=True,
    enable_cyclical=False,
    create_statistical_features=False
)
print("Successfully initialized with custom parameters")

# Test transformation
features = fe.transform(df)
print(f"Generated {len(features.columns)} features")
print("\nSample feature columns:")
print(list(features.columns)[:5])

print("\nImplementation successful!")