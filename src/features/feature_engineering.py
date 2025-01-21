from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced Feature Engineering class for random number prediction.
    Implements sklearn's TransformerMixin for pipeline compatibility.
    """
    def __init__(
        self,
        windows: List[int] = [5, 10, 20],
        lags: List[int] = [1, 2, 3, 5, 7, 14],
        enable_seasonal: bool = True,
        enable_cyclical: bool = True
    ):
        """
        Initialize the FeatureEngineer with configurable parameters.
        
        Args:
            windows: List of window sizes for rolling statistics
            lags: List of lag periods for creating lag features
            enable_seasonal: Enable seasonal feature creation
            enable_cyclical: Enable cyclical feature creation
        """
        self.windows = windows
        self.lags = lags
        self.enable_seasonal = enable_seasonal
        self.enable_cyclical = enable_cyclical
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not all(isinstance(w, int) and w > 0 for w in self.windows):
            raise ValueError("All window sizes must be positive integers")
        if not all(isinstance(l, int) and l > 0 for l in self.lags):
            raise ValueError("All lag periods must be positive integers")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method for sklearn compatibility."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by creating all features.
        
        Args:
            df: Input DataFrame with 'Date' and target number columns
            
        Returns:
            DataFrame with all engineered features
        """
        # Create a copy to avoid modifying original data
        result = df.copy()
        
        # Create features in sequence
        result = self._create_time_features(result)
        result = self._create_rolling_features(result)
        result = self._create_lag_features(result)
        result = self._create_statistical_features(result)
        
        return result

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive time-based features.
        """
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic time components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        
        if self.enable_cyclical:
            # Cyclical encoding for periodic features
            df = self._add_cyclical_features(df, 'Month', 12)
            df = self._add_cyclical_features(df, 'DayOfWeek', 7)
            df = self._add_cyclical_features(df, 'DayOfYear', 365)
        
        if self.enable_seasonal:
            # Seasonal indicators
            df['Season'] = df['Month'].map(self._get_season)
            
            # Holiday features (can be expanded based on specific calendar)
            df['IsHoliday'] = self._is_holiday(df['Date'])
            
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced rolling statistics features.
        """
        target_col = 'Number'  # Assuming 'Number' is the target column
        
        for window in self.windows:
            # Basic statistics
            prefix = f'Rolling_{window}'
            df[f'{prefix}_Mean'] = df[target_col].rolling(window=window).mean()
            df[f'{prefix}_Std'] = df[target_col].rolling(window=window).std()
            df[f'{prefix}_Min'] = df[target_col].rolling(window=window).min()
            df[f'{prefix}_Max'] = df[target_col].rolling(window=window).max()
            
            # Advanced statistics
            df[f'{prefix}_Skew'] = df[target_col].rolling(window=window).skew()
            df[f'{prefix}_Kurt'] = df[target_col].rolling(window=window).kurt()
            
            # Quantile features
            for q in [0.25, 0.75]:
                df[f'{prefix}_Q{int(q*100)}'] = df[target_col].rolling(window=window).quantile(q)
            
            # Exponential moving averages
            df[f'{prefix}_EMA'] = df[target_col].ewm(span=window).mean()
            
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive lag-based features.
        """
        target_col = 'Number'
        
        for lag in self.lags:
            # Basic lags
            df[f'Lag_{lag}'] = df[target_col].shift(lag)
            
            # Lag differences
            if lag > 1:
                df[f'Lag_Diff_{lag}'] = df[target_col].diff(lag)
                
            # Lag ratios (with error handling for zeros)
            lagged_values = df[target_col].shift(lag)
            df[f'Lag_Ratio_{lag}'] = np.where(
                lagged_values != 0,
                df[target_col] / lagged_values,
                1  # Default value for division by zero
            )
            
            # Moving averages of lags
            if lag > 2:
                df[f'Lag_MA_{lag}'] = df[f'Lag_{lag}'].rolling(window=min(lag, 3)).mean()
        
        return df

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced statistical features.
        """
        target_col = 'Number'
        
        # Frequency-based features
        value_counts = df[target_col].value_counts()
        df['Frequency'] = df[target_col].map(value_counts)
        df['FrequencyNorm'] = df['Frequency'] / len(df)
        
        # Deviation features
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        df['ZScore'] = (df[target_col] - mean_val) / std_val
        
        # Pattern detection
        df['IsOutlier'] = abs(df['ZScore']) > 2
        df['IsRepeated'] = df[target_col] == df[target_col].shift(1)
        
        return df

    @staticmethod
    def _add_cyclical_features(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
        """
        Create sine and cosine features for cyclical data.
        """
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
        return df

    @staticmethod
    def _get_season(month: int) -> str:
        """Map month to season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    @staticmethod
    def _is_holiday(dates: pd.Series) -> pd.Series:
        """
        Identify holidays (can be expanded based on specific calendar).
        Currently implements a basic holiday check.
        """
        return dates.dt.month.isin([12, 1]) & dates.dt.day.isin([25, 1])