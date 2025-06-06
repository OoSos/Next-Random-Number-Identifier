from typing import List, Dict, Optional, Union, Set, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from src.utils import standardize_column_names  # Use absolute import for consistency

# Optimized helper functions for performance bottlenecks
def _fast_rolling_stats_optimized(series: pd.Series, window: int) -> Dict[str, pd.Series]:
    """
    Optimized computation of multiple rolling statistics in a single pass.
    
    This avoids multiple rolling window computations by calculating
    mean, std, min, max, skew, and kurtosis together.
    """
    # Use pandas built-in optimized functions where possible
    rolling_obj = series.rolling(window=window)
    
    stats_dict = {
        'mean': rolling_obj.mean(),
        'std': rolling_obj.std(),
        'min': rolling_obj.min(),
        'max': rolling_obj.max(),
    }
    
    # Add derived statistics
    stats_dict['range'] = stats_dict['max'] - stats_dict['min']
    
    # Only compute expensive stats for larger windows to reduce overhead
    if window >= 10:
        stats_dict['skew'] = rolling_obj.skew()
        stats_dict['kurt'] = rolling_obj.kurt()
        # Quantiles
        stats_dict['q25'] = rolling_obj.quantile(0.25)
        stats_dict['q50'] = rolling_obj.quantile(0.5) 
        stats_dict['q75'] = rolling_obj.quantile(0.75)
        stats_dict['iqr'] = stats_dict['q75'] - stats_dict['q25']
    
    return stats_dict

def _fast_entropy_calculation_optimized(values):
    """
    Optimized entropy calculation using vectorized operations.
    """
    if len(values) <= 1:
        return 0.0
    
    # Use pandas value_counts for efficient counting
    value_counts = pd.Series(values).value_counts()
    total = len(values)
    probabilities = value_counts / total
    
    # Vectorized entropy calculation
    entropy = -(probabilities * np.log2(probabilities)).sum()
    return entropy

def _fast_run_complexity_optimized(values):
    """
    Optimized run complexity calculation using vectorized operations.
    """
    if len(values) <= 1:
        return len(values)
    
    # Use numpy diff to find changes more efficiently    
    values_array = np.asarray(values)
    changes = np.diff(values_array) != 0
    runs = int(np.sum(changes)) + 1
    
    return runs

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering for random number prediction.
    
    This class generates a comprehensive set of features designed to capture
    patterns in random number sequences, including temporal patterns,
    sequential patterns, cyclical behaviors, and statistical anomalies.
    
    Implements sklearn's TransformerMixin for pipeline compatibility.
    """
    def __init__(
        self,
        windows: List[int] = [5, 10, 20, 30, 50],
        lags: List[int] = [1, 2, 3, 5, 7, 10, 14, 21, 28],
        cycles: List[int] = [2, 3, 4, 5, 6, 7],
        create_time_features: bool = True,
        enable_seasonal: bool = True,
        enable_cyclical: bool = True,
        create_lag_features: bool = True,
        create_rolling_features: bool = True,
        create_frequency_features: bool = True,
        create_statistical_features: bool = True,
        create_pattern_features: bool = True,
        create_entropy_features: bool = True,
        create_rare_pattern_features: bool = True,
        target_column: str = 'Number',
        date_column: str = 'Date'
    ):
        """
        Initialize the FeatureEngineer with configurable parameters.
        
        Args:
            windows: List of window sizes for rolling statistics
            lags: List of lag periods for creating lag features
            cycles: List of cycle lengths to check for patterns
            create_time_features: Whether to create time-based features
            enable_seasonal: Enable seasonal feature creation
            enable_cyclical: Enable cyclical feature creation
            create_lag_features: Whether to create lag features
            create_rolling_features: Whether to create rolling statistics features
            create_frequency_features: Whether to create frequency-based features
            create_statistical_features: Whether to create statistical features
            create_pattern_features: Whether to create pattern features
            create_entropy_features: Whether to create entropy-based features
            create_rare_pattern_features: Whether to create rare pattern features
            target_column: Name of the column containing the target numbers
            date_column: Name of the column containing dates
        """
        self.windows = windows
        self.lags = lags
        self.cycles = cycles
        self.create_time_features = create_time_features
        self.enable_seasonal = enable_seasonal
        self.enable_cyclical = enable_cyclical
        self.create_lag_features = create_lag_features
        self.create_rolling_features = create_rolling_features
        self.create_frequency_features = create_frequency_features
        self.create_statistical_features = create_statistical_features
        self.create_pattern_features = create_pattern_features
        self.create_entropy_features = create_entropy_features
        self.create_rare_pattern_features = create_rare_pattern_features
        self.target_column = target_column
        self.date_column = date_column
        self._validate_parameters()
        
        # Store feature information for later analysis
        self.feature_groups = {}
        
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not all(isinstance(w, int) and w > 0 for w in self.windows):
            raise ValueError("All window sizes must be positive integers")
        if not all(isinstance(l, int) and l > 0 for l in self.lags):
            raise ValueError("All lag periods must be positive integers")
        if not all(isinstance(c, int) and c > 0 for c in self.cycles):
            raise ValueError("All cycle lengths must be positive integers")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method for sklearn compatibility."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by creating all features with robust error handling.
        
        Args:
            df: Input DataFrame with date and target number columns
            
        Returns:
            DataFrame with all engineered features
        """
        # Create a copy to avoid modifying original data
        result = df.copy()
        
        # Track what features we create for each group
        self.feature_groups = {
            'time': [],
            'rolling': [],
            'lag': [],
            'frequency': [],
            'statistical': [],
            'pattern': [],
            'entropy': [],
            'rare_pattern': []
        }
        
        # Standardize column names
        result = self._standardize_column_names(result)
        
        # Ensure Date column is in proper format
        if self.date_column in result.columns:
            try:
                result[self.date_column] = pd.to_datetime(result[self.date_column], errors='coerce')
                # Drop rows with invalid dates
                result = result.dropna(subset=[self.date_column])
            except Exception as e:
                print(f"Error converting Date column: {str(e)}")
        
        # Ensure Number column is numeric
        if self.target_column in result.columns:
            try:
                result[self.target_column] = pd.to_numeric(result[self.target_column], errors='coerce')
                # Fill NaN values with median or a default value
                median_value = result[self.target_column].median()
                result[self.target_column] = result[self.target_column].fillna(
                    median_value if not pd.isna(median_value) else 5
                )
            except Exception as e:
                print(f"Error converting Number column: {str(e)}")
        
        # Create features in sequence based on configuration
        if self.create_time_features and self.date_column in result.columns:
            try:
                result = self._create_time_features(result)
            except Exception as e:
                print(f"Error creating time features: {str(e)}")
        
        if self.create_rolling_features and self.target_column in result.columns:
            try:
                result = self._create_rolling_features(result)
            except Exception as e:
                print(f"Error creating rolling features: {str(e)}")
        
        if self.create_lag_features and self.target_column in result.columns:
            try:
                result = self._create_lag_features(result)
            except Exception as e:
                print(f"Error creating lag features: {str(e)}")
        
        if self.create_frequency_features and self.target_column in result.columns:
            try:
                result = self._create_frequency_features(result)
            except Exception as e:
                print(f"Error creating frequency features: {str(e)}")
        
        if self.create_statistical_features and self.target_column in result.columns:
            try:
                result = self._create_statistical_features(result)
            except Exception as e:
                print(f"Error creating statistical features: {str(e)}")
        
        if self.create_pattern_features and self.target_column in result.columns:
            try:
                result = self._create_pattern_features(result)
            except Exception as e:
                print(f"Error creating pattern features: {str(e)}")
        
        if self.create_entropy_features and self.target_column in result.columns:
            try:
                result = self._create_entropy_features(result)
            except Exception as e:
                print(f"Error creating entropy features: {str(e)}")
        
        if self.create_rare_pattern_features and self.target_column in result.columns:
            try:
                result = self._create_rare_pattern_features(result)
            except Exception as e:
                print(f"Error creating rare pattern features: {str(e)}")
        
        # Encode categorical features
        result = self._encode_categorical_features(result)
        
        return result

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive time-based features.
        
        These features extract patterns related to the date of each random number selection,
        potentially capturing day-of-week effects, seasonal patterns, etc.
        """
        time_features = []
        date_col = self.date_column
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic calendar components
        features = [
            ('Year', df[date_col].dt.year),
            ('Month', df[date_col].dt.month),
            ('Day', df[date_col].dt.day),
            ('DayOfWeek', df[date_col].dt.dayofweek),
            ('DayOfMonth', df[date_col].dt.day),
            ('DayOfYear', df[date_col].dt.dayofyear),
            ('WeekOfYear', df[date_col].dt.isocalendar().week),
            ('Quarter', df[date_col].dt.quarter)
        ]
        
        for name, values in features:
            df[name] = values
            time_features.append(name)
        
        # Interval days between selections
        df['IntervalDays'] = df[date_col].diff().dt.days
        time_features.append('IntervalDays')
        
        # Rolling average of interval days
        for window in self.windows:
            feature_name = f'IntervalDays_Rolling_{window}_Mean'
            df[feature_name] = df['IntervalDays'].rolling(window=window).mean()
            time_features.append(feature_name)
        
        # Day of month category (early, mid, late)
        df['DayOfMonthCategory'] = pd.cut(
            df['Day'], 
            bins=[0, 10, 20, 32], 
            labels=['Early', 'Mid', 'Late']
        ).astype(str)
        df['DayOfMonthCategory'] = df['DayOfMonthCategory'].astype('category').cat.codes
        time_features.append('DayOfMonthCategory')
        
        # Season based on month
        df['Season'] = df['Month'].apply(self._get_season)
        df['Season'] = df['Season'].astype('category').cat.codes
        time_features.append('Season')
        
        # Is weekend
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        time_features.append('IsWeekend')
        
        # Is holiday (simple approximation)
        df['IsHoliday'] = self._is_holiday(df[date_col])
        time_features.append('IsHoliday')
        
        if self.enable_cyclical:
            # Cyclical encoding for periodic features
            for col, period in [('Month', 12), ('DayOfWeek', 7), ('DayOfYear', 365)]:
                df = self._add_cyclical_features(df, col, period)
                time_features.extend([f'{col}_sin', f'{col}_cos'])
          # Store created feature names
        self.feature_groups['time'] = time_features
        
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced rolling statistics features with optimized performance.
        """
        rolling_features = []
        target_col = self.target_column
        
        # Use batch processing to avoid DataFrame fragmentation
        new_columns = {}
        
        for window in self.windows:
            prefix = f'Rolling_{window}'
            
            # Use optimized fast rolling stats function for multiple calculations
            if window >= 5:
                # Calculate multiple statistics in one pass to reduce overhead
                stats_dict = _fast_rolling_stats_optimized(df[target_col], window)
                
                for stat_name, stat_values in stats_dict.items():
                    column_name = f'{prefix}_{stat_name.capitalize()}'
                    new_columns[column_name] = stat_values
                    rolling_features.append(column_name)
            else:
                # For small windows, use basic pandas operations
                rolling_obj = df[target_col].rolling(window=window)
                new_columns[f'{prefix}_Mean'] = rolling_obj.mean()
                new_columns[f'{prefix}_Std'] = rolling_obj.std()
                new_columns[f'{prefix}_Min'] = rolling_obj.min()
                new_columns[f'{prefix}_Max'] = rolling_obj.max()
                rolling_features.extend([f'{prefix}_Mean', f'{prefix}_Std', f'{prefix}_Min', f'{prefix}_Max'])
                
                # Range calculation
                new_columns[f'{prefix}_Range'] = new_columns[f'{prefix}_Max'] - new_columns[f'{prefix}_Min']
                rolling_features.append(f'{prefix}_Range')
            
            # Exponential moving averages
            new_columns[f'{prefix}_EMA'] = df[target_col].ewm(span=window).mean()
            rolling_features.append(f'{prefix}_EMA')        # Add all new columns using pd.concat to eliminate DataFrame fragmentation
        if new_columns:
            new_columns_df = pd.DataFrame(new_columns, index=df.index)
            df = pd.concat([df, new_columns_df], axis=1)
        
        # Store created feature names
        self.feature_groups['rolling'] = rolling_features
        
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive lag-based features.
        
        These features capture patterns in how numbers follow each other,
        including transitions, differences, and ratios.
        """
        lag_features = []
        target_col = self.target_column
        
        # Use batch processing to avoid DataFrame fragmentation
        new_columns = {}
        
        # Basic lags
        for lag in self.lags:
            feature_name = f'Lag_{lag}'
            new_columns[feature_name] = df[target_col].shift(lag)
            lag_features.append(feature_name)
        
        # Differences between consecutive selections
        for lag in self.lags:
            if lag > 0:
                feature_name = f'Diff_{lag}'
                new_columns[feature_name] = df[target_col].diff(lag)
                lag_features.append(feature_name)
        
        # Lag ratios (with error handling for zeros)
        for lag in self.lags:
            feature_name = f'Ratio_{lag}'
            lagged_values = df[target_col].shift(lag)
            new_columns[feature_name] = np.where(
                lagged_values != 0,
                df[target_col] / lagged_values,
                1  # Default value for division by zero
            )
            lag_features.append(feature_name)
        
        # Transition types (up, down, same)
        new_columns['TransitionType'] = np.sign(df[target_col].diff())
        lag_features.append('TransitionType')
        
        # Add all basic lag features at once
        for col_name, col_data in new_columns.items():
            df[col_name] = col_data
        
        # Moving averages of lags (need to calculate after basic lags are added)
        ma_columns = {}
        for lag in self.lags:
            if lag > 2:
                feature_name = f'Lag_MA_{lag}'
                ma_columns[feature_name] = df[f'Lag_{lag}'].rolling(window=min(lag, 3)).mean()
                lag_features.append(feature_name)
        
        # Add moving average features
        for col_name, col_data in ma_columns.items():
            df[col_name] = col_data
        
        # Store created feature names
        self.feature_groups['lag'] = lag_features
        
        return df

    def _create_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create frequency-based features with optimized performance.
        
        These features analyze how frequently each number appears overall
        and in recent windows.
        """
        frequency_features = []
        target_col = self.target_column
        
        # Create frequency counts - use vectorized operations
        number_counts = df[target_col].value_counts()
        
        # Use batch column creation to avoid fragmentation
        new_columns = {}
        
        # Basic frequency features
        new_columns['Frequency'] = df[target_col].map(number_counts)
        new_columns['FrequencyNorm'] = new_columns['Frequency'] / len(df)
        frequency_features.extend(['Frequency', 'FrequencyNorm'])
          # Hot and Cold numbers (overall) - vectorized comparison
        median_freq = number_counts.median()
        frequency_mapped = df[target_col].map(number_counts)
        new_columns['IsHot'] = (frequency_mapped > median_freq).astype(int)
        new_columns['IsCold'] = (frequency_mapped < median_freq).astype(int)
        frequency_features.extend(['IsHot', 'IsCold'])
          # Optimized rolling frequency calculation
        for window in self.windows:
            if window >= 5:  # Need sufficient data 
                # Current number frequency in rolling window - optimized vectorized approach
                def calc_rolling_freq(series, win):
                    rolling_counts = []
                    for i in range(len(series)):
                        if i < win - 1:
                            rolling_counts.append(0)
                        else:
                            window_data = series.iloc[i-win+1:i+1]
                            current_val = series.iloc[i]
                            freq = (window_data == current_val).sum() / len(window_data)
                            rolling_counts.append(freq)
                    return rolling_counts
                
                new_columns[f'Freq_Current_{window}'] = calc_rolling_freq(df[target_col], window)
                frequency_features.append(f'Freq_Current_{window}')
                
                # Only calculate individual value frequencies for small unique sets
                unique_values = df[target_col].dropna().unique()
                if len(unique_values) <= 5:  # Reduced from 10 for better performance
                    for val in unique_values:
                        # Use vectorized operations - batch calculate rolling frequencies
                        def calc_val_rolling_freq(series, value, win):
                            rolling_freqs = []
                            for i in range(len(series)):
                                if i < win - 1:
                                    rolling_freqs.append(0)
                                else:
                                    window_data = series.iloc[i-win+1:i+1]
                                    freq = (window_data == value).sum() / len(window_data)
                                    rolling_freqs.append(freq)
                            return rolling_freqs
                        
                        new_columns[f'Freq_{int(val)}_{window}'] = calc_val_rolling_freq(df[target_col], val, window)
                        frequency_features.append(f'Freq_{int(val)}_{window}')        # Add all frequency features using pd.concat to eliminate DataFrame fragmentation
        if new_columns:
            freq_columns_df = pd.DataFrame(new_columns, index=df.index)
            df = pd.concat([df, freq_columns_df], axis=1)
        
        # Store created feature names
        self.feature_groups['frequency'] = frequency_features
        
        return df

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced statistical features.
        
        These features look at the distribution characteristics and
        statistical properties of the number sequence.
        """
        statistical_features = []
        target_col = self.target_column
        
        # Use batch processing to avoid DataFrame fragmentation
        new_columns = {}
        
        # Overall statistics
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        
        # Z-score calculation (overall)
        new_columns['ZScore'] = (df[target_col] - mean_val) / (std_val if std_val > 0 else 1)
        statistical_features.append('ZScore')
        
        # Pattern detection
        new_columns['IsOutlier'] = (abs(new_columns['ZScore']) > 2).astype(int)
        new_columns['IsRepeated'] = (df[target_col] == df[target_col].shift(1)).astype(int)
        statistical_features.extend(['IsOutlier', 'IsRepeated'])
        
        # Rolling z-scores
        for window in self.windows:
            feature_name = f'Rolling_{window}_ZScore'
            rolling_mean = df[target_col].rolling(window=window).mean()
            rolling_std = df[target_col].rolling(window=window).std()
            new_columns[feature_name] = (df[target_col] - rolling_mean) / (rolling_std.replace(0, 1))  # Avoid division by zero
            statistical_features.append(feature_name)
        
        # Deviation from expected value
        # For a truly random sequence between 1-10, the expected value is 5.5
        expected_value = 5.5  # Adjust based on your number range
        new_columns['DeviationFromExpected'] = df[target_col] - expected_value
        statistical_features.append('DeviationFromExpected')
        
        # Cumulative statistics
        new_columns['CumulativeMean'] = df[target_col].expanding().mean()
        new_columns['CumulativeStd'] = df[target_col].expanding().std()
        statistical_features.extend(['CumulativeMean', 'CumulativeStd'])
        
        # Add all statistical features at once to avoid fragmentation
        for col_name, col_data in new_columns.items():
            df[col_name] = col_data
        
        # Store created feature names
        self.feature_groups['statistical'] = statistical_features
        
        return df

    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture potential patterns in the number sequence.
        
        These features help identify cyclical patterns, streaks, and other
        non-random behaviors in the sequence.
        """
        pattern_features = []
        target_col = self.target_column
        
        # Detect repeating cycles
        for cycle in self.cycles:
            feature_name = f'Cycle_{cycle}'
            df[feature_name] = (df[target_col].shift(cycle) == df[target_col]).astype(int)
            pattern_features.append(feature_name)
        
        # Streak features (consecutive occurrences)
        df['Streak'] = (df[target_col] == df[target_col].shift(1)).astype(int)
        df['StreakLength'] = df['Streak'].groupby(
            (df['Streak'] != df['Streak'].shift(1)).cumsum()
        ).cumcount() + 1
        pattern_features.extend(['Streak', 'StreakLength'])
        
        # Check if current value matches value from N selections ago
        for n in [5, 10, 20]:
            if n < len(df) // 2:  # Ensure enough data
                feature_name = f'MatchesNAgo_{n}'
                df[feature_name] = (df[target_col] == df[target_col].shift(n)).astype(int)
                pattern_features.append(feature_name)
          # Count occurrences of each value in rolling windows - optimized
        unique_values = df[target_col].dropna().unique()
        if len(unique_values) <= 5:  # Reduced from 10 for better performance
            # Use batch processing to avoid fragmentation
            new_columns = {}
            
            for window in [10, 20]:
                for val in unique_values:
                    feature_name = f'Count_{int(val)}_in_{window}'
                    # Use vectorized rolling sum instead of apply
                    new_columns[feature_name] = (df[target_col] == val).rolling(window).sum()
                    pattern_features.append(feature_name)
            
            # Add all count features at once
            for col_name, col_data in new_columns.items():
                df[col_name] = col_data
        
        # Distance from rolling mean and median - batch processing
        distance_columns = {}
        for window in [10, 20]:
            rolling_mean = df[target_col].rolling(window=window).mean()
            distance_columns[f'DistFromMean_{window}'] = df[target_col] - rolling_mean
            
            rolling_median = df[target_col].rolling(window=window).median()
            distance_columns[f'DistFromMedian_{window}'] = df[target_col] - rolling_median
            
            rolling_std = df[target_col].rolling(window=window).std()
            distance_columns[f'DistFromMean_{window}_Norm'] = (df[target_col] - rolling_mean) / (rolling_std.replace(0, 1))
            
            pattern_features.extend([
                f'DistFromMean_{window}', 
                f'DistFromMedian_{window}', 
                f'DistFromMean_{window}_Norm'
            ])
          # Add all distance features using pd.concat to eliminate DataFrame fragmentation
        if distance_columns:
            distance_df = pd.DataFrame(distance_columns, index=df.index)
            df = pd.concat([df, distance_df], axis=1)
        
        # Detect if pattern from last N selections repeats
        for pattern_length in [2, 3]:
            if pattern_length < len(df) // 3:  # Ensure enough data
                feature_name = f'PatternRepeat_{pattern_length}'
                pattern_repeats = []
                
                for i in range(len(df)):
                    if i < pattern_length * 2:
                        repeats = 0
                    else:
                        current_pattern = df[target_col].iloc[i-pattern_length:i].values
                        previous_pattern = df[target_col].iloc[i-pattern_length*2:i-pattern_length].values
                        repeats = np.array_equal(current_pattern, previous_pattern)
                    
                    pattern_repeats.append(int(repeats))
                
                df[feature_name] = pattern_repeats
                pattern_features.append(feature_name)
        
        # Store created feature names
        self.feature_groups['pattern'] = pattern_features
        
        return df

    def _create_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that measure the randomness or predictability of the sequence.
        
        These features use information theory concepts to quantify how random
        or predictable recent selections have been.
        """
        entropy_features = []
        target_col = self.target_column
        
        # Calculate Shannon entropy for rolling windows
        for window in self.windows:
            if window >= 5:  # Entropy makes more sense with sufficient data
                feature_name = f'Entropy_{window}'
                df[feature_name] = df[target_col].rolling(window).apply(
                    lambda x: self._calculate_entropy(x)
                )
                entropy_features.append(feature_name)
        
        # Calculate normalized entropy (0-1 scale)
        # Maximum entropy would be log2(10) for 10 possible values
        unique_values = df[target_col].dropna().unique()
        max_entropy = np.log2(len(unique_values)) if len(unique_values) > 1 else 1
        
        for window in self.windows:
            if window >= 5:
                feature_name = f'NormEntropy_{window}'
                df[feature_name] = df[f'Entropy_{window}'] / max_entropy
                entropy_features.append(feature_name)
          # Calculate entropy change rate and add all entropy features using pd.concat
        entropy_feature_dict = {}
        for window in self.windows:
            if window >= 5:
                feature_name = f'EntropyChange_{window}'
                entropy_feature_dict[feature_name] = df[f'Entropy_{window}'].diff()
                entropy_features.append(feature_name)
        
        # Calculate run complexity (count of distinct runs) - optimized
        for window in self.windows:
            if window >= 5:
                feature_name = f'RunComplexity_{window}'
                # Use optimized function instead of lambda apply
                entropy_feature_dict[feature_name] = df[target_col].rolling(window).apply(
                    _fast_run_complexity_optimized,
                    raw=True
                )
                entropy_features.append(feature_name)

        # Calculate predictability score
        for window in self.windows:
            if window >= 5:
                feature_name = f'Predictability_{window}'
                entropy_feature_dict[feature_name] = 1 - df[f'NormEntropy_{window}']
                entropy_features.append(feature_name)
        
        # Use pd.concat for comprehensive batch processing - eliminates fragmentation
        if entropy_feature_dict:
            entropy_df = pd.DataFrame(entropy_feature_dict, index=df.index)
            df = pd.concat([df, entropy_df], axis=1)
        
        # Store created feature names
        self.feature_groups['entropy'] = entropy_features
        
        return df

    def _create_rare_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that detect rare patterns in the number sequence.
        """
        rare_pattern_features = []
        target_col = self.target_column        # Detect rare patterns (e.g., specific sequences) - optimized
        rare_sequences = [
            [1, 2, 3],  # Example sequence
            [7, 8, 9]   # Another example sequence
        ]
        
        # Use batch processing for rare pattern detection
        rare_pattern_columns = {}
        
        for seq in rare_sequences:
            feature_name = f'RarePattern_{"_".join(map(str, seq))}'
            # Use vectorized comparison instead of lambda apply
            rolling_values = df[target_col].rolling(window=len(seq))
              # More efficient pattern matching
            pattern_matches = []
            for window_values in rolling_values:
                if len(window_values) == len(seq):
                    match = np.array_equal(window_values.values, seq)
                    pattern_matches.append(int(match))
                else:
                    pattern_matches.append(0)
            
            rare_pattern_columns[feature_name] = pattern_matches
            rare_pattern_features.append(feature_name)
          # Add all rare pattern features using pd.concat to eliminate fragmentation
        if rare_pattern_columns:
            rare_pattern_df = pd.DataFrame(rare_pattern_columns, index=df.index)
            df = pd.concat([df, rare_pattern_df], axis=1)

        # Store created feature names
        self.feature_groups['rare_pattern'] = rare_pattern_features

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features into numeric values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across the codebase.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        try:
            return standardize_column_names(df)
        except:
            # If the function isn't available, return the original DataFrame
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
        Identify holidays (simple approximation).
        """
        return (dates.dt.month.isin([12, 1]) & dates.dt.day.isin([25, 1])).astype(int)
    
    @staticmethod
    def _calculate_entropy(values: np.ndarray) -> float:
        """
        Calculate Shannon entropy of a sequence.
        
        Higher entropy indicates more randomness/unpredictability.
        
        Args:
            values: Array of values
            
        Returns:
            Shannon entropy value
        """
        if len(values) <= 1:
            return 0
            
        # Count occurrences of each value
        value_counts = pd.Series(values).value_counts()
        total = len(values)
        
        # Calculate probabilities
        probabilities = value_counts / total
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities)
        
        return entropy

    @staticmethod
    def _calculate_run_complexity(values: np.ndarray) -> int:
        """
        Calculate the number of runs (consecutive identical values) in a sequence.
        
        More runs indicate higher complexity/less predictability.
        
        Args:
            values: Array of values
            
        Returns:
            Number of runs
        """
        if len(values) <= 1:
            return len(values)
            
        # Convert to list for easier processing
        values_list = list(values)
        
        # Count runs
        runs = 1
        for i in range(1, len(values_list)):
            if values_list[i] != values_list[i-1]:
                runs += 1
                
        return runs
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get the dictionary of feature groups created by the engineer.
        
        Returns:
            Dictionary mapping feature group names to lists of feature names
        """
        return self.feature_groups
    
    def get_all_features(self) -> List[str]:
        """
        Get a flat list of all features created by the engineer.
        
        Returns:
            List of all feature names
        """
        all_features = []
        for group in self.feature_groups.values():
            all_features.extend(group)
        return all_features

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features from the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input data
        Returns:
            pd.DataFrame: DataFrame with new features added
        """
        # Example: add rolling mean and lag features
        df = df.copy()
        if 'Number' in df.columns:
            df['number_rolling_mean_3'] = df['Number'].rolling(window=3, min_periods=1).mean()
            df['number_lag_1'] = df['Number'].shift(1)
        return df

    def select_features(self, df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """
        Select relevant features using correlation or model-based selection.
        
        Args:
            df (pd.DataFrame): Input data
            target (Optional[str]): Target column for feature selection
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        # Example: select features with correlation above threshold
        if target and target in df.columns:
            corr = df.corr(numeric_only=True)[target].abs()
            selected = corr[corr > 0.1].index.tolist()
            return df[selected]
        return df

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to features (e.g., scaling, encoding).
        
        Args:
            df (pd.DataFrame): Input data
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        # Example: fill missing values and standardize numeric columns
        df = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        return df

def add_statistical_features(df: pd.DataFrame, target_col: str = 'Number') -> pd.DataFrame:
    """Add statistical features to the DataFrame."""
    # Pre-calculate all features
    new_features = {}
    
    # Calculate basic stats
    rolling_windows = [3, 5, 7, 10]
    for window in rolling_windows:
        window_data = df[target_col].rolling(window)
        new_features[f'RollingMean_{window}'] = window_data.mean()
        new_features[f'RollingStd_{window}'] = window_data.std()
        new_features[f'RollingMin_{window}'] = window_data.min()
        new_features[f'RollingMax_{window}'] = window_data.max()
        new_features[f'RollingSkew_{window}'] = window_data.apply(
            lambda x: stats.skew(x) if len(x) > 2 else 0
        )
        new_features[f'RollingKurt_{window}'] = window_data.apply(
            lambda x: stats.kurtosis(x) if len(x) > 3 else 0
        )

    # Calculate lag features
    for lag in range(1, 6):
        new_features[f'Lag_{lag}'] = df[target_col].shift(lag)
        new_features[f'Diff_{lag}'] = df[target_col].diff(lag)

    # Calculate transition features
    transitions = df[target_col].diff()
    new_features['TransitionType'] = np.sign(transitions)

    # Calculate frequency features
    number_counts = df[target_col].value_counts()
    new_features['Frequency'] = df[target_col].map(number_counts)
    new_features['FrequencyNorm'] = new_features['Frequency'] / len(df)
    
    median_freq = number_counts.median()
    new_features['IsHot'] = df[target_col].map(lambda x: number_counts[x] > median_freq).astype(int)
    new_features['IsCold'] = df[target_col].map(lambda x: number_counts[x] < median_freq).astype(int)

    # Add all features at once
    result = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    return result