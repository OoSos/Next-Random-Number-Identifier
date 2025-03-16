from typing import Dict, Any
import pandas as pd
from scipy.stats import chi2
import numpy as np

class StatisticalAnalyzer:
    """
    Statistical analysis tools for random number sequences.
    """
    def __init__(self):
        self.analysis_results = {}

    def analyze_frequency_distribution(self, numbers: pd.Series) -> Dict[str, Any]:
        """
        Analyze the frequency distribution of numbers.
        """
        # Count occurrences
        value_counts = numbers.value_counts().sort_index()

        # Calculate expected frequencies for uniform distribution
        n = len(numbers)
        unique_values = sorted(numbers.unique())
        k = len(unique_values)
        expected = n / k

        # Chi-square test for uniformity
        chi2_stat = sum([(observed - expected)**2 / expected 
                        for observed in value_counts.values])

        # Calculate p-value
        p_value = 1 - chi2.cdf(chi2_stat, df=k-1)

        return {
            'value_counts': value_counts.to_dict(),
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'is_uniform': p_value > 0.05  # Null hypothesis: distribution is uniform
        }

    def analyze_time_series(self, numbers: pd.Series) -> Dict[str, Any]:
        """
        Analyze the time series properties of numbers.
        """
        # Calculate autocorrelation
        autocorrelation = [numbers.autocorr(lag) for lag in range(1, 11)]

        return {
            'autocorrelation': autocorrelation
        }

    def analyze_markov_chain(self, numbers: pd.Series, order: int = 1) -> Dict[str, Any]:
        """
        Analyze the Markov chain properties of numbers.
        """
        # Create transition matrix
        transitions = np.zeros((10, 10))
        for (i, j) in zip(numbers[:-1], numbers[1:]):
            transitions[i-1, j-1] += 1

        # Normalize to get probabilities
        transition_probabilities = transitions / transitions.sum(axis=1, keepdims=True)

        return {
            'transition_matrix': transition_probabilities
        }
