from typing import Dict, Any
import pandas as pd
from scipy.stats import chi2

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
    # ...existing code...
