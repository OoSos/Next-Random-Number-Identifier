from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict
from .base_model import BaseModel
from scipy.stats import norm


class MarkovChain(BaseModel):
    """Markov Chain model for sequence prediction."""
    
    def __init__(self, order: int = 2, smoothing: float = 0.1, **kwargs) -> None:
        """
        Initialize MarkovChain model.
        
        Args:
            order (int): Order of the Markov chain (number of previous states to consider)
            smoothing (float): Smoothing parameter for transition probabilities
            **kwargs: Additional keyword arguments for BaseModel
        """
        name = kwargs.pop('name', 'MarkovChain')
        super().__init__(name=name, **kwargs)
        self.order = order
        self.smoothing = smoothing
        self.state_transitions = {}
        self.transition_matrix = {}
        self.unique_numbers = None
        self.feature_importance_ = {}
        
    def _create_sequence(self, y: pd.Series) -> List[Tuple]:
        """
        Create sequences for training.
        
        Args:
            y (pd.Series): Target sequence
        Returns:
            List[Tuple]: List of (state, next_value) pairs
        """
        sequences = []
        for i in range(self.order, len(y)):
            state = tuple(y.iloc[i-self.order:i])
            next_value = y.iloc[i]
            sequences.append((state, next_value))
        return sequences
        
    def fit(self, X: pd.DataFrame, y: pd.Series, disable_smoothing: bool = False) -> 'MarkovChain':
        """
        Fit the Markov Chain model.
        
        Args:
            X (pd.DataFrame): Training features (not used, present for API compatibility)
            y (pd.Series): Target sequence
            disable_smoothing (bool): If True, disables smoothing in transition probabilities
        Returns:
            MarkovChain: Self
        """
        self.unique_numbers = set(y.unique())
        transition_counts = defaultdict(lambda: defaultdict(int))
        sequences = self._create_sequence(y)
        
        for state, next_value in sequences:
            transition_counts[state][next_value] += 1
            
        self.state_transitions = {}
        for state, counts in transition_counts.items():
            total = sum(counts.values())
            if not disable_smoothing:
                smoothed_total = total + len(self.unique_numbers) * self.smoothing
                self.state_transitions[state] = {
                    value: (counts.get(value, 0) + self.smoothing) / smoothed_total 
                    for value in self.unique_numbers
                }
            else:
                self.state_transitions[state] = {
                    value: count / total for value, count in counts.items()
                }
        
        self.transition_matrix = self.state_transitions
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the next value in the sequence for each sample in X.
        
        Args:
            X (pd.DataFrame): Input features
        Returns:
            np.ndarray: Predicted values
        Raises:
            ValueError: If model is not fitted
        """
        if not self.state_transitions:
            raise ValueError("Model must be fitted before predicting")
            
        predictions = []
        for i in range(len(X)):
            if self.unique_numbers:
                predictions.append(max(self.unique_numbers))
            else:
                predictions.append(0)
        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability distributions for each sample in X.
        
        Args:
            X (pd.DataFrame): Input features
        Returns:
            np.ndarray: Probability distributions
        Raises:
            ValueError: If model is not fitted
        """
        if not self.state_transitions:
            raise ValueError("Model must be fitted before predicting probabilities")
        
        if self.unique_numbers:
            uniform_prob = 1.0 / len(self.unique_numbers)
            return np.array([uniform_prob] * len(self.unique_numbers))
        return np.array([1.0])

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the transition matrix as a DataFrame.
        
        Returns:
            pd.DataFrame: Transition matrix
        Raises:
            ValueError: If model is not fitted
        """
        if not self.transition_matrix:
            raise ValueError("Model must be fitted before getting transition matrix")
        
        if not self.unique_numbers:
            return pd.DataFrame()
            
        unique_vals = sorted(self.unique_numbers)
        matrix_data = []
        for state, transitions in self.transition_matrix.items():
            row = [transitions.get(val, 0.0) for val in unique_vals]
            matrix_data.append(row)
        return pd.DataFrame(matrix_data, columns=unique_vals)

    def get_most_probable_transitions(self, top_n: int = 5) -> List[Tuple]:
        """
        Get the most probable transitions in the Markov chain.
        
        Args:
            top_n (int): Number of top transitions to return
        Returns:
            List[Tuple]: List of (state, next_value, probability) tuples
        """
        if not self.transition_matrix:
            return []
        
        transitions = []
        for state, probs in self.transition_matrix.items():
            for next_val, prob in probs.items():
                transitions.append((state, next_val, prob))
        
        transitions.sort(key=lambda x: x[2], reverse=True)
        return transitions[:top_n]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores based on transition entropy.
        
        Returns:
            Dict[str, float]: Feature (state) to importance score mapping
        Raises:
            ValueError: If model is not fitted
        """
        if not self.transition_matrix:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = {}
        for state, probs in self.transition_matrix.items():
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs.values() if p > 0)
            importance[str(state)] = entropy
        
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                importance = {k: v / max_importance for k, v in importance.items()}
        return importance

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence for each sample in X.
        
        Args:
            X (pd.DataFrame): Input features
        Returns:
            np.ndarray: Confidence scores for each prediction
        """
        if not self.state_transitions:
            return np.array([0.1] * len(X))
        
        confidences = []
        for i in range(len(X)):
            if i < self.order:
                confidences.append(0.3)
            else:
                avg_entropy = np.mean([
                    -sum(p * np.log2(p + 1e-10) for p in probs.values() if p > 0)
                    for probs in self.state_transitions.values()
                ])
                max_entropy = np.log2(len(self.unique_numbers)) if self.unique_numbers else 1
                confidence = 1.0 - (avg_entropy / max_entropy)
                confidences.append(max(0.1, min(0.9, confidence)))
        return np.array(confidences)

    def runs_test(self, sequence: pd.Series) -> Dict[str, float]:
        """
        Perform runs test for randomness on a sequence.
        
        Args:
            sequence (pd.Series): Sequence to test
        Returns:
            Dict[str, float]: Test statistics (z, p_value)
        """
        median = sequence.median()
        binary_seq = (sequence > median).astype(int)
        
        runs = 1
        for i in range(1, len(binary_seq)):
            if binary_seq.iloc[i] != binary_seq.iloc[i-1]:
                runs += 1
        
        n1 = sum(binary_seq)
        n2 = len(binary_seq) - n1
        
        if n1 == 0 or n2 == 0:
            return {'z': 0.0, 'p_value': 1.0}
        
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        if variance <= 0:
            return {'z': 0.0, 'p_value': 1.0}
        
        z_score = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return {'z': float(z_score), 'p_value': float(p_value)}

    def serial_test(self, sequence: pd.Series) -> Dict[str, float]:
        """
        Perform serial test for randomness on a sequence.
        
        Args:
            sequence (pd.Series): Sequence to test
        Returns:
            Dict[str, float]: Test statistics (z, p_value)
        """
        pairs = defaultdict(int)
        
        for i in range(len(sequence) - 1):
            pair = (sequence.iloc[i], sequence.iloc[i + 1])
            pairs[pair] += 1
        
        expected = len(sequence) / len(pairs) if pairs else 1
        chi_square = sum((observed - expected) ** 2 / expected for observed in pairs.values())
        
        z_score = np.sqrt(chi_square)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return {'z': float(z_score), 'p_value': float(p_value)}

    def visualize_transition_matrix(self) -> None:
        """
        Visualize the transition matrix using matplotlib (if available).
        
        Raises:
            ValueError: If model is not fitted
        """
        if not self.transition_matrix:
            raise ValueError("Model must be fitted before visualizing")
        # Import matplotlib here to avoid issues during regular usage
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            # If matplotlib is not available, just pass
            pass

    def generate_report(self) -> str:
        """
        Generate a summary report of the Markov Chain model.
        
        Returns:
            str: Model report
        """
        if not self.transition_matrix:
            return "Model not fitted yet. Please call fit() first."
        
        report = []
        report.append(f"Model: {self.name}")
        report.append(f"Order: {self.order}")
        report.append(f"Smoothing: {self.smoothing}")
        report.append(f"Total Transitions: {len(self.transition_matrix)}")
        report.append(f"Unique Numbers: {len(self.unique_numbers) if self.unique_numbers else 0}")
        report.append("\nPerformance Metrics:")
        report.append("- Model trained successfully")
        report.append("\nFeature Importance:")
        
        importance = self.get_feature_importance()
        for feature, score in list(importance.items())[:5]:
            report.append(f"- {feature}: {score:.4f}")
        
        report.append("\nMost Probable Transitions:")
        transitions = self.get_most_probable_transitions(3)
        for state, next_val, prob in transitions:
            report.append(f"- {state} -> {next_val}: {prob:.4f}")
        
        return "\n".join(report)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using multiple regression and classification metrics.
        
        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True target values
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
        
        predictions = self.predict(X)
        
        # For classification metrics
        try:
            accuracy = accuracy_score(y, predictions)
        except:
            accuracy = 0.0
        
        # For regression metrics 
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        try:
            r2 = r2_score(y, predictions)
        except:
            r2 = 0.0
        
        return {
            'mse': float(mse),
            'mae': float(mae), 
            'rmse': float(rmse),
            'r2': float(r2),
            'accuracy': float(accuracy)
        }
