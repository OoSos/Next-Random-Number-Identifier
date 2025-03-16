from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
from .base_model import BaseModel
from scipy.stats import zscore, norm

class MarkovChain(BaseModel):
    """
    Enhanced Markov Chain implementation for random number prediction.
    Supports variable order and multiple prediction strategies.
    """
    
    def __init__(self, order: int = 2, smoothing: float = 0.1, **kwargs):
        """
        Initialize Markov Chain model.
        
        Args:
            order (int): Order of the Markov Chain (memory length)
            smoothing (float): Laplace smoothing parameter
            **kwargs: Additional parameters
        """
        # Set a default name if not provided in kwargs
        name = kwargs.pop('name', "MarkovChain")
        super().__init__(name=name, **kwargs)
        self.order = order
        self.smoothing = smoothing
        self.transition_matrix: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.state_counts: Dict[Tuple, int] = defaultdict(int)
        self.total_transitions = 0
        self.unique_numbers: Optional[List[int]] = None
        self.performance_metrics: Dict[str, float] = {}

    def _create_sequence(self, numbers: pd.Series) -> List[Tuple[Tuple, int]]:
        """Create state-next_state pairs from input sequence."""
        sequences = []
        numbers_list = numbers.tolist()
    
        for i in range(len(numbers_list) - self.order):
            current_state = tuple(numbers_list[i:i + self.order])
            next_state = numbers_list[i + self.order]
            sequences.append((current_state, next_state))
    
        return sequences

    def fit(self, X: pd.DataFrame, y: pd.Series, disable_smoothing: bool = False) -> 'MarkovChain':
        """
        Fit Markov Chain model to the training data.
        
        Args:
            X (pd.DataFrame): Not used in Markov Chain (kept for consistency)
            y (pd.Series): Training sequence
            disable_smoothing (bool): If True, disable Laplace smoothing (for testing)
            
        Returns:
            self: The fitted model instance
        """
        self.unique_numbers = sorted(y.unique())
        sequences = self._create_sequence(y)
        
        # Count transitions
        for state, next_state in sequences:
            self.state_counts[state] += 1
            self.transition_matrix[state][next_state] += 1
            self.total_transitions += 1
        
        # Convert counts to probabilities with smoothing
        for state in self.transition_matrix:
            smoothing_factor = 0 if disable_smoothing else self.smoothing
            total = sum(self.transition_matrix[state].values()) + smoothing_factor * len(self.unique_numbers)
            for next_state in self.unique_numbers:
                count = self.transition_matrix[state][next_state] + smoothing_factor
                self.transition_matrix[state][next_state] = count / total
                
        return self

    def optimize_order(self, X: pd.DataFrame, y: pd.Series, max_order: int = 5) -> int:
        """
        Optimize the order of the Markov Chain based on performance metrics.
        
        Args:
            X (pd.DataFrame): Not used in Markov Chain (kept for consistency)
            y (pd.Series): Training sequence
            max_order (int): Maximum order to try
            
        Returns:
            int: Best performing order
        """
        best_order = 1
        best_accuracy = 0.0
        
        for order in range(1, max_order + 1):
            self.order = order
            self.fit(X, y)
            accuracy = self.evaluate(X, y)['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_order = order
        
        self.order = best_order
        self.fit(X, y)
        return best_order

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Contains recent history for prediction
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Get the most recent state from X
        if 'Number' in X.columns and len(X) >= self.order:
            current_state = tuple(X['Number'].tail(self.order))
        else:
            # If we don't have enough history, return random predictions
            return np.array([np.random.choice(self.unique_numbers) for _ in range(len(X))])
            
        predictions = []
        
        for _ in range(len(X)):
            if current_state in self.transition_matrix:
                # Extract probabilities for each possible next state
                probabilities = [self.transition_matrix[current_state][n] for n in self.unique_numbers]
                predicted_number = np.random.choice(self.unique_numbers, p=probabilities)
            else:
                # If state not seen in training, use uniform distribution
                predicted_number = np.random.choice(self.unique_numbers)
            
            predictions.append(predicted_number)
            
            # Update the current state for the next prediction
            if self.order > 1:
                current_state = current_state[1:] + (predicted_number,)
            else:
                current_state = (predicted_number,)
            
        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get transition probabilities for the next state.
        
        Args:
            X (pd.DataFrame): Contains recent history for prediction
            
        Returns:
            np.ndarray: Probability matrix for next states
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before calculating probabilities")
            
        if 'Number' in X.columns and len(X) >= self.order:
            current_state = tuple(X['Number'].tail(self.order))
        else:
            # Return uniform probabilities if we don't have enough history
            return np.array([1.0 / len(self.unique_numbers)] * len(self.unique_numbers))
        
        if current_state in self.transition_matrix:
            probabilities = [self.transition_matrix[current_state][n] for n in self.unique_numbers]
        else:
            # Uniform distribution for unseen states
            probabilities = [1.0 / len(self.unique_numbers)] * len(self.unique_numbers)
            
        return np.array(probabilities)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance. For Markov Chain, this is a placeholder
        since traditional classification metrics don't apply directly.
        
        Args:
            X (pd.DataFrame): Contains historical data
            y (pd.Series): True target values
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Calculate simple accuracy by comparing next-state predictions
        correct = 0
        total = 0
        
        for i in range(len(X) - self.order):
            current_state = tuple(X.iloc[i:i+self.order]['Number'].tolist())
            actual_next = y.iloc[i+self.order]
            
            if current_state in self.transition_matrix:
                probabilities = [self.transition_matrix[current_state][n] for n in self.unique_numbers]
                predicted_idx = np.argmax(probabilities)
                predicted_next = self.unique_numbers[predicted_idx]
                
                if predicted_next == actual_next:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        metrics = {'accuracy': accuracy}
        
        self.performance_metrics = metrics
        return metrics

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the transition probability matrix.
        
        Returns:
            pd.DataFrame: Transition probability matrix
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before getting transition matrix")
            
        matrix_dict = {}
        for state in self.transition_matrix:
            matrix_dict[state] = {n: self.transition_matrix[state][n] 
                                for n in self.unique_numbers}
            
        return pd.DataFrame.from_dict(matrix_dict, orient='index')

    def get_most_probable_transitions(self, top_n: int = 5) -> List[Tuple[Tuple, int, float]]:
        """
        Get the most probable state transitions.
        
        Args:
            top_n (int): Number of transitions to return
            
        Returns:
            List[Tuple[Tuple, int, float]]: List of (state, next_state, probability)
        """
        transitions = []
        for state in self.transition_matrix:
            for next_state, prob in self.transition_matrix[state].items():
                transitions.append((state, next_state, prob))
                
        return sorted(transitions, key=lambda x: x[2], reverse=True)[:top_n]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for Markov Chain model based on transition probabilities.
        Uses both state frequency and transition probability distributions.
        
        Returns:
            Dict[str, float]: Dictionary mapping state patterns to their importance scores
        
        Raises:
            ValueError: If the model hasn't been trained yet
        """
        if not self.transition_matrix:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_dict: Dict[str, float] = {}
    
        # Calculate importance based on a combination of:
        # 1. State frequency (how common is this state)
        # 2. Transition entropy (how predictable is the next state)
        total_states = sum(self.state_counts.values())
        
        for state in self.transition_matrix:
            # Frequency component
            state_freq = self.state_counts[state] / total_states if total_states > 0 else 0
            
            # Entropy/predictability component (variance of transition probabilities)
            probs = list(self.transition_matrix[state].values())
            if probs:
                predictability = float(np.std(probs))
                
                # Combine both metrics - states that are both common and have predictable
                # transitions are most important
                importance = state_freq * predictability
                state_str = f"state_{'_'.join(map(str, state))}"
                importance_dict[state_str] = importance
    
        # Normalize importance scores
        if importance_dict:
            max_importance = max(importance_dict.values())
            if max_importance > 0:
                importance_dict = {
                    k: float(v / max_importance) 
                    for k, v in importance_dict.items()
                }
                return importance_dict
    
        # Return empty dict if no valid importance scores
        return {}

    def runs_test(self, sequence: pd.Series) -> Dict[str, float]:
        """
        Perform runs test for randomness on the given sequence.
        """
        n1 = sum(sequence > sequence.median())
        n2 = sum(sequence <= sequence.median())
        runs = 1 + sum((sequence[:-1].reset_index(drop=True) > sequence.median()) != (sequence[1:].reset_index(drop=True) > sequence.median()))
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        z = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return {'z': z, 'p_value': p_value}

    def serial_test(self, sequence: pd.Series, lag: int = 1) -> Dict[str, float]:
        """
        Perform serial test for randomness on the given sequence.
        """
        n = len(sequence)
        pairs = [(sequence[i], sequence[i + lag]) for i in range(n - lag)]
        unique_pairs = len(set(pairs))
        expected_pairs = (n - lag) / 2
        variance_pairs = (n - lag) * (n - lag - 1) / 4
        z = (unique_pairs - expected_pairs) / np.sqrt(variance_pairs)
        p_value = 2 * (1 - norm.cdf(abs(z)))
        return {'z': z, 'p_value': p_value}

    def visualize_transition_matrix(self) -> None:
        """
        Visualize the transition matrix using a heatmap.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        if self.transition_matrix is None:
            raise ValueError("Model must be trained before visualizing transition matrix")

        matrix_df = self.get_transition_matrix()
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Transition Matrix Heatmap")
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        plt.show()

    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the Markov Chain model.
        """
        report = []
        report.append(f"Model: {self.name}")
        report.append(f"Order: {self.order}")
        report.append(f"Smoothing: {self.smoothing}")
        report.append(f"Total Transitions: {self.total_transitions}")
        report.append(f"Unique Numbers: {self.unique_numbers}")
        report.append(f"Performance Metrics: {self.performance_metrics}")
        report.append(f"Feature Importance: {self.get_feature_importance()}")
        return "\n".join(report)

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence based on transition probabilities.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Confidence scores for each prediction
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before estimating confidence")
            
        # Get the most recent state from X
        if 'Number' in X.columns and len(X) >= self.order:
            current_state = tuple(X['Number'].tail(self.order))
        else:
            # If we don't have enough history, return low confidence
            return np.ones(len(X)) * 0.1
            
        # Calculate confidence based on the distribution of transition probabilities
        confidences = []
        
        for _ in range(len(X)):
            if current_state in self.transition_matrix:
                # Extract probabilities for each possible next state
                probabilities = [self.transition_matrix[current_state][n] for n in self.unique_numbers]
                
                # Higher max probability indicates higher confidence
                max_prob = max(probabilities)
                
                # Alternative: use entropy as a measure of confidence
                # entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                # max_entropy = np.log2(len(probabilities))
                # confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                
                # Use the maximum probability as confidence
                confidences.append(max_prob)
            else:
                # If state not seen in training, use low confidence
                confidences.append(0.1)
                
            # Update the current state for the next prediction (using the most likely next state)
            if current_state in self.transition_matrix:
                probabilities = [self.transition_matrix[current_state][n] for n in self.unique_numbers]
                next_state = self.unique_numbers[np.argmax(probabilities)]
                
                # Update current state
                if self.order > 1:
                    current_state = current_state[1:] + (next_state,)
                else:
                    current_state = (next_state,)
        
        return np.array(confidences)

class EnhancedMarkovChain(BaseModel):
    """
    Enhanced Markov Chain with variable order selection and bayesian estimation.
    """
    def __init__(self, max_order: int = 3, smoothing: float = 0.1):
        """
        Initialize EnhancedMarkovChain model.
        
        Args:
            max_order: Maximum chain order to try
            smoothing: Laplace smoothing parameter
        """
        super().__init__(name="EnhancedMarkovChain")
        self.max_order = max_order
        self.smoothing = smoothing
        self.transition_matrices = {}  # One matrix per order
        self.order_performances = {}   # Track performance by order
        self.best_order = None         # Best performing order
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnhancedMarkovChain':
        """
        Fit Markov Chain models of different orders and select best performer.
        """
        # Implement order selection logic
        # Train transition matrices for each order
        # Measure performance on validation set
        # Select best order
        
        return self