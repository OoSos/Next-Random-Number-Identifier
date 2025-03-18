from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict
from .base_model import BaseModel
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MarkovChain(BaseModel):
    """
    Enhanced Markov Chain implementation for random number prediction.
    
    This implementation supports variable order Markov Chain modeling with
    Laplace smoothing for robust probability estimation. It includes methods
    for prediction, probability estimation, and feature importance calculation.
    
    Attributes:
        order: Memory length of the Markov Chain
        smoothing: Laplace smoothing parameter
        transition_matrix: Dictionary mapping states to next-state probabilities
        state_counts: Dictionary tracking frequency of each state
        unique_numbers: List of unique numbers in the training data
        performance_metrics: Dictionary of performance metrics
    """
    
    def __init__(self, order: int = 2, smoothing: float = 0.1, **kwargs):
        """
        Initialize the Markov Chain model.
        
        Args:
            order: Order of the Markov Chain (memory length)
            smoothing: Laplace smoothing parameter for probability estimation
            **kwargs: Additional parameters to pass to the base model
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
        """
        Create state-next_state pairs from input sequence.
        
        Args:
            numbers: Series of numbers to process
            
        Returns:
            List of (state, next_state) tuples where state is a tuple of length self.order
        """
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
            X: Training features (not used in Markov Chain but kept for API consistency)
            y: Training sequence of numbers
            disable_smoothing: If True, disable Laplace smoothing (for testing)
            
        Returns:
            self: The fitted model instance
        """
        # Store unique numbers and ensure they're integers
        self.unique_numbers = sorted(y.unique())
        
        # Create state-next_state sequences
        sequences = self._create_sequence(y)
        
        # Reset counters
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.state_counts = defaultdict(int)
        self.total_transitions = 0
        
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        For Markov Chain, this returns the most probable next number for each row in X,
        based on the most recent numbers (specified by order).
        
        Args:
            X: Features containing recent history for prediction
        Returns:
            np.ndarray: Predicted values
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        # Get the most recent numbers to form the initial state
        recent_numbers = self._get_recent_numbers(X)
        current_state = tuple(recent_numbers[-self.order:]) if len(recent_numbers) >= self.order else None
        
        for _ in range(len(X)):
            if current_state in self.transition_matrix:
                # Get probabilities for all possible next states
                probs = np.array([self.transition_matrix[current_state][n] for n in self.unique_numbers])
                # Predict the most probable next number
                predicted_number = self.unique_numbers[np.argmax(probs)]
            else:
                # If state not seen in training, use the most frequent number
                predicted_number = self.unique_numbers[0]  # Default to first number
            
            predictions.append(predicted_number)
            
            # Update state for next prediction if needed
            if len(X) > 1:
                if self.order > 1 and current_state is not None:
                    current_state = current_state[1:] + (predicted_number,)
                else:
                    current_state = (predicted_number,)
        
        return np.array(predictions)
    
    def _get_recent_numbers(self, X: pd.DataFrame) -> List[int]:
        """
        Extract recent numbers from the input features.
        
        Args:
            X: Input features
            
        Returns:
            List of recent numbers
        """
        # Try to get recent numbers from different possible column formats
        if 'Number' in X.columns:
            return X['Number'].tolist()
        
        # Look for lag features which might contain recent numbers
        lag_columns = [col for col in X.columns if col.startswith('Lag_')]
        if lag_columns:
            # Sort by lag number to get the most recent first
            lag_columns.sort(key=lambda x: int(x.split('_')[1]))
            recent_numbers = []
            for col in lag_columns:
                if len(recent_numbers) < self.order:
                    values = X[col].fillna(method='ffill').iloc[-1]
                    recent_numbers.append(values)
            return recent_numbers
        
        # If we can't find recent numbers, return an empty list
        return []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get transition probabilities for the next state.
        
        Args:
            X: Features containing recent history
            
        Returns:
            np.ndarray: Probability matrix for each possible next number
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before calculating probabilities")
        
        # Get recent numbers to form the current state
        recent_numbers = self._get_recent_numbers(X)
        
        if len(recent_numbers) >= self.order:
            current_state = tuple(recent_numbers[-self.order:])
            
            if current_state in self.transition_matrix:
                probs = [self.transition_matrix[current_state][n] for n in self.unique_numbers]
                return np.array(probs)
        
        # If state not seen or insufficient history, return uniform distribution
        return np.ones(len(self.unique_numbers)) / len(self.unique_numbers)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance using multiple metrics.
        
        Args:
            X: Test features
            y: True target values
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
        }
        
        # Calculate accuracy for classification-like evaluation
        correct = (predictions == y.values).sum()
        metrics['accuracy'] = correct / len(y)
        
        # Store metrics for later reference
        self.performance_metrics = {k: float(v) for k, v in metrics.items()}
        
        return self.performance_metrics

    def optimize_order(self, X: pd.DataFrame, y: pd.Series, max_order: int = 5) -> int:
        """
        Optimize the order of the Markov Chain based on performance metrics.
        
        Args:
            X: Not used in Markov Chain (kept for consistency)
            y: Training sequence
            max_order: Maximum order to try
            
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

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the transition probability matrix as a DataFrame.
        
        Returns:
            pd.DataFrame: Transition probability matrix
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before getting transition matrix")
        
        # Convert transition matrix to DataFrame
        matrix_dict = {}
        for state in self.transition_matrix:
            matrix_dict[str(state)] = {
                n: self.transition_matrix[state][n] for n in self.unique_numbers
            }
        
        return pd.DataFrame.from_dict(matrix_dict, orient='index')

    def get_most_probable_transitions(self, top_n: int = 5) -> List[Tuple[Tuple, int, float]]:
        """
        Get the most probable state transitions.
        
        Args:
            top_n: Number of top transitions to return
            
        Returns:
            List of (state, next_state, probability) tuples
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before getting transitions")
        
        transitions = []
        for state in self.transition_matrix:
            for next_state, prob in self.transition_matrix[state].items():
                transitions.append((state, next_state, prob))
        
        # Sort by probability (descending) and return top N
        return sorted(transitions, key=lambda x: x[2], reverse=True)[:top_n]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on transition probabilities.
        
        For Markov Chain, feature importance is calculated based on how predictive
        each state is, which is measured by the variance in transition probabilities.
        
        Returns:
            Dict[str, float]: Dictionary mapping state patterns to importance scores
        """
        if not self.transition_matrix:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_dict: Dict[str, float] = {}
        
        # Calculate importance based on a combination of:
        # 1. State frequency (how common is this state)
        # 2. Predictability (variance in transition probabilities)
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

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence based on transition probabilities.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Confidence scores for each prediction
        """
        if self.unique_numbers is None:
            raise ValueError("Model must be trained before estimating confidence")
        
        # Get the most recent state from X
        recent_numbers = self._get_recent_numbers(X)
        current_state = tuple(recent_numbers[-self.order:]) if len(recent_numbers) >= self.order else None
        
        # Calculate confidence based on the distribution of transition probabilities
        confidences = []
        
        for _ in range(len(X)):
            if current_state in self.transition_matrix:
                # Extract probabilities for each possible next state
                probabilities = [self.transition_matrix[current_state][n] for n in self.unique_numbers]
                
                # Higher max probability indicates higher confidence
                max_prob = max(probabilities)
                
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information
        """
        return {
            'name': self.name,
            'order': self.order,
            'smoothing': self.smoothing,
            'states_count': len(self.transition_matrix),
            'unique_numbers': self.unique_numbers,
            'total_transitions': self.total_transitions,
            'performance': self.performance_metrics
        }


class VariableOrderMarkovChain(BaseModel):
    """
    An enhanced Markov Chain that maintains multiple orders simultaneously
    and selects the best one for each prediction based on context.
    """
    
    def __init__(self, 
                 max_order: int = 5, 
                 min_order: int = 1,
                 smoothing: float = 0.1, 
                 **kwargs):
        """
        Initialize the Variable Order Markov Chain model.
        
        Args:
            max_order: Maximum Markov Chain order to consider
            min_order: Minimum Markov Chain order to consider
            smoothing: Laplace smoothing parameter
            **kwargs: Additional parameters for the base model
        """
        name = kwargs.pop('name', "VariableOrderMarkovChain")
        super().__init__(name=name, **kwargs)
        self.max_order = max_order
        self.min_order = min_order
        self.smoothing = smoothing
        self.models: Dict[int, MarkovChain] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.feature_importance_: Dict[str, float] = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'VariableOrderMarkovChain':
        """
        Fit multiple Markov Chain models of different orders.
        
        Args:
            X: Training features
            y: Training sequence
            
        Returns:
            self: The fitted model instance
        """
        # Train a model for each order
        for order in range(self.min_order, self.max_order + 1):
            model = MarkovChain(order=order, smoothing=self.smoothing)
            model.fit(X, y)
            self.models[order] = model
        
        # Aggregate feature importance from all models
        self._update_feature_importance()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the optimal model for each context.
        
        Args:
            X: Features containing recent history
            
        Returns:
            np.ndarray: Predicted values
        """
        if not self.models:
            raise ValueError("Models must be trained before making predictions")
        
        # Get predictions from all models
        all_predictions = {
            order: model.predict(X) 
            for order, model in self.models.items()
        }
        
        # Determine the best order based on context (we'll use confidence in predictions)
        # For each row in X, select the model with highest confidence (non-uniform distribution)
        confidences = {
            order: model.estimate_confidence(X)
            for order, model in self.models.items()
        }
        
        # Select predictions from the most confident model
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            # Find the order with highest confidence for this observation
            best_order = max(
                confidences.keys(),
                key=lambda order: confidences[order][i]
            )
            predictions[i] = all_predictions[best_order][i]
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions using ensemble of models.
        
        Args:
            X: Features containing recent history
            
        Returns:
            np.ndarray: Probability predictions
        """
        if not self.models:
            raise ValueError("Models must be trained before predicting probabilities")
        
        # Get probabilities from each model
        all_probs = [model.predict_proba(X) for model in self.models.values()]
        
        # Average probabilities from all models
        return np.mean(all_probs, axis=0)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True target values
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        predictions = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
        }
        
        # Calculate accuracy
        correct = (predictions == y.values).sum()
        metrics['accuracy'] = correct / len(y)
        
        # Store metrics
        self.performance_metrics = {k: float(v) for k, v in metrics.items()}
        
        return self.performance_metrics
    
    def _update_feature_importance(self) -> None:
        """Update feature importance by combining from all models."""
        all_importance = {}
        
        # Collect feature importance from all models
        for order, model in self.models.items():
            importance = model.get_feature_importance()
            
            # Weight importance by order (higher orders get higher weight)
            weight = order / sum(range(self.min_order, self.max_order + 1))
            
            for feature, value in importance.items():
                if feature not in all_importance:
                    all_importance[feature] = 0
                all_importance[feature] += value * weight
        
        # Normalize
        if all_importance:
            max_value = max(all_importance.values())
            if max_value > 0:
                all_importance = {
                    k: float(v / max_value)
                    for k, v in all_importance.items()
                }
        
        self.feature_importance_ = all_importance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not self.feature_importance_:
            self._update_feature_importance()
        
        return self.feature_importance_
    
    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence by combining confidence from all models.
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
            
        Returns:
            np.ndarray: Confidence estimates for each prediction
        """
        if not self.models:
            raise ValueError("Models must be trained before estimating confidence")
        
        # Get confidence from each model
        all_confidences = np.array([
            model.estimate_confidence(X) for model in self.models.values()
        ])
        
        # Use the maximum confidence among all models
        return np.max(all_confidences, axis=0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'name': self.name,
            'min_order': self.min_order,
            'max_order': self.max_order,
            'smoothing': self.smoothing,
            'models_count': len(self.models),
            'performance': self.performance_metrics
        }