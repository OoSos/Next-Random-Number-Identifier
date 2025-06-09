# Standard library imports
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Any

# Third-party imports
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Local application imports
from src.models.base_model import BaseModel

class MarkovChain(BaseModel):
    """Markov Chain model for sequence prediction."""
    
    def __init__(self, order: int = 2, smoothing: float = 0.1, **kwargs):
        """Initialize MarkovChain model.
        
        Args:
            order: The order of the Markov Chain (number of past states to consider)
            smoothing: Laplace smoothing parameter
            **kwargs: Additional parameters for the base model
        """
        # Parameter validation
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order <= 0:
            raise ValueError(f"order must be positive, got {order}")
        if not isinstance(smoothing, (int, float)):
            raise TypeError(f"smoothing must be a number, got {type(smoothing).__name__}")
        if smoothing < 0 or smoothing > 1:
            raise ValueError(f"smoothing must be between 0 and 1, got {smoothing}")
            
        name = kwargs.pop('name', 'MarkovChain')
        super().__init__(name=name, **kwargs)
        self.order = order
        self.smoothing = smoothing
        self.state_transitions = {}
        self.transition_matrix = {}
        self.unique_numbers = set()
        self.feature_importance_ = {}
        
    def _encode_state(self, features: pd.Series) -> str:
        """Encode feature values into a state string.
        
        Args:
            features: Series of feature values
            
        Returns:
            Encoded state string
        """
        # Convert all features to strings and join with separator
        return "|".join(str(v) for v in features)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, disable_smoothing: bool = False) -> 'MarkovChain':
        """Fit the model using feature matrix X and target y.
        
        Args:
            X: Feature matrix
            y: Target values
            disable_smoothing: Whether to disable Laplace smoothing
            
        Returns:
            self: The fitted model instance
        """
        # Store unique numbers for transition matrix
        self.unique_numbers = set(y.unique())
        
        # Initialize transition counts  
        transition_counts: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Convert features to strings to handle any feature type
        sequences = self._create_sequence(y)
        
        # Count transitions
        for state, next_value in sequences:
            transition_counts[state][next_value] += 1
            
        # Convert counts to probabilities with optional smoothing
        self.state_transitions = {}
        self.transition_matrix = {}
        
        for state, counts in transition_counts.items():
            total = sum(counts.values())
            if not disable_smoothing:
                # Apply Laplace smoothing
                smoothed_total = total + len(self.unique_numbers) * self.smoothing
                self.state_transitions[state] = {
                    value: (counts.get(value, 0) + self.smoothing) / smoothed_total 
                    for value in self.unique_numbers
                }
            else:
                self.state_transitions[state] = {
                    value: count / total 
                    for value, count in counts.items()
                }
            
        # Store transition matrix
        self.transition_matrix = self.state_transitions
        
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        predictions = []
        
        for i in range(len(X)):
            # For Markov chain, we need the last 'order' values from y (not X)
            # Since we don't have y history, we'll use a simple approach
            if i < self.order:
                # Not enough history, use most common transition
                all_transitions = defaultdict(float)
                for state_trans in self.state_transitions.values():
                    for value, prob in state_trans.items():
                        all_transitions[value] += prob
                if all_transitions:
                    prediction = max(all_transitions.items(), key=lambda x: x[1])[0]
                else:
                    prediction = list(self.unique_numbers)[0] if self.unique_numbers else 0
            else:
                # Create state from feature row (simplified approach)
                current_state = tuple(X.iloc[i].values[:self.order])
                
                # If state exists in transitions, use most likely next value
                if current_state in self.state_transitions:
                    transitions = self.state_transitions[current_state]
                    prediction = max(transitions.items(), key=lambda x: x[1])[0]
                else:
                    # Use most common transition across all states as default
                    all_transitions = defaultdict(float)
                    for state_trans in self.state_transitions.values():
                        for value, prob in state_trans.items():
                            all_transitions[value] += prob
                    if all_transitions:
                        prediction = max(all_transitions.items(), key=lambda x: x[1])[0]
                    else:
                        prediction = list(self.unique_numbers)[0] if self.unique_numbers else 0
            
            predictions.append(prediction)
            
        return np.array(predictions)
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'order': self.order,
            'n_states': len(self.state_transitions)
        }
        
    def set_params(self, **params: Any) -> None:
        """Set model parameters."""
        if 'order' in params:
            self.order = params['order']

    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence for each sample.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of confidence scores between 0 and 1
        """
        confidences = []
        for i in range(len(X)):
            if i < self.order:
                confidences.append(0.5)  # Default confidence when not enough history
            else:
                current_state = tuple(X.iloc[i].values[:self.order])
                
                if current_state in self.state_transitions:
                    # Use highest transition probability as confidence
                    max_prob = max(self.state_transitions[current_state].values())
                    confidences.append(max_prob)
                else:
                    # Use prior probabilities if state not seen
                    confidences.append(0.5)  # Default confidence when state unknown
                
        return np.array(confidences)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # For Markov Chain, use state transition probabilities as proxy for importance
        importance_scores = {}
        
        if not self.state_transitions:
            return {}
            
        # Calculate average probability for each unique value in transitions
        value_probs = defaultdict(list)
        for state_trans in self.state_transitions.values():
            for value, prob in state_trans.items():
                value_probs[value].append(prob)
                
        # Average probabilities represent how important each value is
        for value, probs in value_probs.items():
            importance_scores[f'value_{value}'] = float(np.mean(probs))
            
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        return {k: v/max_score for k, v in importance_scores.items()}

    def _create_sequence(self, y: pd.Series) -> List[Tuple[Tuple, int]]:
        """Create sequence pairs for training.
        
        Args:
            y: Target sequence
            
        Returns:
            List of (state, next_value) tuples
        """
        sequences = []
        for i in range(len(y) - self.order):
            # Create state from previous 'order' values
            state = tuple(y.iloc[i:i + self.order])
            next_value = int(y.iloc[i + self.order])
            sequences.append((state, next_value))
        return sequences

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability distributions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probability distributions
        """
        if not self.state_transitions:
            raise ValueError("Model must be fitted before predicting probabilities")
            
        probabilities = []
        n_classes = len(self.unique_numbers)
        sorted_values = sorted(self.unique_numbers)
        
        for i in range(len(X)):
            # Create state from last 'order' predictions or default
            if i < self.order:
                # Not enough history, use uniform distribution
                prob_dist = np.ones(n_classes) / n_classes
            else:
                current_state = tuple(X.iloc[i].values[:self.order])
                
                if current_state in self.state_transitions:
                    # Create probability distribution
                    prob_dist = np.zeros(n_classes)
                    for idx, value in enumerate(sorted_values):
                        prob_dist[idx] = self.state_transitions[current_state].get(value, 0)
                else:
                    # Unknown state, use uniform distribution
                    prob_dist = np.ones(n_classes) / n_classes
            
            probabilities.append(prob_dist)
            
        return np.array(probabilities)

    def get_transition_matrix(self) -> pd.DataFrame:
        """Get transition matrix as DataFrame.
        
        Returns:
            Transition matrix DataFrame
        """
        if not self.transition_matrix:
            raise ValueError("Model must be fitted before getting transition matrix")
            
        # Create DataFrame from transition matrix
        states = list(self.transition_matrix.keys())
        values = sorted(self.unique_numbers)
        
        matrix_data = []
        for state in states:
            row = []
            for value in values:
                prob = self.transition_matrix[state].get(value, 0.0)
                row.append(prob)
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data, index=states, columns=values)

    def get_most_probable_transitions(self, top_n: int = 5) -> List[Tuple[str, int, float]]:
        """Get most probable transitions.
        
        Args:
            top_n: Number of top transitions to return
            
        Returns:
            List of (state, next_value, probability) tuples
        """
        if not self.state_transitions:
            raise ValueError("Model must be fitted before getting transitions")
            
        transitions = []
        for state, trans_dict in self.state_transitions.items():
            for value, prob in trans_dict.items():
                transitions.append((str(state), value, prob))
        
        # Sort by probability and return top_n
        transitions.sort(key=lambda x: x[2], reverse=True)
        return transitions[:top_n]

    def generate_report(self) -> Dict[str, Any]:
        """Generate model report.
        
        Returns:
            Model report dictionary
        """
        if not self.state_transitions:
            raise ValueError("Model must be fitted before generating report")
            
        return {
            'model_name': self.name,
            'order': self.order,
            'smoothing': self.smoothing,
            'n_states': len(self.state_transitions),
            'n_unique_values': len(self.unique_numbers),
            'unique_values': sorted(self.unique_numbers),
            'most_probable_transitions': self.get_most_probable_transitions(3)
        }

    def runs_test(self, sequence: pd.Series) -> Dict[str, Any]:
        """Perform runs test for randomness.
        
        Args:
            sequence: Sequence to test
            
        Returns:
            Test results
        """
        # Simple runs test implementation
        runs = 1
        for i in range(1, len(sequence)):
            if sequence.iloc[i] != sequence.iloc[i-1]:
                runs += 1
        
        n = len(sequence)
        expected_runs = (2 * n - 1) / 3
        variance = (16 * n - 29) / 90
        
        z_score = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0
        
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_score': z_score,
            'p_value': 2 * (1 - norm.cdf(abs(z_score)))
        }

    def serial_test(self, sequence: pd.Series) -> Dict[str, Any]:
        """Perform serial test for randomness.
        
        Args:
            sequence: Sequence to test
            
        Returns:
            Test results
        """
        # Simple serial test - check autocorrelation
        n = len(sequence)
        if n < 2:
            return {'correlation': 0, 'p_value': 1.0}
        
        # Calculate lag-1 autocorrelation
        mean_val = sequence.mean()
        numerator = sum((sequence.iloc[i] - mean_val) * (sequence.iloc[i-1] - mean_val) 
                       for i in range(1, n))
        denominator = sum((sequence.iloc[i] - mean_val) ** 2 for i in range(n))
        
        correlation = numerator / denominator if denominator > 0 else 0
        
        # Approximate p-value (simplified)
        z_score = correlation * np.sqrt(n - 1)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return {
            'correlation': correlation,
            'z_score': z_score,
            'p_value': p_value
        }

    def visualize_transition_matrix(self) -> None:
        """Visualize transition matrix.
        """
        if not self.transition_matrix:
            raise ValueError("Model must be fitted before visualizing")
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            matrix_df = self.get_transition_matrix()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix_df, annot=True, cmap='Blues', fmt='.3f')
            plt.title(f'Markov Chain Transition Matrix (Order {self.order})')
            plt.xlabel('Next Value')
            plt.ylabel('Current State')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib/Seaborn not available for visualization")


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
