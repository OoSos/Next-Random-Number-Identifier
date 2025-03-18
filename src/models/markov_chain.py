from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from collections import defaultdict
from .base_model import BaseModel
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MarkovChain(BaseModel):
    """Markov Chain model for sequence prediction."""
    
    def __init__(self, order: int = 1, **kwargs):
        """Initialize MarkovChain model.
        
        Args:
            order: The order of the Markov Chain (number of past states to consider)
            **kwargs: Additional parameters for the base model
        """
        name = kwargs.pop('name', 'MarkovChain')
        super().__init__(name=name, **kwargs)
        self.order = order
        self.state_transitions = {}
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
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model using feature matrix X and target y.
        
        Args:
            X: Feature matrix
            y: Target values
        """
        # Initialize transition counts
        transition_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Convert features to strings to handle any feature type
        sequences = []
        for i in range(len(X) - self.order):
            state_features = []
            for j in range(self.order):
                state_features.extend(X.iloc[i + j])
            current_state = self._encode_state(pd.Series(state_features))
            next_value = int(y.iloc[i + self.order])
            sequences.append((current_state, next_value))
            
        # Count transitions
        for state, next_value in sequences:
            transition_counts[state][next_value] += 1
            
        # Convert counts to probabilities
        self.state_transitions = {}
        for state, counts in transition_counts.items():
            total = sum(counts.values())
            self.state_transitions[state] = {
                value: count / total 
                for value, count in counts.items()
            }
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        predictions = []
        
        for i in range(len(X)):
            state_features = []
            for j in range(min(self.order, i + 1)):
                idx = i - j if i >= self.order else j
                state_features.extend(X.iloc[idx])
            
            current_state = self._encode_state(pd.Series(state_features))
            
            # If state exists in transitions, use most likely next value
            # Otherwise use default value
            if current_state in self.state_transitions:
                transitions = self.state_transitions[current_state]
                prediction = max(transitions.items(), key=lambda x: x[1])[0]
            else:
                # Use most common transition across all states as default
                all_transitions = defaultdict(float)
                for state_trans in self.state_transitions.values():
                    for value, prob in state_trans.items():
                        all_transitions[value] += prob
                prediction = max(all_transitions.items(), key=lambda x: x[1])[0]
            
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
            state_features = []
            for j in range(min(self.order, i + 1)):
                idx = i - j if i >= self.order else j
                state_features.extend(X.iloc[idx])
            
            current_state = self._encode_state(pd.Series(state_features))
            
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