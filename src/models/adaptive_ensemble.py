from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel
from .random_forest import RandomForestModel
from .markov_chain import MarkovChain, VariableOrderMarkovChain
try:
    from .xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Add new AdaptiveEnsemble implementation
class AdaptiveEnsemble(BaseModel):
    """Adaptive ensemble with dynamic model selection based on context."""
    
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        meta_feature_strategy: str = 'predictions',
        cv_folds: int = 5,
        **kwargs
    ):
        # ... paste new AdaptiveEnsemble implementation here ...
