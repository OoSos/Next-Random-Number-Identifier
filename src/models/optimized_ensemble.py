from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

from .base_model import BaseModel
from .random_forest import RandomForestModel
from .markov_chain import MarkovChain
try:
    from .xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class OptimizedWeightEnsemble(BaseModel):
    """Ensemble model with weights optimized to minimize prediction error."""
    
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        validation_fraction: float = 0.2,
        min_weight: float = 0.01,
        loss_function: str = 'mse',
        **kwargs
    ):
        # ... paste OptimizedWeightEnsemble implementation here ...
