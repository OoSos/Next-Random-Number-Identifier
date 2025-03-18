"""
End-to-end integration tests.
This file combines tests from:
- test_integration.py (original)
- test_pipeline.py (if relevant tests exist)
"""

import pytest
import numpy as np
import torch
from src.data.data_loader import DataLoader
from src.models.lstm_model import LSTMModel
from src.pipeline.training_pipeline import TrainingPipeline

# ... existing code ...

# Tests migrated from test_pipeline.py (if any)
def test_pipeline_execution():
    # ... existing code ...

# ... existing code ...
