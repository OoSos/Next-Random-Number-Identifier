import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # New import

@pytest.fixture(scope='session')
def setup_environment():
    # Setup code for the test environment
    yield
    # Teardown code for the test environment
