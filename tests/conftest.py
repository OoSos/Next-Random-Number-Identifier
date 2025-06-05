import sys
import os
from pathlib import Path

# Add project root and src directory to path
project_root = Path(__file__).resolve().parent.parent # Get absolute path of project root
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest

@pytest.fixture(scope='session')
def setup_environment():
    # Setup code for the test environment
    # Example: print(f"sys.path in conftest: {sys.path}")
    yield
    # Teardown code for the test environment
