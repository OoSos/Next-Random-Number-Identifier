import sys
import os
import pytest  # New import

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

@pytest.fixture(scope='session')
def setup_environment():
    # Setup code for the test environment
    yield
    # Teardown code for the test environment
