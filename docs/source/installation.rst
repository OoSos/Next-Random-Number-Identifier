Installation
============

Requirements
-----------

Core Dependencies (Always Required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **pandas >= 1.3.5** - Data manipulation and analysis
- **numpy >= 1.21.6** - Numerical computing
- **scikit-learn >= 1.0.2** - Machine learning algorithms
- **matplotlib >= 3.4.3** - Basic plotting and visualization
- **statsmodels >= 0.13.2** - Statistical analysis and modeling
- **seaborn >= 0.11.2** - Enhanced statistical visualization

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

- **XGBoost** - Advanced gradient boosting models and enhanced ensemble methods
- **PyTorch** - Neural network models (experimental features)
- **Development Tools** - pytest, black, isort, flake8, mypy for contributing
- **Documentation Tools** - Sphinx and related tools for building documentation

Installation Steps
----------------

Basic Installation (Core Features Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository::

    git clone https://github.com/OoSos/next-random-number-identifier.git
    cd next-random-number-identifier

2. Install core dependencies only::

    pip install -e .

Full Installation (All Features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install with all optional dependencies::

    pip install -e .[full]

Or install specific feature sets::

    pip install -e .[xgboost]  # Core + XGBoost
    pip install -e .[dev]      # Core + Development tools
    pip install -e .[doc]      # Core + Documentation tools

Production Installation
~~~~~~~~~~~~~~~~~~~~~~

For production environments (core + XGBoost, no dev tools)::

    pip install -e .[xgboost]

Verification
-----------

After installation, verify that your desired features are available::

    python -c "
    # Test core functionality
    from src.models.random_forest import RandomForestModel
    from src.models.markov_chain import MarkovChain
    print('✅ Core features available')
    
    # Test XGBoost (if installed)
    try:
        from src.models.xgboost_model import XGBoostModel
        print('✅ XGBoost features available')
    except ImportError:
        print('❌ XGBoost not available')
    "