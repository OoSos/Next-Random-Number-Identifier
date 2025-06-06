.. // filepath: c:\\Users\\Owner\\GitHubProjects\\Next-Random-Number-Identifier\\docs\\source\\installation.rst
.. highlight:: bash

Installation
============

Requirements
------------

Core Dependencies (Always Required)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following dependencies are required for basic NRNI functionality:

- **pandas >= 1.3.5** - Data manipulation and analysis
- **numpy >= 1.21.6** - Numerical computing
- **scikit-learn >= 1.0.2** - Machine learning algorithms (Random Forest, etc.)
- **matplotlib >= 3.4.3** - Basic plotting and visualization
- **statsmodels >= 0.13.2** - Statistical analysis and modeling
- **seaborn >= 0.11.2** - Enhanced statistical visualization

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Advanced Machine Learning Features
**********************************
- **XGBoost**: ``pip install next_random_number_identifier[xgboost]``
  - Required for gradient boosting models and advanced ensemble methods.
  - Provides XGBoostModel and enhanced ensemble performance.
  - Install: ``pip install xgboost==1.5.2``

Deep Learning Features (Experimental)
*************************************
- **PyTorch**: ``pip install next_random_number_identifier[torch]``
  - Required for neural network models (experimental features).
  - Currently used for advanced pattern recognition research.
  - Install: ``pip install torch==1.9.1``

Development Tools
*****************
- **Development Suite**: ``pip install next_random_number_identifier[dev]``
  - Includes pytest, black, isort, flake8, mypy, and other development tools.
  - Required for contributing to the project.
  - Enforced through CI/CD pipeline.

Documentation Tools
*******************
- **Documentation**: ``pip install next_random_number_identifier[doc]``
  - Sphinx and related tools for building documentation.
  - Required for generating and updating project documentation.

Installation Options
--------------------

Basic Installation (Core Features Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Clone the repository::

    git clone https://github.com/OoSos/next-random-number-identifier.git
    cd next-random-number-identifier

2. Install core dependencies only::

    pip install -e .

Full Installation (All Features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install with all optional dependencies::

    pip install -e .[full]

Or install specific feature sets::

    pip install -e .[xgboost]  # Core + XGBoost
    pip install -e .[dev]      # Core + Development tools
    pip install -e .[doc]      # Core + Documentation tools

Production Installation
~~~~~~~~~~~~~~~~~~~~~~~
For production environments (core + XGBoost, no dev tools)::

    pip install -e .[xgboost]

Feature Availability by Installation
------------------------------------

.. list-table::
   :widths: 20 15 15 15 15
   :header-rows: 1

   * - Feature
     - Core
     - +XGBoost
     - +Torch
     - +Dev
   * - Random Forest Model
     - ✅
     - ✅
     - ✅
     - ✅
   * - Markov Chain Analysis
     - ✅
     - ✅
     - ✅
     - ✅
   * - Frequency Analysis
     - ✅
     - ✅
     - ✅
     - ✅
   * - Basic Ensemble
     - ✅
     - ✅
     - ✅
     - ✅
   * - XGBoost Model
     - ❌
     - ✅
     - ✅
     - ✅
   * - Enhanced Ensemble
     - ❌
     - ✅
     - ✅
     - ✅
   * - Neural Networks
     - ❌
     - ❌
     - ✅
     - ✅
   * - Code Quality Tools
     - ❌
     - ❌
     - ❌
     - ✅

Verification of Installation
----------------------------
After installation, verify that your desired features are available by running the following Python script:

.. code-block:: python

    # Test core functionality
    from src.models.random_forest import RandomForestModel
    from src.models.markov_chain import MarkovChain

    # Test XGBoost (if installed)
    try:
        from src.models.xgboost_model import XGBoostModel
        from src.models.ensemble import EnhancedEnsemble
        print("✅ XGBoost features available")
    except ImportError:
        print("❌ XGBoost not available - install with: pip install -e .[xgboost]")

    # Test PyTorch (if installed)
    try:
        import torch
        print("✅ PyTorch features available")
    except ImportError:
        print("❌ PyTorch not available - install with: pip install -e .[torch]")

Troubleshooting Installation
----------------------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: XGBoost installation fails**
  - **Solution 1**: Update pip and try again::

      pip install --upgrade pip
      pip install xgboost==1.5.2

  - **Solution 2**: Use conda (if available)::

      conda install -c conda-forge xgboost=1.5.2

  - **Solution 3**: Install without XGBoost for now::

      pip install -e .  # Core installation only

**Issue: PyTorch installation is slow or fails**
  - **Solution**: Install CPU-only version for faster installation::

      pip install torch==1.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

  - Or skip PyTorch for now (it's experimental)::

      pip install -e .[xgboost]  # Skip torch features

**Issue: "Module not found" errors during import**
  - **Solution**: Ensure you're in the project directory and using editable install::

      cd next-random-number-identifier
      pip install -e .

  - Verify your Python path includes the project::

      python -c "import sys; print(sys.path)"

**Issue: Version conflicts with existing packages**
  - **Solution**: Use a virtual environment::

      python -m venv nrni_env
      # On Windows:
      nrni_env\\Scripts\\activate
      # On macOS/Linux:
      source nrni_env/bin/activate

  - Then install in the clean environment::

      pip install -e .[full]

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Memory Usage**: Core installation requires ~500MB RAM, full installation ~2GB
- **CPU Cores**: Feature engineering benefits from multiple cores (4+ recommended)
- **Installation Time**:
  - Core: ~2-3 minutes
  - With XGBoost: ~5-7 minutes
  - With PyTorch: ~10-15 minutes

Environment Verification
~~~~~~~~~~~~~~~~~~~~~~~~
Run this script to verify your environment:

.. code-block:: python

    # environment_check.py
    import sys
    import pkg_resources

    def check_environment():
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")

        required_packages = [
            'numpy', 'pandas', 'scikit-learn',
            'matplotlib', 'statsmodels', 'seaborn'
        ]

        optional_packages = ['xgboost', 'torch']

        print("\\n=== Required Packages ===")
        for package in required_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"✅ {package}: {version}")
            except pkg_resources.DistributionNotFound:
                print(f"❌ {package}: NOT INSTALLED")

        print("\\n=== Optional Packages ===")
        for package in optional_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"✅ {package}: {version}")
            except pkg_resources.DistributionNotFound:
                print(f"⚠️  {package}: NOT INSTALLED (optional)")

    if __name__ == "__main__":
        check_environment()