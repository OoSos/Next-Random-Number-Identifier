# Next Random Number Identifier

[![GitHub Actions Status](https://github.com/OoSos/next-random-number-identifier/workflows/Enhanced%20Next%20Random Number Identifier%20CI/CD/badge.svg)](https://github.com/OoSos/next-random-number-identifier/actions)
[![codecov](https://codecov.io/gh/OoSos/next-random-number-identifier/branch/main/graph/badge.svg)](https://codecov.io/gh/OoSos/next-random-number-identifier)
[![Documentation Status](https://github.com/OoSos/next-random-number-identifier/workflows/docs/badge.svg)](https://OoSos.github.io/next-random-number-identifier/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning-based system for analyzing and forecasting random number sequences using multiple approaches:
- Random Forest Regression
- XGBoost Classification
- Markov Chain Analysis
- Frequency Analysis
- Enhanced Ensemble Integration

## Architecture

The NRNI system uses a layered architecture with data processing, feature engineering, 
modeling, and ensemble components. The system is designed with clear separation of concerns
and modular components for maximum flexibility and maintainability.

For detailed documentation:
- [Architecture Documentation](docs/Next%20Random%20Number%20Identifier-architecture-documentation.md)
- [Component Interaction Diagram](docs/diagrams/NRNI%20Component%20Interaction-diagrams.png)
- [Data Flow Diagram](docs/diagrams/NRNI%20Data-flow-diagram.png)
- [Prediction Sequence Diagram](docs/diagrams/NRNI%20Prediction%20sequence-diagram.png)

### Interactive Diagrams

All architecture diagrams are available in Mermaid format for interactive viewing:
- GitHub natively renders Mermaid diagrams in markdown files
- Source `.mermaid` files are available in the `docs/diagrams/` directory
- For interactive editing, use [Mermaid Live Editor](https://mermaid.live/)

## Project Structure
- `src/`: Source code files
  - `models/`: ML and statistical models
    - `base_model.py`: Abstract base class for models
    - `random_forest.py`: Random Forest implementation
    - `xgboost_model.py`: XGBoost implementation
    - `markov_chain.py`: Markov Chain implementation
    - `ensemble.py`: Ensemble model integration
  - `features/`: Feature engineering
    - `feature_engineering.py`: Feature creation
    - `feature_selection.py`: Feature importance and selection
  - `utils/`: Utility functions
    - `data_loader.py`: Data loading and preprocessing
    - `evaluation.py`: Model evaluation metrics
    - `monitoring.py`: Model drift detection
  - `visualization/`: Visualization tools
    - `plots.py`: Plotting functions
- `tests/`: Test files
- `data/`: Data files and datasets
  - `historical_random_numbers.csv`: Historical random numbers dataset
- `docs/`: Documentation
- `models/`: Saved model files

## Requirements & Installation

### Core Dependencies (Always Required)
The following dependencies are required for basic NRNI functionality:
- **pandas >= 1.3.5** - Data manipulation and analysis
- **numpy >= 1.21.6** - Numerical computing
- **scikit-learn >= 1.0.2** - Machine learning algorithms (Random Forest, etc.)
- **matplotlib >= 3.4.3** - Basic plotting and visualization
- **statsmodels >= 0.13.2** - Statistical analysis and modeling
- **seaborn >= 0.11.2** - Enhanced statistical visualization

### Optional Dependencies

#### Advanced Machine Learning Features
- **XGBoost**: `pip install next_random_number_identifier[xgboost]`
  - Required for gradient boosting models and advanced ensemble methods
  - Provides XGBoostModel and enhanced ensemble performance
  - Install: `pip install xgboost==1.5.2`

#### Deep Learning Features (Experimental)
- **PyTorch**: `pip install next_random_number_identifier[torch]`
  - Required for neural network models (experimental features)
  - Currently used for advanced pattern recognition research
  - Install: `pip install torch==1.9.1`

#### Development Tools
- **Development Suite**: `pip install next_random_number_identifier[dev]`
  - Includes pytest, black, isort, flake8, mypy, and other development tools
  - Required for contributing to the project
  - Enforced through CI/CD pipeline

#### Documentation Tools
- **Documentation**: `pip install next_random_number_identifier[doc]`
  - Sphinx and related tools for building documentation
  - Required for generating and updating project documentation

### Installation Options

#### Basic Installation (Core Features Only)
```bash
# Clone the repository
git clone https://github.com/OoSos/next-random-number-identifier.git
cd next-random-number-identifier

# Install core dependencies only
pip install -e .
```

#### Full Installation (All Features)
```bash
# Install with all optional dependencies
pip install -e .[full]

# Or install specific feature sets
pip install -e .[xgboost]  # Core + XGBoost
pip install -e .[dev]      # Core + Development tools
pip install -e .[doc]      # Core + Documentation tools
```

#### Production Installation
```bash
# For production environments (core + XGBoost, no dev tools)
pip install -e .[xgboost]
```

### Feature Availability by Installation

| Feature | Core | +XGBoost | +Torch | +Dev |
|---------|------|----------|---------|------|
| Random Forest Model | ✅ | ✅ | ✅ | ✅ |
| Markov Chain Analysis | ✅ | ✅ | ✅ | ✅ |
| Frequency Analysis | ✅ | ✅ | ✅ | ✅ |
| Basic Ensemble | ✅ | ✅ | ✅ | ✅ |
| XGBoost Model | ❌ | ✅ | ✅ | ✅ |
| Enhanced Ensemble | ❌ | ✅ | ✅ | ✅ |
| Neural Networks | ❌ | ❌ | ✅ | ✅ |
| Code Quality Tools | ❌ | ❌ | ❌ | ✅ |

### Verification of Installation

After installation, verify that your desired features are available:

```python
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
```

## Usage

### Command Line Interface

The package provides a command-line interface for easy interaction:

```bash
# Train the ensemble model
python -m src.cli --mode train --model ensemble

# Make predictions
python -m src.cli --mode predict

# Evaluate model performance
python -m src.cli --mode evaluate --model xgb

# Monitor model drift
python -m src.cli --mode monitor --model ensemble
```

### Python API

```python
from src.utils.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.ensemble import EnhancedEnsemble

# Load and prepare data
data_loader = DataLoader("data")
df = data_loader.load_csv("historical_random_numbers.csv")
df = data_loader.preprocess_data(df)

# Create features
feature_engineer = FeatureEngineer()
df_features = feature_engineer.transform(df)

# Prepare data
X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
y = df_features['Number']

# Train ensemble model
ensemble = EnhancedEnsemble()
ensemble.fit(X, y)

# Make predictions
predictions = ensemble.predict(X.head(5))
print(f"Next predicted numbers: {predictions}")
```

## Development

This project follows the best practices for Python development including:

- Type checking with MyPy
- Code formatting with Black
- Import ordering with isort
- Linting with Flake8
- Testing with pytest

All of these checks are enforced through GitHub Actions CI/CD.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run the tests and linting checks (`pytest` and `flake8`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* AIQube Centaur Systems Team for project development
* Anthropic Claude for implementation guidance

## Troubleshooting Installation

#### Common Installation Issues

**Issue: XGBoost installation fails**
```bash
# Solution 1: Update pip and try again
pip install --upgrade pip
pip install xgboost==1.5.2

# Solution 2: Use conda (if available)
conda install -c conda-forge xgboost=1.5.2

# Solution 3: Install without XGBoost for now
pip install -e .  # Core installation only
```

**Issue: PyTorch installation is slow or fails**
```bash
# Solution: Install CPU-only version for faster installation
pip install torch==1.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Or skip PyTorch for now (it's experimental)
pip install -e .[xgboost]  # Skip torch features
```

**Issue: "Module not found" errors during import**
```bash
# Solution: Ensure you're in the project directory and using editable install
cd next-random-number-identifier
pip install -e .

# Verify your Python path includes the project
python -c "import sys; print(sys.path)"
```

**Issue: Version conflicts with existing packages**
```bash
# Solution: Use a virtual environment
python -m venv nrni_env
# On Windows:
nrni_env\Scripts\activate
# On macOS/Linux:
source nrni_env/bin/activate

# Then install in the clean environment
pip install -e .[full]
```

#### Performance Considerations

- **Memory Usage**: Core installation requires ~500MB RAM, full installation ~2GB
- **CPU Cores**: Feature engineering benefits from multiple cores (4+ recommended)
- **Installation Time**: 
  - Core: ~2-3 minutes
  - With XGBoost: ~5-7 minutes  
  - With PyTorch: ~10-15 minutes

#### Environment Verification

Run this script to verify your environment:

```python
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
    
    print("\n=== Required Packages ===")
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: NOT INSTALLED")
    
    print("\n=== Optional Packages ===")
    for package in optional_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"⚠️  {package}: NOT INSTALLED (optional)")

if __name__ == "__main__":
    check_environment()
```