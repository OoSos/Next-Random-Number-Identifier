# Next Random Number Identifier

A machine learning-based system for analyzing and forecasting random number sequences using multiple approaches:
- Random Forest Regression
- XGBoost Classification
- Markov Chain Analysis
- Frequency Analysis
- Enhanced Ensemble Integration

## Project Structure
- `src/`: Source code files
  - `random_number_forecast.py`: Main forecasting model (v1.0)
- `data/`: Data files and datasets
- `models/`: Saved model files
- `tests/`: Test files

## Requirements
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

## Setup
[Setup instructions will be added]

## Usage
[Usage instructions will be added]

## Dataset
The project uses a historical dataset of random numbers (`data/historical_random_numbers.csv`) which includes:
- Historical random number selections
- Dates of selections
- [Add any other relevant details about your dataset]

## File Structure
- `src/`
  - `random_number_forecast.py`: Main forecasting model (v1.0)
- `data/`
  - `historical_random_numbers.csv`: Historical random numbers dataset
- `models/`
- `tests/`

# Next Random Number Identifier

[![GitHub Actions Status](https://github.com/OoSos/next-random-number-identifier/workflows/Enhanced%20Next%20Random%20Number%20Identifier%20CI/CD/badge.svg)](https://github.com/OoSos/next-random-number-identifier/actions)
[![codecov](https://codecov.io/gh/OoSos/next-random-number-identifier/branch/main/graph/badge.svg)](https://codecov.io/gh/OoSos/next-random-number-identifier)
[![Documentation Status](https://github.com/OoSos/next-random-number-identifier/workflows/docs/badge.svg)](https://OoSos.github.io/next-random-number-identifier/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning-based system for analyzing and forecasting random number sequences using multiple approaches:
- Random Forest Regression
- XGBoost Classification
- Markov Chain Analysis
- Frequency Analysis
- Enhanced Ensemble Integration

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
- `docs/`: Documentation

## Requirements
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/OoSos/next-random-number-identifier.git
cd next-random-number-identifier

# Install the package and dependencies
pip install -e .
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