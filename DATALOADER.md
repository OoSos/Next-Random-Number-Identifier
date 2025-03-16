# EnhancedDataLoader Documentation

## Overview

The `EnhancedDataLoader` is a robust data loading utility designed to handle various data formats, perform comprehensive validation, and preprocess data for analysis. It includes advanced error handling, data profiling, and validation capabilities to ensure data integrity and quality.

## Capabilities

- Load data from CSV, Excel, and JSON files
- Standardize column names for consistency
- Preprocess data with robust error handling
- Validate data schema, structure, and content integrity
- Generate comprehensive data profiles for exploratory analysis
- Handle missing values and data anomalies

## Usage Examples

### Initialization

```python
from src.utils.enhanced_data_loader import EnhancedDataLoader

# Initialize the EnhancedDataLoader
loader = EnhancedDataLoader("data")
```

### Loading and Preprocessing Data

```python
# Load and preprocess data
filename = "historical_random_numbers.csv"
df = loader.load_and_preprocess(filename)

print(f"Loaded data with shape: {df.shape}")
```

### Data Validation

```python
# Validate the data
validation_results = loader.validate_data(df)

print(f"Data is valid: {validation_results['valid']}")
```

### Data Profiling

```python
# Generate data profile
profile = loader.get_data_profile(df)

print(f"Data quality summary:")
print(f"  - Rows: {profile['basic_info']['rows']}")
print(f"  - Missing values: {profile['basic_info']['missing_cells']}")
print(f"  - Missing percentage: {profile['basic_info']['missing_percentage']:.2f}%")
```

## Validation and Profiling Features

### Data Validation

The `EnhancedDataLoader` provides comprehensive validation of data schema, structure, and content integrity. It checks for missing columns, data types, missing values, and more.

### Data Profiling

The data profiling feature generates a detailed profile of the data, including basic information, column-specific statistics, and distribution analysis.

## Error Handling Strategies

The `EnhancedDataLoader` includes advanced error handling strategies to ensure robust data loading and preprocessing. It handles various file formats, missing values, and data anomalies gracefully.

## Migration from Old DataLoader

To migrate from the old `DataLoader` to the `EnhancedDataLoader`, follow these steps:

1. Replace instances of `DataLoader` with `EnhancedDataLoader` in your code.
2. Update the initialization to include the data directory and optional configuration.
3. Use the new methods for loading, preprocessing, validating, and profiling data.

Example migration:

```python
# Old DataLoader
# from src.utils.data_loader import DataLoader
# loader = DataLoader("data")

# New EnhancedDataLoader
from src.utils.enhanced_data_loader import EnhancedDataLoader
loader = EnhancedDataLoader("data")
```

For more detailed migration guidance, refer to the project's documentation.
