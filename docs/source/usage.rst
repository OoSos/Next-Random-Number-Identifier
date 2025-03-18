Usage
=====

Command Line Interface
--------------------

The package provides a command-line interface for easy interaction::

    # Train the ensemble model
    python -m src.cli --mode train --model ensemble

    # Make predictions
    python -m src.cli --mode predict

    # Evaluate model performance
    python -m src.cli --mode evaluate --model xgb

    # Monitor model drift
    python -m src.cli --mode monitor --model ensemble

Python API
---------

Here's an example of using the Python API::

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