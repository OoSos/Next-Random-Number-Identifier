import argparse
import logging
import pandas as pd  # Standard import convention for pandas
from pathlib import Path
from src.main import main, setup_logging  # Use absolute import for consistency
from src.utils.enhanced_data_loader import EnhancedDataLoader
from src.utils import standardize_column_names  # Import centralized function
from src.utils.monitoring_pipeline import setup_monitoring, run_monitoring_cycle
import matplotlib.pyplot as plt
import json
from typing import Optional
from pydantic import BaseModel, ValidationError

try:
    from src.features.feature_engineering import FeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Next Random Number Identifier')
    parser.add_argument('--data', type=str, default='data/historical_random_numbers.csv',
                       help='Path to historical data CSV')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate', 'monitor', 'interactive'],
                       default='predict', help='Operation mode')
    parser.add_argument('--model', type=str, choices=['rf', 'xgb', 'markov', 'ensemble', 'hybrid'],
                       default='ensemble', help='Model type to use')
    parser.add_argument('--monitor', action='store_true', help='Enable model monitoring')
    parser.add_argument('--clean-data', action='store_true', help='Clean data before processing')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--batch', type=str, help='Path to batch processing file')
    return parser.parse_args()


def clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading CSV with pandas: {str(e)}")
        logger.info("Attempting manual CSV loading...")
        rows = []
        with open(file_path, 'r') as f:
            header = f.readline().strip().split(',')
            for line in f:
                try:
                    values = line.strip().split(',')
                    rows.append(dict(zip(header, values)))
                except Exception as line_e:
                    logger.error(f"Error parsing line: {line} - {str(line_e)}")
        
        df = pd.DataFrame(rows)
        
    # Standardize column names using the imported function
    df = standardize_column_names(df)
    
    if 'Number' in df.columns:
        df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    logger.info(f"Before cleaning: {df.shape}, NaN count: {df.isna().sum().sum()}")
    
    # Fill NaN values first
    df = df.ffill().bfill()  # Use the updated syntax instead of deprecated method
    
    # Then drop any remaining NaN values
    df.dropna(inplace=True)
    
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    if cleaned_file_path == file_path:  # Ensure we don't overwrite the original file
        cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    
    df.to_csv(cleaned_file_path, index=False)
    logger.info(f"After cleaning: {df.shape}, NaN count: {df.isna().sum().sum()}")
    return cleaned_file_path


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    for model, model_metrics in metrics.items():
        print(f"\n{model.upper()} Model Performance:")
        for metric, value in model_metrics.items():
            print(f"  {metric}: {value:.4f}")


def visualize_results(results):
    """Visualize the results using matplotlib."""
    for model, model_metrics in results['metrics'].items():
        plt.figure(figsize=(10, 5))
        plt.title(f"{model.upper()} Model Performance")
        plt.bar(model_metrics.keys(), model_metrics.values())
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.show()


class CLIConfig(BaseModel):
    data: str = 'data/historical_random_numbers.csv'
    mode: str = 'predict'
    model: str = 'ensemble'
    monitor: bool = False
    clean_data: bool = False
    batch: Optional[str] = None


def load_config(config_path):
    """Load and validate configuration from a JSON file using pydantic."""
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
    try:
        config = CLIConfig(**config_dict)
    except ValidationError as e:
        logger.error(f"Invalid configuration file: {e}")
        raise SystemExit(1)
    return config


def run_batch_processing(batch_path):
    """Run batch processing based on a configuration file."""
    with open(batch_path, 'r') as file:
        batch_config = json.load(file)
    for task in batch_config['tasks']:
        logger.info(f"Running task: {task['description']}")
        main(data_path=task['data_path'], model_type=task['model_type'])


def run_cli():
    """Run the CLI application with parsed arguments."""
    setup_logging()
    args = parse_args()

    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        data_path = config.data
        mode = config.mode
        model = config.model
    else:
        data_path = args.data
        mode = args.mode
        model = args.model

    # Clean data if requested
    if args.clean_data:
        data_path = clean_data(data_path)

    # Run batch processing if requested
    if args.batch:
        run_batch_processing(args.batch)
        return

    # Run the main functionality with the selected model
    logger.info(f"Running {mode} with model: {model}")
    results = main(data_path=data_path, model_type=model)

    if not results['success']:
        logger.error(f"Error: {results.get('error', 'Unknown error')}")
        return

    # Handle different modes
    if mode == 'train':
        logger.info(f"Training completed successfully.")
        print_metrics(results['metrics'])

    elif mode == 'predict':
        model_key = model
        # Fall back to ensemble if the requested model is not available
        if model_key not in results['models']:
            available_models = list(results['models'].keys())
            if available_models:
                model_key = available_models[0]
                logger.warning(f"Model '{model}' not available. Using '{model_key}' instead.")
            else:
                logger.error("No models available for prediction.")
                return

        model = results['models'][model_key]
        data_loader = EnhancedDataLoader(str(Path(data_path).parent))

        # Load latest data for prediction
        df = data_loader.load_csv(Path(data_path).name)
        df = data_loader.preprocess_data(df)

        # Create features
        if FEATURE_ENGINEER_AVAILABLE:
            try:
                feature_engineer = FeatureEngineer()
                df_features = feature_engineer.transform(df)
            except Exception as e:
                logger.error(f"Error in feature engineering: {str(e)}")
                df_features = df.copy()
                # Add some basic features as fallback
                if 'Date' in df_features.columns:
                    df_features['Year'] = df_features['Date'].dt.year
                    df_features['Month'] = df_features['Date'].dt.month
                    df_features['Day'] = df_features['Date'].dt.day
                    df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
        else:
            df_features = df.copy()
            # Add some basic features as fallback
            if 'Date' in df_features.columns:
                df_features['Year'] = df_features['Date'].dt.year
                df_features['Month'] = df_features['Date'].dt.month
                df_features['Day'] = df_features['Date'].dt.day
                df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
            logger.info("Used fallback feature engineering (FeatureEngineer not available)")

        # Prepare data for prediction, handling both feature formats
        if all(col in df_features.columns for col in ['Date', 'Number']):
            X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
        else:
            # Use all columns if Date/Number not present
            X = df_features.fillna(0)

        # Make predictions with selected model
        predictions = model.predict(X.tail(1))
        logger.info(f"Next predicted number: {predictions[0]}")
        print(f"Next predicted number: {predictions[0]}")

    elif mode == 'evaluate':
        print_metrics(results['metrics'])
        visualize_results(results)

    elif mode == 'monitor':
        # Monitor model performance over time
        model_name = model
        metrics = results['metrics'].get(model_name, results['metrics'].get('ensemble', {}))
        monitor = setup_monitoring(metrics)

        # This would typically be run with new metrics as they become available
        logger.info("Running monitoring cycle...")
        monitoring_results = run_monitoring_cycle(monitor, metrics)

        if monitoring_results.get('drift_detected', False):
            logger.warning("Model drift detected! Performance has changed significantly.")
            print("WARNING: Model drift detected!")
        else:
            logger.info("No significant model drift detected.")
            print("No significant model drift detected.")

    elif mode == 'interactive':
        # Interactive mode implementation
        while True:
            user_input = input("Enter command (train, predict, evaluate, monitor, exit): ")
            if user_input == 'exit':
                break
            elif user_input in ['train', 'predict', 'evaluate', 'monitor']:
                args.mode = user_input
                run_cli()
            else:
                print("Invalid command. Please try again.")


if __name__ == "__main__":
    run_cli()