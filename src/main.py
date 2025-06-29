# Standard library imports
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Local application imports
from src.utils.enhanced_data_loader import EnhancedDataLoader
from src.models.xgboost_model import XGBoostModel
try:
    from src.features.feature_engineering import FeatureEngineer
    from src.models.random_forest import RandomForestModel
    from src.models.markov_chain import MarkovChain
    from src.models.ensemble import EnhancedEnsemble
    from src.models.adaptive_ensemble import AdaptiveEnsemble
    from src.models.hybrid_forecaster import HybridForecaster
    FULL_MODELS_AVAILABLE = True
except ImportError:
    FULL_MODELS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Utility functions

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration for the project.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across the codebase.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    column_mapping = {
        'Super Ball': 'Number',
        'super ball': 'Number',
        'super_ball': 'Number',
        'superball': 'Number',
        'SUPER BALL': 'Number',
        'Ball': 'Number'
    }
    
    # Apply mapping to rename columns
    return df.rename(columns=column_mapping)


def debug_file_path(file_path: str = None, file_name: str = "historical_random_numbers.csv") -> dict:
    """
    Debug function to check file paths and verify CSV file accessibility.
    
    Args:
        file_path: Full path to the file (optional)
        file_name: Name of the file to check (if file_path not provided)
        
    Returns:
        Dictionary with path information
    """
    # Determine project root
    current_dir = Path(os.getcwd())
    project_root = current_dir
    
    # Look for common project markers to find the actual root
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "setup.py").exists() or (parent / "requirements.txt").exists():
            project_root = parent
            break
            
    # Resolve the data directory
    data_dir = project_root / "data"
    
    # Resolve the full file path
    if file_path is None:
        csv_path = data_dir / file_name
    else:
        csv_path = Path(file_path)
    
    # Print debug information
    logger.info(f"Current working directory: {current_dir}")
    logger.info(f"Project root directory: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"CSV file path: {csv_path}")
    logger.info(f"Data directory exists: {data_dir.exists()}")
    logger.info(f"CSV file exists: {csv_path.exists()}")
    
    result = {
        "cwd": str(current_dir),
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "file_path": str(csv_path),
        "data_dir_exists": data_dir.exists(),
        "file_exists": csv_path.exists(),
        "file_content": None
    }
    
    # Try to peek at the file content
    if csv_path.exists():
        try:
            with open(csv_path, 'r') as f:
                head = [next(f) for _ in range(5) if f]
            result["file_content"] = head
            logger.info(f"File content (first 5 lines): {head}")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            result["error"] = str(e)
            
    return result


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if the directory exists or was created, False otherwise
    """
    path = Path(directory_path)
    
    if path.exists() and path.is_dir():
        return True
        
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {str(e)}")
        return False

# Export public symbols
__all__ = [
    'setup_logging',
    'standardize_column_names',
    'debug_file_path',
    'ensure_directory_exists',
]


def main(data_path=None, model_type='ensemble'):
    """
    Main entry point with robust error handling and logging.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    results = {'success': False, 'models': {}, 'metrics': {}}

    try:
        # Resolve paths correctly
        if data_path is None:
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_path = data_dir / "historical_random_numbers.csv"
        else:
            data_path = Path(data_path)
            
        logger.info(f"Using data file: {data_path}")
        
        # Check if file exists
        if not data_path.exists():
            logger.warning(f"Data file {data_path} not found. Running debug path check...")
            path_debug_info = debug_file_path(data_path.name)
            data_path = Path(path_debug_info["file_path"])
            
            if not data_path.exists():
                logger.warning("Creating synthetic data for modeling...")
                # Create synthetic data
                data_dir = Path(path_debug_info["data_dir"])
                if not data_dir.exists():
                    data_dir.mkdir(parents=True, exist_ok=True)
                
                dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
                numbers = np.random.randint(1, 11, size=len(dates))
                df = pd.DataFrame({'Date': dates, 'Number': numbers})
                df.to_csv(data_path, index=False)
                logger.info(f"Created and saved synthetic data to {data_path}")
            else:
                data_path = Path(path_debug_info["file_path"])
        
        # Load and preprocess data with more robust error handling
        data_loader = EnhancedDataLoader(str(data_path.parent))
        df = data_loader.load_csv(data_path.name)
        
        if df.empty:
            logger.error("Failed to load data. Dataframe is empty.")
            results['error'] = "Failed to load data"
            return results
        
        df = data_loader.preprocess_data(df)
        logger.info(f"Preprocessed data shape: {df.shape}")
        
        # Create features
        if FULL_MODELS_AVAILABLE:
            try:
                feature_engineer = FeatureEngineer()
                df_features = feature_engineer.transform(df)
                logger.info(f"Created {len(df_features.columns)} features")
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
        
        # Prepare data for models
        feature_cols = [col for col in df_features.columns if col not in ['Date', 'Number']]
        if not feature_cols:
            logger.warning("No feature columns found. Using basic synthetic features.")
            df_features['feature_1'] = np.random.randn(len(df_features))
            df_features['feature_2'] = np.random.randn(len(df_features))
            feature_cols = ['feature_1', 'feature_2']
            
        X = df_features[feature_cols].fillna(0)
        if 'Number' in df_features.columns:
            y = df_features['Number']
        else:
            y = pd.Series(np.random.randint(1, 11, size=len(df_features)))
            logger.warning("Target column 'Number' not found. Using synthetic target.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples)")
          # Train selected model(s)
        if not FULL_MODELS_AVAILABLE:
            logger.warning("Full models not available. Defaulting to XGBoost only.")
            model_type = 'xgb'
        
        if model_type == 'rf' or model_type == 'ensemble':
            logger.info("Training Random Forest model...")
            rf_model = RandomForestModel(n_estimators=100)
            rf_model.fit(X_train, y_train)
            rf_metrics = rf_model.evaluate(X_test, y_test)
            results['models']['rf'] = rf_model
            results['metrics']['rf'] = rf_metrics
            logger.info(f"Random Forest performance: {rf_metrics}")
        
        if model_type == 'xgb' or model_type == 'ensemble':
            logger.info("Training XGBoost model...")
            xgb_model = XGBoostModel(n_estimators=100)
            xgb_model.fit(X_train, y_train)
            xgb_metrics = xgb_model.evaluate(X_test, y_test)
            results['models']['xgb'] = xgb_model
            results['metrics']['xgb'] = xgb_metrics
            logger.info(f"XGBoost performance: {xgb_metrics}")
        
        if model_type == 'markov' or model_type == 'ensemble':
            logger.info("Training Markov Chain model...")
            markov_model = MarkovChain(order=2)
            markov_model.fit(X_train, y_train)
            markov_metrics = markov_model.evaluate(X_test, y_test)
            results['models']['markov'] = markov_model
            results['metrics']['markov'] = markov_metrics
            logger.info(f"Markov Chain performance: {markov_metrics}")
        
        if model_type == 'hybrid' and FULL_MODELS_AVAILABLE:
            logger.info("Training Hybrid Forecaster model...")
            rf_model = RandomForestModel(n_estimators=100)
            hybrid_model = HybridForecaster(ml_model=rf_model, arima_order=(1, 0, 1))
            hybrid_model.fit(X_train, y_train)
            hybrid_pred = hybrid_model.predict(X_test)
            hybrid_metrics = {'mse': mean_squared_error(y_test, hybrid_pred)}
            results['models']['hybrid'] = hybrid_model
            results['metrics']['hybrid'] = hybrid_metrics
            logger.info(f"Hybrid model performance: {hybrid_metrics}")
        
        # Train ensemble if requested
        if model_type == 'ensemble' and FULL_MODELS_AVAILABLE:
            logger.info("Training Enhanced Ensemble model...")
            models = [results['models'][m] for m in ['rf', 'xgb', 'markov'] if m in results['models']]
            ensemble = EnhancedEnsemble(models=models)
            ensemble.fit(X_train, y_train)
            ensemble_pred = ensemble.predict(X_test)
            ensemble_metrics = {'mse': mean_squared_error(y_test, ensemble_pred)}
            results['models']['ensemble'] = ensemble
            results['metrics']['ensemble'] = ensemble_metrics
            logger.info(f"Ensemble model performance: {ensemble_metrics}")

        if model_type == 'adaptive_ensemble' and FULL_MODELS_AVAILABLE:
            logger.info("Training Adaptive Ensemble model...")
            models = [results['models'][m] for m in ['rf', 'xgb', 'markov'] if m in results['models']]
            adaptive_ensemble = AdaptiveEnsemble(models=models)
            adaptive_ensemble.fit(X_train, y_train)
            adaptive_ensemble_pred = adaptive_ensemble.predict(X_test)
            adaptive_ensemble_metrics = {'mse': mean_squared_error(y_test, adaptive_ensemble_pred)}
            results['models']['adaptive_ensemble'] = adaptive_ensemble
            results['metrics']['adaptive_ensemble'] = adaptive_ensemble_metrics
            logger.info(f"Adaptive Ensemble model performance: {adaptive_ensemble_metrics}")
        
        # Store success status
        results['success'] = True
        return results
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        results['error'] = str(e)
        return results

class PredictionPipeline:
    """
    End-to-end pipeline for random number prediction.
    """
    def __init__(self, data_loader, feature_engineer, models, ensemble=None):
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.models = models
        self.ensemble = ensemble
        self.prediction_history = []
        self.confidence_history = []

    def predict_next(self, data_path: str) -> Dict[str, Any]:
        """
        Predict the next number in the sequence.
        """
        # Load and process data
        df = self.data_loader.load_and_preprocess(data_path)
        
        # Create features
        df_features = self.feature_engineer.transform(df)
        
        # Prepare data for prediction
        X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
        
        # Get individual model predictions
        model_predictions = {}
        for name, model in self.models.items():
            try:
                model_predictions[name] = model.predict(X.tail(1))[0]
            except Exception as e:
                logging.error(f"Error in model {name} prediction: {str(e)}")
                model_predictions[name] = None
        
        # Get ensemble prediction if available
        ensemble_prediction = None
        if self.ensemble:
            try:
                ensemble_prediction = self.ensemble.predict(X.tail(1))[0]
            except Exception as e:
                logging.error(f"Error in ensemble prediction: {str(e)}")
                ensemble_prediction = None
        
        # Determine confidence level
        # Higher confidence when models agree
        model_values = [pred for pred in model_predictions.values() if pred is not None]
        if model_values:
            agreement_ratio = max(model_values.count(x) for x in set(model_values)) / len(model_values)
        else:
            agreement_ratio = 0
        
        # Track prediction and confidence history
        self.prediction_history.append(ensemble_prediction if ensemble_prediction is not None else max(model_predictions.items(), key=lambda x: x[1])[1])
        self.confidence_history.append(agreement_ratio)
        
        return {
            'individual_predictions': model_predictions,
            'ensemble_prediction': ensemble_prediction,
            'confidence': agreement_ratio,
            'most_likely': ensemble_prediction if ensemble_prediction is not None 
                         else max(model_predictions.items(), key=lambda x: x[1])[1]
        }

    def get_prediction_history(self) -> List[Any]:
        """
        Get the history of predictions.
        """
        return self.prediction_history

    def get_confidence_history(self) -> List[float]:
        """
        Get the history of confidence levels.
        """
        return self.confidence_history

if __name__ == "__main__":
    main()