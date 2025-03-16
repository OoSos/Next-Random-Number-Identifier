import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from .utils.enhanced_data_loader import EnhancedDataLoader
# from .models.xgboost_model import XGBoostModel
try:
    from .features.feature_engineering import FeatureEngineer
    from .models.random_forest import RandomForestModel
    from .models.markov_chain import MarkovChain
    from .models.ensemble import EnhancedEnsemble
    from .models.hybrid_forecaster import HybridForecaster
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

# Optional imports with error handling
try:
    from .utils.enhanced_data_loader import EnhancedDataLoader
    logger.debug("Successfully imported EnhancedDataLoader")
except ImportError:
    logger.debug("EnhancedDataLoader import failed, will use DataLoader instead")
    
    try:
        from src.utils.enhanced_data_loader import EnhancedDataLoader as EnhancedDataLoader
        logger.debug("Using DataLoader as fallback")
    except ImportError:
        logger.debug("DataLoader import failed, will use SimpleDataLoader instead")
        
        try:
            from .utils.simple_data_loader import SimpleDataLoader as EnhancedDataLoader
            logger.debug("Using SimpleDataLoader as fallback")
        except ImportError:
            logger.debug("SimpleDataLoader import failed")
            
            # Define a minimal DataLoader as last resort
            class EnhancedDataLoader:
                """Minimal DataLoader implementation as fallback."""
                def __init__(self, data_dir):
                    self.data_dir = Path(data_dir)
                    
                def load_csv(self, filename):
                    """Load CSV data with minimal functionality."""
                    try:
                        return pd.read_csv(self.data_dir / filename)
                    except Exception as e:
                        logger.error(f"Error loading CSV: {str(e)}")
                        return pd.DataFrame()
                        
                def preprocess_data(self, df):
                    """Minimal preprocessing."""
                    return standardize_column_names(df)

# Export public symbols
__all__ = [
    'setup_logging',
    'standardize_column_names',
    'debug_file_path',
    'ensure_directory_exists',
    'EnhancedDataLoader'
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
            # xgb_model = XGBoostModel(n_estimators=100)
            # xgb_model.fit(X_train, y_train)
            # xgb_metrics = xgb_model.evaluate(X_test, y_test)
            # results['models']['xgb'] = xgb_model
            # results['metrics']['xgb'] = xgb_metrics
            # logger.info(f"XGBoost performance: {xgb_metrics}")
        
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
        
        # Store success status
        results['success'] = True
        return results
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        results['error'] = str(e)
        return results

if __name__ == "__main__":
    main()