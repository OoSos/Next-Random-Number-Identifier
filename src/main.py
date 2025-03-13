import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.utils.data_loader import DataLoader
from src.models.xgboost_model import XGBoostModel
try:
    from src.features.feature_engineering import FeatureEngineer
    from src.models.random_forest import RandomForestModel
    from src.models.markov_chain import MarkovChain
    from src.models.ensemble import EnhancedEnsemble
    from src.models.hybrid_forecaster import HybridForecaster
    FULL_MODELS_AVAILABLE = True
except ImportError:
    FULL_MODELS_AVAILABLE = False
    
def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_file_path(file_name="historical_random_numbers.csv"):
    """
    Debug function to check file paths and verify CSV file accessibility.
    """
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).parent.parent.absolute()
    data_dir = project_root / "data"
    file_path = data_dir / file_name
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Project root directory: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"CSV file path: {file_path}")
    logger.info(f"Data directory exists: {data_dir.exists()}")
    logger.info(f"CSV file exists: {file_path.exists()}")
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist. Creating it now.")
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {data_dir}")
        except Exception as e:
            logger.error(f"Failed to create data directory: {str(e)}")
    
    if file_path.exists():
        # Get file size
        file_size = file_path.stat().st_size
        logger.info(f"CSV file size: {file_size} bytes")
        
        # Read first few lines
        try:
            with open(file_path, 'r') as f:
                head = [next(f).strip() for _ in range(5) if f]
            logger.info(f"CSV file first 5 lines: {head}")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
    else:
        logger.warning(f"CSV file does not exist at: {file_path}")
    
    return {
        "cwd": os.getcwd(),
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "file_path": str(file_path),
        "data_dir_exists": data_dir.exists(),
        "file_exists": file_path.exists()
    }

def main(data_path=None, model_type='ensemble'):
    """
    Main entry point with robust error handling and logging.
    
    Args:
        data_path: Path to the data file. If None, uses default path
        model_type: Type of model to use ('rf', 'xgb', 'markov', 'ensemble', 'hybrid')
        
    Returns:
        Dict with results including trained models and performance metrics
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
        
        # Check if file exists before proceeding
        if not data_path.exists():
            logger.warning(f"Data file {data_path} not found. Running debug path check...")
            path_debug_info = debug_file_path(data_path.name)
            
            # If file still doesn't exist, try to create synthetic data
            if not Path(path_debug_info["file_path"]).exists():
                logger.warning("Creating synthetic data for modeling")
                data_dir = Path(path_debug_info["data_dir"])
                if not data_dir.exists():
                    data_dir.mkdir(parents=True, exist_ok=True)
                
                # Create synthetic data
                dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
                numbers = np.random.randint(1, 11, size=len(dates))
                df = pd.DataFrame({'Date': dates, 'Number': numbers})
                
                # Save synthetic data
                df.to_csv(data_path, index=False)
                logger.info(f"Created and saved synthetic data to {data_path}")
            else:
                data_path = Path(path_debug_info["file_path"])
        
        # Create data loader
        data_loader = DataLoader(str(data_path.parent))
        
        # Load and preprocess data
        df = data_loader.load_csv(data_path.name)
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
        
        # Store success status
        results['success'] = True
        return results
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        results['error'] = str(e)
        return results

if __name__ == "__main__":
    main()