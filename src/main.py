import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from src.utils.data_loader import DataLoader
from src.models.xgboost_model import XGBoostModel
try:
    from src.features.feature_engineering import FeatureEngineer
    from src.models.random_forest import RandomForestModel
    from src.models.markov_chain import MarkovChain
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

def main():
    """Main entry point with robust error handling and logging."""
    setup_logging()
    logger = logging.getLogger(__name__)
    results = {}

    try:
        # Debug file paths before attempting to load data
        path_debug_info = debug_file_path()
        logger.info(f"Path debug info: {path_debug_info}")
        
        # Resolve paths correctly
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Data directory exists: {data_dir.exists()}")
        
        # Create data directory if needed
        if not data_dir.exists():
            logger.info(f"Creating data directory: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # List files in data directory
        if data_dir.exists():
            logger.info("Files in data directory:")
            file_found = False
            for file in data_dir.iterdir():
                file_found = True
                logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
            if not file_found:
                logger.info("  (no files found)")
        
        # Initialize data loader
        data_loader = DataLoader(str(data_dir))
        
        # Load and preprocess data
        df = data_loader.load_csv("historical_random_numbers.csv")
        
        if df.empty:
            logger.warning("Loaded DataFrame is empty! Creating synthetic data.")
            # Create synthetic data
            dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            numbers = np.random.randint(1, 11, size=len(dates))
            df = pd.DataFrame({'Date': dates, 'Number': numbers})
            logger.info(f"Created synthetic data with {len(df)} rows")
            
            # Save synthetic data for future use
            synthetic_file_path = data_dir / "historical_random_numbers.csv"
            df.to_csv(synthetic_file_path, index=False)
            logger.info(f"Saved synthetic data to {synthetic_file_path}")
        
        df = data_loader.preprocess_data(df)
        logger.info(f"Preprocessed data shape: {df.shape}")

        # Create features if FeatureEngineer is available
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
                logger.info("Used fallback feature engineering")
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
        
        # Use our own split to ensure we have data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples)")
        
        # Initialize and train multiple models if available
        if FULL_MODELS_AVAILABLE:
            models = {
                'rf': RandomForestModel(n_estimators=100),
                'xgb': XGBoostModel(n_estimators=100),
                'markov': MarkovChain(order=2)
            }
            
            # Train models
            logger.info("Training models...")
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    if hasattr(model, 'evaluate'):
                        metrics = model.evaluate(X_test, y_test)
                    else:
                        # Simple MSE calculation if no evaluate method
                        pred = model.predict(X_test)
                        metrics = {'mse': float(np.mean((y_test - pred) ** 2))}
                    
                    logger.info(f"{name.upper()} model performance: {metrics}")
                    results[f"{name}_metrics"] = metrics
                except Exception as e:
                    logger.error(f"Error training {name} model: {str(e)}")
        else:
            # Just use XGBoost model
            logger.info("Training XGBoost model (other models not available)...")
            model = XGBoostModel(n_estimators=100)
            try:
                model.fit(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)
                logger.info(f"XGBoost model performance: {metrics}")
                results["xgb_metrics"] = metrics
            except Exception as e:
                logger.error(f"Error training XGBoost model: {str(e)}")
        
        # Store results
        results['success'] = True
        return results
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    main()