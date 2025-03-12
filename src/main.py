import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from src.utils.data_loader import DataLoader
from src.models.xgboost_model import XGBoostModel

def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point with robust error handling and logging.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Resolve paths correctly
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        
        # Initialize data loader with correct path
        data_loader = DataLoader(str(data_dir))
        
        # Load and preprocess data with explicit error handling
        try:
            df = data_loader.load_csv("historical_random_numbers.csv")
            df = data_loader.preprocess_data(df)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Create synthetic data as fallback
            dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            numbers = np.random.randint(1, 11, size=len(dates))
            df = pd.DataFrame({'Date': dates, 'Number': numbers})
        
        # Initialize and train the model
        model = XGBoostModel()
        X_train, X_test, y_train, y_test = data_loader.load_and_prepare_data("historical_random_numbers.csv", "Number")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        performance_metrics = model.evaluate(X_test, y_test)
        logger.info(f"Model performance: {performance_metrics}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    main()