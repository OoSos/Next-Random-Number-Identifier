import pandas as pd
from utils.data_loader import DataLoader
from features.feature_engineering import FeatureEngineer
from features.feature_selection import FeatureSelector
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.markov_chain import MarkovChain
from models.ensemble import EnhancedEnsemble
from models.hybrid_forecaster import HybridForecaster
from utils.evaluation import ModelEvaluator
from visualization.plots import plot_predictions, plot_feature_importance

def main():
    # Initialize data loader
    data_loader = DataLoader("data")
    
    # Load and preprocess data
    df = data_loader.load_csv("historical_random_numbers.csv")
    df = data_loader.preprocess_data(df)
    
    # Convert 'Super Ball' column to numeric, placing NaN where non-numeric values exist
    if 'Super Ball' in df.columns:
        df['Super Ball'] = pd.to_numeric(df['Super Ball'], errors='coerce')
        df = df[df['Super Ball'].notna()]
    
    # Create features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.transform(df)
    
    # Select important features
    feature_selector = FeatureSelector(n_features=20)
    X = df_features.drop(['Date', 'Super Ball'], axis=1).fillna(0)
    y = df_features['Super Ball']
    print('Checking for NaNs in y:', y.isna().sum())
    y = y.fillna(y.median())
    feature_selector.fit(X, y)
    X_selected = feature_selector.transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = data_loader.split_data(
        pd.concat([X_selected, y], axis=1), 'Super Ball'
    )
    
    # Train individual models
    rf_model = RandomForestModel()
    xgb_model = XGBoostModel()
    markov_model = MarkovChain()
    
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    markov_model.fit(X_train, y_train)
    
    # Train ensemble model
    ensemble = EnhancedEnsemble(models=[rf_model, xgb_model, markov_model])
    ensemble.fit(X_train, y_train)
    
    # Train hybrid forecaster (combining ML with time series)
    hybrid_model = HybridForecaster(ml_model=rf_model)
    hybrid_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_ensemble = ensemble.predict(X_test)
    y_pred_hybrid = hybrid_model.predict(X_test)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    ensemble_metrics = evaluator.evaluate_regression(y_test, y_pred_ensemble)
    hybrid_metrics = evaluator.evaluate_regression(y_test, y_pred_hybrid)
    
    print("Ensemble model performance:", ensemble_metrics)
    print("Hybrid model performance:", hybrid_metrics)
    
    # Visualize results
    plot_predictions(y_test, y_pred_ensemble, "Ensemble Predictions").show()
    plot_predictions(y_test, y_pred_hybrid, "Hybrid Model Predictions").show()
    plot_feature_importance(ensemble.feature_importance_, "Ensemble Feature Importance").show()
    
    return {
        "ensemble_metrics": ensemble_metrics,
        "hybrid_metrics": hybrid_metrics,
        "models": {
            "rf": rf_model,
            "xgb": xgb_model,
            "markov": markov_model,
            "ensemble": ensemble,
            "hybrid": hybrid_model
        }
    }

if __name__ == "__main__":
    main()