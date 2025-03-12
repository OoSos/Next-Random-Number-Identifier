import pandas as pd
from src.utils.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.markov_chain import MarkovChain
from src.models.ensemble import EnhancedEnsemble
from src.models.hybrid_forecaster import HybridForecaster
from src.utils.evaluation import ModelEvaluator
from src.visualization.plots import plot_predictions, plot_feature_importance

def main():
    # Initialize data loader
    data_loader = DataLoader("data")
    
    # Load and preprocess data
    df = data_loader.load_csv("historical_random_numbers.csv")
    df = data_loader.preprocess_data(df)
    
    # Convert 'Number' column to numeric, placing NaN where non-numeric values exist
    if 'Number' in df.columns:
        df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
        df = df[df['Number'].notna()]
    
    # Create features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.transform(df)
    
    # Select important features
    feature_selector = FeatureSelector(n_features=20)
    X = df_features.drop(['Date', 'Number'], axis=1).fillna(0)
    y = df_features['Number']
    print('Checking for NaNs in y:', y.isna().sum())
    y = y.fillna(y.median())
    feature_selector.fit(X, y)
    X_selected = feature_selector.transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = data_loader.split_data(
        pd.concat([X_selected, y], axis=1), 'Number'
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