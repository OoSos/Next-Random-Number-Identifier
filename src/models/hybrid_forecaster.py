from statsmodels.tsa.arima.model import ARIMA
from src.models.base_model import BaseModel
import pandas as pd
import numpy as np

class HybridForecaster(BaseModel):
    """
    Hybrid model combining machine learning and ARIMA for time series forecasting.
    
    Attributes:
        ml_model: The machine learning model to be used.
        arima_order: The order of the ARIMA model.
        arima_model: The ARIMA model instance.
        feature_importance_: Feature importance from the machine learning model.
    """
    
    def __init__(self, ml_model, arima_order=(1, 0, 0)):
        """
        Initialize the HybridForecaster.
        
        Args:
            ml_model: The machine learning model to be used.
            arima_order: The order of the ARIMA model.
        """
        super().__init__(name="HybridForecaster")
        self.ml_model = ml_model
        self.arima_order = arima_order
        self.arima_model = None
        self.feature_importance_ = None
    
    def fit(self, X, y):
        """
        Fit the hybrid model.
        
        Args:
            X: Features for the machine learning model.
            y: Target variable.
        
        Returns:
            self: The fitted model instance.
        """
        # Validate input data
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X must be a DataFrame and y must be a Series")
        
        # First, fit the machine learning model
        self.ml_model.fit(X, y)
        
        # Get ML model predictions
        ml_predictions = self.ml_model.predict(X)
        
        # Calculate residuals (what ML model couldn't predict)
        residuals = y - ml_predictions
        
        # Ensure the index is a supported class
        residuals.index = pd.RangeIndex(start=0, stop=len(residuals), step=1)
        
        # Fit ARIMA on the residuals with method specification
        self.arima_model = ARIMA(residuals, order=self.arima_order)
        method_kwargs = {'maxiter': 500}
        self.arima_results = self.arima_model.fit(method='statespace', method_kwargs=method_kwargs)
        
        # Store feature importance from ML model
        self.feature_importance_ = self.ml_model.get_feature_importance()
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the hybrid model.
        
        Args:
            X: Features for the machine learning model.
        
        Returns:
            combined_predictions: Combined predictions from the ML model and ARIMA model.
        """
        # Validate input data
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        
        # Get ML model predictions
        ml_predictions = self.ml_model.predict(X)
        
        # Forecast residuals using ARIMA
        n_steps = len(X)
        arima_forecast = self.arima_results.forecast(steps=n_steps)
        
        # Combine predictions
        combined_predictions = ml_predictions + arima_forecast
        
        return combined_predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from the machine learning model.
        
        Returns:
            feature_importance_: Feature importance from the machine learning model.
        """
        return self.feature_importance_
    
    def estimate_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate prediction confidence using the ML model's predict_proba method.
        
        Args:
            X (pd.DataFrame): Features to estimate confidence for
            
        Returns:
            np.ndarray: Confidence estimates for each prediction
        """
        return self.ml_model.estimate_confidence(X)
    
    @staticmethod
    def usage_example():
        """
        Usage example for the HybridForecaster.
        """
        from sklearn.ensemble import RandomForestRegressor
        import pandas as pd
        import numpy as np
        
        # Generate sample data
        dates = pd.date_range(start='2020-01-01', periods=100)
        X = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)}, index=dates)
        y = pd.Series(np.random.randn(100), index=dates)
        
        # Initialize models
        ml_model = RandomForestRegressor()
        hybrid_model = HybridForecaster(ml_model=ml_model, arima_order=(1, 1, 1))
        
        # Fit the hybrid model
        hybrid_model.fit(X, y)
        
        # Make predictions
        predictions = hybrid_model.predict(X)
        print(predictions)

# Example usage
if __name__ == "__main__":
    HybridForecaster.usage_example()