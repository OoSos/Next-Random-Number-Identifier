from statsmodels.tsa.arima.model import ARIMA
from models.base_model import BaseModel

class HybridForecaster(BaseModel):
    def __init__(self, ml_model, arima_order=(1, 0, 0)):
        super().__init__(name="HybridForecaster")
        self.ml_model = ml_model
        self.arima_order = arima_order
        self.arima_model = None
        self.feature_importance_ = None
    
    def fit(self, X, y):
        # First, fit the machine learning model
        self.ml_model.fit(X, y)
        
        # Get ML model predictions
        ml_predictions = self.ml_model.predict(X)
        
        # Calculate residuals (what ML model couldn't predict)
        residuals = y - ml_predictions
        
        # Fit ARIMA on the residuals
        self.arima_model = ARIMA(residuals, order=self.arima_order)
        self.arima_results = self.arima_model.fit()
        
        # Store feature importance from ML model
        self.feature_importance_ = self.ml_model.get_feature_importance()
        
        return self
    
    def predict(self, X):
        # Get ML model predictions
        ml_predictions = self.ml_model.predict(X)
        
        # Forecast residuals using ARIMA
        n_steps = len(X)
        arima_forecast = self.arima_results.forecast(steps=n_steps)
        
        # Combine predictions
        combined_predictions = ml_predictions + arima_forecast
        
        return combined_predictions
    
    def get_feature_importance(self):
        return self.feature_importance_