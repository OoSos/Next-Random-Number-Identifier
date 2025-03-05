# src/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_predictions(y_true, predictions, title="Model Predictions"):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.index, y_true, label='Actual', marker='o')
    plt.plot(y_true.index, predictions, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    return plt

def plot_feature_importance(feature_importance, title="Feature Importance"):
    """Plot feature importance scores."""
    importance_df = pd.DataFrame(feature_importance.items(), 
                               columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title(title)
    plt.tight_layout()
    return plt