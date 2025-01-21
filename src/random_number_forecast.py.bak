import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from datetime import timedelta
from collections import defaultdict

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

# Feature engineering with lag features and additional features
def create_features(df):
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    for lag in range(1, 8):
        df[f'Lag_{lag}'] = df['Number'].shift(lag)
    
    # Add rolling statistics
    df['RollingMean'] = df['Number'].rolling(window=5).mean()
    df['RollingStd'] = df['Number'].rolling(window=5).std()
    
    # Hot and Cold numbers
    number_counts = df['Number'].value_counts()
    df['IsHot'] = df['Number'].map(lambda x: number_counts[x] > number_counts.median())
    df['IsCold'] = df['Number'].map(lambda x: number_counts[x] < number_counts.median())
    
    return df.dropna()

# Markov Chain Analysis
class MarkovChain:
    def __init__(self, order=1):
        self.order = order
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)

    def train(self, sequence):
        sequence = sequence.reset_index(drop=True)  # Reset index to avoid KeyError
        for i in range(len(sequence) - self.order):
            current_state = tuple(sequence.iloc[i:i+self.order])
            if i + self.order < len(sequence):
                next_state = sequence.iloc[i+self.order]
                self.transition_matrix[current_state][next_state] += 1
                self.state_counts[current_state] += 1
        
        # Convert counts to probabilities
        for state in self.transition_matrix:
            total = sum(self.transition_matrix[state].values())
            if total > 0:
                for next_state in self.transition_matrix[state]:
                    self.transition_matrix[state][next_state] /= total

    def predict(self, current_state):
        if current_state not in self.transition_matrix or not self.transition_matrix[current_state]:
            return np.random.randint(1, 11)  # If unseen state or no transitions, return random number
        probabilities = list(self.transition_matrix[current_state].items())
        states, probs = zip(*probabilities)
        return np.random.choice(states, p=probs)

# Train the Random Forest model
def train_rf_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Train the XGBoost model
def train_xgb_model(X, y):
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
    model.fit(X, y)
    return model

# Train and predict using Markov Chain
def train_and_predict_markov(y_train, y_test):
    model = MarkovChain(order=2)  # Increased order to 2
    model.train(y_train)
    
    predictions = []
    for i in range(len(y_test)):
        if i == 0:
            current_state = tuple(y_train.iloc[-2:])
        elif i == 1:
            current_state = (y_train.iloc[-1], predictions[-1])
        else:
            current_state = tuple(predictions[-2:])
        predictions.append(model.predict(current_state))
    
    return predictions

# Make predictions
def make_predictions(model, X):
    return model.predict(X)

# Evaluate the models
def evaluate_models(y_true, y_pred_rf, y_pred_xgb, y_pred_markov):
    # Random Forest evaluation
    rf_mse = mean_squared_error(y_true, y_pred_rf)
    rf_mae = mean_absolute_error(y_true, y_pred_rf)
    
    # XGBoost evaluation
    xgb_accuracy = accuracy_score(y_true - 1, y_pred_xgb)  # Subtract 1 to align with 0-based XGBoost predictions
    xgb_precision = precision_score(y_true - 1, y_pred_xgb, average='weighted')
    xgb_recall = recall_score(y_true - 1, y_pred_xgb, average='weighted')
    xgb_f1 = f1_score(y_true - 1, y_pred_xgb, average='weighted')
    
    # Markov Chain evaluation
    markov_mse = mean_squared_error(y_true, y_pred_markov)
    markov_mae = mean_absolute_error(y_true, y_pred_markov)
    markov_accuracy = accuracy_score(y_true, y_pred_markov)
    markov_precision = precision_score(y_true, y_pred_markov, average='weighted')
    markov_recall = recall_score(y_true, y_pred_markov, average='weighted')
    markov_f1 = f1_score(y_true, y_pred_markov, average='weighted')
    
    return {
        'Random Forest': {'MSE': rf_mse, 'MAE': rf_mae},
        'XGBoost': {'Accuracy': xgb_accuracy, 'Precision': xgb_precision, 'Recall': xgb_recall, 'F1': xgb_f1},
        'Markov Chain': {'MSE': markov_mse, 'MAE': markov_mae, 'Accuracy': markov_accuracy, 
                         'Precision': markov_precision, 'Recall': markov_recall, 'F1': markov_f1}
    }

# Visualize results
def plot_results(y_true, y_pred_rf, y_pred_xgb, y_pred_markov):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Actual')
    plt.plot(y_true.index, y_pred_rf, label='Random Forest')
    plt.plot(y_true.index, y_pred_xgb, label='XGBoost')
    plt.plot(y_true.index, y_pred_markov, label='Markov Chain')
    plt.legend()
    plt.title('Actual vs Predicted Numbers')
    plt.xlabel('Sample')
    plt.ylabel('Number')
    plt.show()

# Time series plot
def plot_time_series(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Number'])
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.title('Time Series Plot')
    plt.show()

# Frequency analysis
def frequency_analysis(data):
    frequency_counts = data['Number'].value_counts()
    most_frequent_number = frequency_counts.index[0]
    predicted_number_frequency = frequency_counts.idxmax()
    
    print("\nFrequency Analysis:")
    print(frequency_counts)
    print(f"Most frequent number: {most_frequent_number}")
        
    return predicted_number_frequency

# Predict next date
def predict_next_date(df):
    last_date = df['Date'].iloc[-1]
    date_diffs = df['Date'].diff().dt.days
    
    last_diffs = date_diffs.tail(4).tolist()
    
    if last_diffs[-1] == 3:
        next_diff = 4
    elif last_diffs[-1] == 4:
        if last_diffs[-2] == 3:
            next_diff = 3
        else:
            next_diff = 4
    else:
        next_diff = 3
    
    next_date = last_date + timedelta(days=next_diff)
    
    return next_date, next_diff

# Main function
def main(file_path):
    # Load and prepare data
    df = load_data(file_path)
    
    # Perform frequency analysis
    predicted_number_frequency = frequency_analysis(df)
    
    # Plot time series
    plot_time_series(df)
    
    # Prepare data for models
    df = create_features(df)
    
    # Split features and target
    X = df.drop(['Date', 'Number'], axis=1)
    y = df['Number']
    y_class = y - 1  # For XGBoost classification
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, _, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = train_rf_model(X_train, y_train)
    xgb_model = train_xgb_model(X_train, y_train_class)
    
    # Make predictions
    rf_pred = make_predictions(rf_model, X_test)
    xgb_pred = make_predictions(xgb_model, X_test)
    markov_pred = train_and_predict_markov(y_train, y_test)
    
    # Evaluate models
    evaluation_results = evaluate_models(y_test, rf_pred, xgb_pred, markov_pred)
    
    # Print evaluation results
    for model, metrics in evaluation_results.items():
        print(f"\n{model} Model Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Visualize results
    plot_results(y_test, rf_pred, xgb_pred, markov_pred)
    
    # Feature importance for Random Forest
    rf_feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
    print("\nTop 5 important features (Random Forest):")
    print(rf_feature_importance.sort_values('importance', ascending=False).head())
    
    # Feature importance for XGBoost
    xgb_feature_importance = pd.DataFrame({'feature': X.columns, 'importance': xgb_model.feature_importances_})
    print("\nTop 5 important features (XGBoost):")
    print(xgb_feature_importance.sort_values('importance', ascending=False).head())
    
    # Markov Chain transition probabilities
    markov_model = MarkovChain(order=2)
    markov_model.train(df['Number'])
    print("\nTop 5 most probable transitions (Markov Chain):")
    all_transitions = [(state, next_state, prob) 
                       for state in markov_model.transition_matrix 
                       for next_state, prob in markov_model.transition_matrix[state].items()]
    all_transitions.sort(key=lambda x: x[2], reverse=True)
    for state, next_state, prob in all_transitions[:5]:
        print(f"From {state} to {next_state}: {prob:.2f}")
    
    # Predict next date
    next_date, next_diff = predict_next_date(df)
    
    # Prepare last row for prediction
    last_row = X.iloc[-1:].copy()
    for lag in range(1, 8):
        last_row[f'Lag_{lag}'] = df['Number'].iloc[-lag]
    last_row['DayOfWeek'] = next_date.dayofweek
    last_row['Month'] = next_date.month
    last_row['Year'] = next_date.year
    last_row['DayOfYear'] = next_date.dayofyear
    last_row['RollingMean'] = df['Number'].tail(5).mean()
    last_row['RollingStd'] = df['Number'].tail(5).std()
    last_row['IsHot'] = df['Number'].iloc[-1] in df[df['IsHot']]['Number'].unique()
    last_row['IsCold'] = df['Number'].iloc[-1] in df[df['IsCold']]['Number'].unique()
    
    # Predict next number using all models
    predicted_number_rf = int(round(rf_model.predict(last_row)[0]))
    predicted_number_xgb = int(xgb_model.predict(last_row)[0]) + 1  # Add 1 to adjust for 0-based classification
    predicted_number_markov = markov_model.predict(tuple(df['Number'].tail(2)))
    
    print(f"\nNext date for next number: {next_date.date()} (Interval: {next_diff} days)")
    print(f"Forecasted next number (Random Forest): {predicted_number_rf}")
    print(f"Forecasted next number (XGBoost): {predicted_number_xgb}")
    print(f"Forecasted next number (Markov Chain): {predicted_number_markov}")
    print(f"Forecasted next number (Frequency Analysis): {predicted_number_frequency}")
    
    # Hot and Cold numbers strategy
    print("\nHot and Cold Numbers Strategy:")
    recent_numbers = df['Number'].tail(5).tolist()
    hot_numbers = df[df['IsHot']]['Number'].unique()
    cold_numbers = df[df['IsCold']]['Number'].unique()
    
    print("Recent numbers:", recent_numbers)
    print("Hot numbers to avoid:", [n for n in hot_numbers if n in recent_numbers[-2:]])
    print("Cold numbers to consider:", [n for n in cold_numbers if n not in recent_numbers])

if __name__ == "__main__":
    file_path = "data/historical_random_numbers.csv"
    main(file_path)