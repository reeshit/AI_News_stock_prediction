import pandas as pd
import yfinance as yf
import numpy as np
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Fetch stock data function
def fetch_single_stock(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")  # Changed from "1d" to "1y" for more data
    return hist

# Load dataset (Replace with your actual dataset)
# Example: Generating random data for demonstration
time_steps = 60  # Example time step value
X_train = np.random.rand(1000, time_steps, 5)  # 1000 samples, 60 time steps, 5 features
y_train = np.random.rand(1000, 2)  # Predicting Open & Close prices
X_test = np.random.rand(200, time_steps, 5)  # Test set
y_test = np.random.rand(200, 2)

# Define the LSTM model with tunable hyperparameters
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units', min_value=50, max_value=200, step=50),
                   return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(hp.Int('units_2', min_value=50, max_value=200, step=50)))
    model.add(Dropout(0.2))
    model.add(Dense(hp.Int('dense_units', min_value=25, max_value=100, step=25), activation='relu'))
    model.add(Dense(2))  # Predicting Open and Close prices

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Initialize hyperparameter tuner
tuner = RandomSearch(
    build_lstm_model, 
    objective='val_loss', 
    max_trials=10,
    executions_per_trial=1
)

# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Retrieve the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Optimal LSTM Units: {best_hps.get('units')}")
print(f"Optimal LSTM Units 2: {best_hps.get('units_2')}")
print(f"Optimal Dense Units: {best_hps.get('dense_units')}")
