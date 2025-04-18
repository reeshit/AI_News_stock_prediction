import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import pipeline
import ta
import json  # for manual JSON parsing if needed

st.title("Enhanced NSE Stock Market Prediction via Upstox API v2")
st.write("This app predicts the next day’s opening and closing prices for an NSE stock using an LSTM model with extra features. Data is fetched in real-time using Upstox API v2.")

# --- Upstox API v2 Credentials ---
# Replace these with your actual Upstox API v2 credentials.
API_KEY = "YOUR_API_KEY"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

# --- Function to Fetch Historical Data using Upstox API v2 ---
@st.cache_data(show_spinner=False)
def load_data_upstox_v2(ticker_symbol):
    # Remove the '.NS' suffix if present.
    stock_symbol = ticker_symbol.split('.')[0]
    start_date = datetime(2015, 1, 1)
    end_date = datetime.today()
    
    # Placeholder URL—check Upstox API v2 docs for the correct endpoint.
    url = "https://api.upstox.com/historical/v2"
    
    # Parameters as required by the API v2.
    params = {
        "exchange": "NSE",               # For NSE stocks
        "symbol": stock_symbol,          # e.g., "RELIANCE"
        "interval": "1day",              # Daily data
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "x-api-key": API_KEY
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    # Debug: if needed, you can uncomment the next line to see the raw response.
    # st.write("Response text:", response.text)
    
    try:
        json_data = response.json()
    except Exception as e:
        st.error("Error parsing response as JSON: " + str(e))
        st.stop()
    
    # If the JSON data is still a string, try to parse it.
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except Exception as e:
            st.error("Failed to load JSON from string: " + str(e))
            st.stop()
    
    # Check if json_data is a dictionary and contains the expected key.
    if not isinstance(json_data, dict):
        st.error("Unexpected response format: " + str(json_data))
        st.stop()
    
    # Assuming the API returns a key "data" that contains the records.
    records = json_data.get("data", [])
    if not records:
        st.error("No records found in the API response.")
        st.stop()
    
    # Convert the list of records into a pandas DataFrame.
    df = pd.DataFrame(records)
    # Rename columns to title-case for consistency.
    df.rename(columns={
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'volume': 'Volume'
    }, inplace=True)
    
    # Convert timestamp to datetime and set as index if available.
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
    
    return df[['Open', 'Close']]

# 1. Data Acquisition: Get the ticker symbol from the user.
ticker = st.text_input("Enter NSE Stock Ticker (with .NS appended, e.g., RELIANCE.NS):", "RELIANCE.NS")

data = load_data_upstox_v2(ticker)
if data is None or data.empty:
    st.error("No data found for the ticker provided via Upstox API v2. Please check the ticker symbol and your API credentials.")
    st.stop()

st.write("### Historical Data (last 5 rows)")
st.dataframe(data.tail())

# 2. Feature Engineering: Add Technical Indicators (RSI, MACD, MACD Signal)
# Ensure 'Close' is a 1D Series.
close_series = data['Close'].squeeze()

# Compute RSI using a 14-day window.
data['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

# Compute MACD and its signal line.
macd = ta.trend.MACD(close=close_series)
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()

# Remove any rows with NaN values from indicator calculations.
data.dropna(inplace=True)

st.write("### Data with Technical Indicators (last 5 rows)")
st.dataframe(data[['Open', 'Close', 'RSI', 'MACD', 'MACD_Signal']].tail())

# 3. Prepare Features and Target for the model.
# Input features: Open, Close, RSI, MACD, MACD_Signal.
# Target: Next day’s Open and Close prices.
features = data[["Open", "Close", "RSI", "MACD", "MACD_Signal"]]
target = data[["Open", "Close"]]

# Normalize features and target separately.
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_X = scaler_X.fit_transform(features)
scaled_y = scaler_y.fit_transform(target)

# Create sequences using a sliding window approach.
def create_sequences(X_array, y_array, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X_array) - time_steps):
        X_seq.append(X_array[i:i+time_steps])
        y_seq.append(y_array[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 60  # Number of past days used for prediction.
X_seq, y_seq = create_sequences(scaled_X, scaled_y, time_steps)

# Split the data into training and testing sets (80% train, 20% test).
split_index = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

st.write("Training data shape:", X_train.shape)
st.write("Testing data shape:", X_test.shape)

# 4. News Sentiment Analysis Feature (Optional).
st.write("### News Sentiment Analysis")
news_headline = st.text_input("Enter a news headline about the stock (optional):", "")
if news_headline:
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment = sentiment_pipeline(news_headline)[0]
    st.write("News Sentiment:", sentiment)

# 5. Build the LSTM Model for multi-output regression (predicting both Open and Close).
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(2))  # Two outputs: predicted Open and predicted Close.
model.compile(optimizer='adam', loss='mean_squared_error')

if st.button("Train Model and Predict"):
    with st.spinner("Training model..."):
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    st.success("Training complete!")
    
    # 6. Make Predictions: Use test data and inverse transform predictions.
    predictions = model.predict(X_test)
    predictions_rescaled = scaler_y.inverse_transform(predictions)
    y_test_rescaled = scaler_y.inverse_transform(y_test)

    st.write("### Sample Predictions vs Actual Values:")
    for i in range(min(5, len(predictions_rescaled))):
        pred_open, pred_close = predictions_rescaled[i]
        actual_open, actual_close = y_test_rescaled[i]
        st.write(f"**Prediction** -> Open: {pred_open:.2f}, Close: {pred_close:.2f}")
        st.write(f"**Actual**     -> Open: {actual_open:.2f}, Close: {actual_close:.2f}")
        st.markdown("---")
    
    # Plot predicted vs. actual closing prices for visualization.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([x[1] for x in y_test_rescaled], color='blue', label='Actual Close')
    ax.plot([x[1] for x in predictions_rescaled], color='red', label='Predicted Close')
    ax.set_title(f"{ticker} - Actual vs Predicted Closing Prices")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
