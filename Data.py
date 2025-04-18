import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline
import ta
from datetime import datetime
import requests  # For NewsAPI

# Set page configuration and add header with image
st.set_page_config(page_title="Live NSE Stock Prices", layout="wide")
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/NSE_Logo.svg/2560px-NSE_Logo.svg.png", width=100)
with col2:
    st.title("NSE Stock Market Prediction")
    st.markdown("#### 🎯 AI-Powered Stock Price Predictions & Analysis")

# Define helper functions
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    today = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker_symbol, start="2015-01-01", end=today)
    if data.empty:
        return None
    return data

def create_sequences(X_array, y_array, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X_array) - time_steps):
        X_seq.append(X_array[i:i+time_steps])
        y_seq.append(y_array[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

# Define NSE stocks list
nse_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            stock_data.append({
                "Stock": ticker.replace(".NS", ""),
                "Price": round(hist["Close"].iloc[-1], 2),
                "High": round(hist["High"].iloc[-1], 2),
                "Low": round(hist["Low"].iloc[-1], 2),
                "Volume": hist["Volume"].iloc[-1],
                "Time": datetime.now().strftime("%H:%M:%S")
            })
    return pd.DataFrame(stock_data)

# Add tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Prices", "📈 Analysis", "🤖 Predictions", "📰 News"])

with tab1:
    st.subheader("Live Market Overview")
    stock_df = fetch_stock_data(nse_stocks)
    
    if not stock_df.empty:
        metrics_cols = st.columns(len(stock_df))
        for idx, (_, stock) in enumerate(stock_df.iterrows()):
            with metrics_cols[idx]:
                st.metric(
                    label=stock["Stock"],
                    value=f"₹{stock['Price']}",
                    delta=f"₹{stock['High'] - stock['Low']:.2f}"
                )
    else:
        st.error("⚠️ Unable to fetch live market data")

with tab2:
    st.subheader("Technical Analysis")
    # Add time period selector
    analysis_period = st.select_slider(
        "Select Analysis Period",
        options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
        value="6M"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### RSI Indicator")
        # Add RSI visualization
    with col2:
        st.write("### MACD Indicator")
        # Add MACD visualization

with tab3:
    st.subheader("AI Predictions")
    
    with st.expander("🔧 Model Configuration", expanded=True):
        pred_ticker = st.selectbox("Select Stock for Prediction:", nse_stocks, key="pred_stock")
        
        # Model architecture selection
        model_type = st.selectbox(
            "Select Model Architecture",
            ["LSTM", "GRU", "LSTM with Attention", "Hybrid (LSTM+GRU)"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Training Epochs", 10, 200, 50)
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
            patience = st.slider("Early Stopping Patience", 5, 20, 10)
        with col2:
            units = st.slider("Hidden Units", 50, 400, 200)
            num_layers = st.slider("Number of Layers", 1, 4, 2)
            dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2)

    if st.button("🚀 Train Model and Predict"):
        try:
            with st.spinner("Loading and preparing data..."):
                data = load_data(pred_ticker)
                if data is not None:
                    # Data preparation
                    close_series = data['Close'].squeeze()
                    data['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
                    macd = ta.trend.MACD(close=close_series)
                    data['MACD'] = macd.macd()
                    data['MACD_Signal'] = macd.macd_signal()
                    data.dropna(inplace=True)

                    features = data[["Open", "Close", "RSI", "MACD", "MACD_Signal"]]
                    target = data[["Open", "Close"]]
                    scaler_X, scaler_y = MinMaxScaler((0, 1)), MinMaxScaler((0, 1))
                    scaled_X, scaled_y = scaler_X.fit_transform(features), scaler_y.fit_transform(target)
                    
                    time_steps = 60
                    X_seq, y_seq = create_sequences(scaled_X, scaled_y, time_steps)
                    split_index = int(0.8 * len(X_seq))
                    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
                    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

                    # Model architecture selection function
                    def create_model(model_type, input_shape, units, num_layers, dropout_rate):
                        model = Sequential()
                        
                        if model_type == "LSTM":
                            model.add(LSTM(units, return_sequences=True if num_layers > 1 else False, 
                                         input_shape=input_shape))
                            model.add(Dropout(dropout_rate))
                            
                            for i in range(num_layers - 1):
                                if i == num_layers - 2:
                                    model.add(LSTM(units // 2))
                                else:
                                    model.add(LSTM(units // 2, return_sequences=True))
                                model.add(Dropout(dropout_rate))
                                
                        elif model_type == "GRU":
                            from tensorflow.keras.layers import GRU
                            model.add(GRU(units, return_sequences=True if num_layers > 1 else False, 
                                        input_shape=input_shape))
                            model.add(Dropout(dropout_rate))
                            
                            for i in range(num_layers - 1):
                                if i == num_layers - 2:
                                    model.add(GRU(units // 2))
                                else:
                                    model.add(GRU(units // 2, return_sequences=True))
                                model.add(Dropout(dropout_rate))
                                
                        elif model_type == "LSTM with Attention":
                            from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
                            model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
                            model.add(MultiHeadAttention(num_heads=2, key_dim=units))
                            model.add(LayerNormalization())
                            model.add(Dropout(dropout_rate))
                            model.add(LSTM(units // 2))
                            
                        else:  # Hybrid
                            model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
                            model.add(Dropout(dropout_rate))
                            model.add(GRU(units // 2))
                            
                        model.add(Dense(25, activation='relu'))
                        model.add(Dense(2))
                        return model

                    # Create and compile model
                    model = create_model(model_type, (time_steps, X_train.shape[2]), 
                                      units, num_layers, dropout_rate)
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    # Add early stopping
                    early_stopping = EarlyStopping(
                        monitor='loss',
                        patience=patience,
                        restore_best_weights=True
                    )

                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    st.success("✅ Training complete!")

                    predictions = model.predict(X_test)
                    predictions_rescaled = scaler_y.inverse_transform(predictions)
                    y_test_rescaled = scaler_y.inverse_transform(y_test)

                    # Calculate accuracy metrics
                    def calculate_accuracy(y_true, y_pred, threshold=0.05):
                        correct_predictions = 0
                        total_predictions = len(y_true)
                        
                        for i in range(total_predictions):
                            # Calculate percentage error
                            true_close = y_true[i][1]
                            pred_close = y_pred[i][1]
                            error_percentage = abs((true_close - pred_close) / true_close)
                            
                            if error_percentage <= threshold:
                                correct_predictions += 1
                                
                        return (correct_predictions / total_predictions) * 100

                    accuracy = calculate_accuracy(y_test_rescaled, predictions_rescaled)
                    
                    # Display metrics in columns
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Model Accuracy", f"{accuracy:.2f}%")
                    with metrics_cols[1]:
                        st.metric("MSE", f"{mean_squared_error(y_test_rescaled, predictions_rescaled):.4f}")
                    with metrics_cols[2]:
                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled)):.4f}")
                    with metrics_cols[3]:
                        st.metric("MAE", f"{mean_absolute_error(y_test_rescaled, predictions_rescaled):.4f}")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot([x[1] for x in y_test_rescaled], color='blue', label='Actual Close')
                    ax.plot([x[1] for x in predictions_rescaled], color='red', label='Predicted Close')
                    ax.fill_between(range(len(y_test_rescaled)),
                                  [x[1] for x in y_test_rescaled],
                                  [x[1] for x in predictions_rescaled],
                                  alpha=0.2, color='gray')
                    ax.set_title(f"{pred_ticker} - Actual vs Predicted Closing Prices")
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Price (₹)")
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.info("Please try again with different parameters or select another stock.")

with tab4:
    st.subheader("Market News & Sentiment")
    # Add news filters
    news_filters = st.multiselect(
        "Filter News By:",
        ["Positive Sentiment", "Negative Sentiment", "High Impact", "Company News", "Market News"],
        ["Company News"]
    )
    # Your existing news sentiment code

# Add a footer
st.markdown("""
    <div style='text-align: center; color: grey; padding: 20px;'>
        <p>Powered by AI & Machine Learning</p>
        <p>Data Source: Yahoo Finance | News: NewsAPI</p>
    </div>
""", unsafe_allow_html=True)

# Enhanced styling
st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 10px 20px;
            background-color: #2E2E2E;
            border-radius: 5px;
        }
        .stMetric {
            background-color: #2E2E2E;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #3E3E3E;
        }
        .stExpander {
            background-color: #2E2E2E;
            border-radius: 10px;
            border: 1px solid #3E3E3E;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
    </style>
""", unsafe_allow_html=True)

# Allow user to search for real-time stock prices
user_ticker = st.text_input("Enter any NSE stock ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")

@st.cache_data(ttl=3600)
def fetch_single_stock(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y")
    
    if hist.empty:
        return None  # Return None if no data is found
    
    return {
        "Stock": ticker.replace(".NS", ""),
        "Price": round(hist["Close"].iloc[-1], 2),
        "High": round(hist["High"].iloc[-1], 2),
        "Low": round(hist["Low"].iloc[-1], 2),
        "Volume": hist["Volume"].iloc[-1],
        "Time Updated": datetime.now().strftime("%H:%M:%S")
    }

# Fetch stock price for the user-entered stock
stock_data = fetch_single_stock(user_ticker)

if stock_data:
    st.write("### 📈 Real-Time Stock Price")
    st.write(f"**Stock:** {stock_data['Stock']}")
    st.write(f"**Price:** ₹{stock_data['Price']}")
    st.write(f"**High:** ₹{stock_data['High']}")
    st.write(f"**Low:** ₹{stock_data['Low']}")
    st.write(f"**Volume:** {stock_data['Volume']}")
    st.write(f"**Last Updated:** {stock_data['Time Updated']}")
else:
    st.error("⚠️ No data found. Please check the stock ticker and try again.")

# 1. Data Acquisition: User inputs the NSE stock ticker
ticker = st.text_input("Enter the name of NSE Stock(with .NS appended, e.g., RELIANCE.NS):", "RELIANCE.NS")

data = load_data(ticker)
if data is None or data.empty:
    st.error("No data found for the ticker provided. Please check the ticker symbol and try again.")
    st.stop()

st.write("### Historical Data")
st.dataframe(data.tail())

# 2. Feature Engineering: Add Technical Indicators (RSI, MACD, MACD Signal)
close_series = data['Close'].squeeze()
data['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
macd = ta.trend.MACD(close=close_series)
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()
data.dropna(inplace=True)

st.write("### Technical Indicators ")
st.dataframe(data[['Open', 'Close', 'RSI', 'MACD', 'MACD_Signal']].tail())

# 3. Prepare Features and Target
features = data[["Open", "Close", "RSI", "MACD", "MACD_Signal"]]
target = data[["Open", "Close"]]

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_X = scaler_X.fit_transform(features)
scaled_y = scaler_y.fit_transform(target)

time_steps = 60  # Number of past days used for prediction
X_seq, y_seq = create_sequences(scaled_X, scaled_y, time_steps)

# 4. News Sentiment Analysis Feature
st.write("### 📢 Latest News & Sentiment Analysis")

@st.cache_data(ttl=600)  # Cache news for 10 minutes
def fetch_latest_news(query):
    api_key = "76ea7e3a63974d5886126c5bbb034430"  # Your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&sortBy=publishedAt&pageSize=5"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            if news_data["totalResults"] > 0:
                articles = news_data["articles"]
                latest_news = []
                for article in articles:
                    latest_news.append({
                        "Title": article.get("title", "").strip(),
                        "URL": article.get("url", "#"),
                        "Publisher": article.get("source", {}).get("name", "Unknown"),
                        "Published Time": article.get("publishedAt", "N/A")
                    })
                return latest_news
            else:
                return None  # No news found
        else:
            st.error(f"⚠️ Error fetching news: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")
        return None

if ticker:
    stock_name = ticker.replace(".NS", "")
    news_data = fetch_latest_news(stock_name)
else:
    st.error("⚠️ No stock selected. Please enter a valid stock ticker.")
    st.stop()

if news_data:
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        for article in news_data:
            try:
                if not article["Title"]:
                    st.warning("⚠️ Skipping sentiment analysis for an article with no title.")
                    continue
                
                sentiment = sentiment_pipeline(article["Title"])[0]
                st.write(f"📰 **[{article['Title']}]({article['URL']})**")
                st.write(f"**Publisher:** {article['Publisher']}")
                st.write(f"**Published Time:** {article['Published Time']}")
                st.write(f"📊 Sentiment: **{sentiment['label']}** (Confidence: {sentiment['score']:.2f})")
                st.markdown("---")
            except Exception as e:
                st.error(f"⚠️ Error analyzing sentiment for: {article['Title']}")
    except Exception as e:
        st.error(f"⚠️ Error initializing sentiment analysis: {str(e)}")
        for article in news_data:
            st.write(f"📰 **[{article['Title']}]({article['URL']})**")
            st.write(f"**Publisher:** {article['Publisher']}")
            st.write(f"**Published Time:** {article['Published Time']}")
            st.markdown("---")

elif news_data is None:
    st.warning(f"⚠️ No recent news found for {stock_name}. Please try another stock or check back later.")
else:
    st.error(f"⚠️ Unable to fetch news for {stock_name}. Please ensure the stock name is correct.")

# Split the data into training and testing sets (80% train, 20% test)
split_index = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

st.write("Training data shape:", X_train.shape)
st.write("Testing data shape:", X_test.shape)

# 5. Build the LSTM Model: Multi-output regression (predicts both Open and Close)
model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mean_squared_error')

if st.button("Train Model and Predict", key="final_predict"):
    with st.spinner("Training model..."):
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    st.success("Training complete!")

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
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([x[1] for x in y_test_rescaled], color='blue', label='Actual Close')
    ax.plot([x[1] for x in predictions_rescaled], color='red', label='Predicted Close')
    ax.set_title(f"{ticker} - Actual vs Predicted Closing Prices")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
