# 📈 AI-Based Stock Market Prediction with News Sentiment Analysis

This project is an end-to-end AI-powered stock market prediction system that combines deep learning models with real-time news sentiment analysis to forecast stock price movements. It leverages financial time series data and NLP-based sentiment extraction from news articles to enhance prediction accuracy.

---

## 🚀 Features

- 🔮 **Predictive Models**: LSTM, GRU, Attention-based, and Hybrid neural networks for time-series forecasting.
- 📰 **News Sentiment Analysis**: Uses NLP models from Hugging Face Transformers to analyze the sentiment of recent news articles fetched via NewsAPI.
- 📊 **Technical Indicators**: Integration of moving averages, RSI, MACD, and more for feature enhancement.
- 💡 **User-Friendly Interface**: Built with Streamlit for an interactive, responsive front-end.
- 🔄 **Live Stock Prices**: Fetches real-time stock data using NSE API.
- 🧠 **Multimodal Learning**: Combines textual (news) and numerical (stock data) inputs for more robust predictions.

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: 
  - Deep Learning: TensorFlow, Keras
  - Data Handling: Pandas, NumPy
  - Visualization: Matplotlib, Plotly
  - Sentiment Analysis: Hugging Face Transformers
  - Web Framework: Streamlit
  - APIs: NewsAPI, yfinance/NSEpy

---

## 📂 Project Structure

 stock-predictor-app/ ├── data/ # Historical stock and news data ├── models/ # Trained LSTM, GRU, and hybrid models ├── sentiment/ # Scripts for news scraping and sentiment analysis ├── app.py # Streamlit frontend application ├── utils.py # Helper functions for preprocessing, visualization ├── requirements.txt # Python dependencies └── README.md # Project documentation.
