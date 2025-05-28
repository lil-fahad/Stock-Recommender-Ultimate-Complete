import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import requests
import plotly.express as px
from data.fetch_yahoo import get_yahoo_data
from data.feature_engineering import add_indicators

st.set_page_config(page_title="🔥 Stock Recommender God Mode", layout="wide")
st.title("📈 Ultimate Stock Recommender (God Mode Edition)")

symbol = st.text_input("Stock Symbol", "AAPL")
if st.button("Load Data"):
    df = get_yahoo_data(symbol, '2023-01-01', '2023-12-31')
    df = add_indicators(df)
    fig = px.line(df, x=df.index, y=['Close', 'MA20', 'MA50', 'UpperBand', 'LowerBand'], title=f'{symbol} with Indicators')
    st.plotly_chart(fig, use_container_width=True)

if st.button("Get Recommendation"):
    df = get_yahoo_data(symbol, '2023-01-01', '2023-12-31')
    last_60 = df['Close'][-60:].tolist()
    response = requests.post("http://api:8000/predict", json={"features": last_60})
    if response.status_code == 200:
        result = response.json()
        st.success(f"🔥 Final Prediction: ${result['final_prediction']}")
        st.info(f"LSTM: ${result['lstm_prediction']} | XGBoost: ${result['xgb_prediction']}")
        last_close = df['Close'].iloc[-1]
        if result['final_prediction'] > last_close * 1.02:
            st.info("📈 Recommendation: STRONG BUY")
        elif result['final_prediction'] > last_close:
            st.info("✅ Recommendation: Buy")
        else:
            st.info("⚠️ Recommendation: Hold/Sell")
    else:
        st.error("❌ Prediction API error")
# --- Advanced Models Integration ---
from models.advanced_lstm import build_lstm_model, prepare_lstm_data
from models.prophet_model import train_prophet, predict_prophet
from data.feature_engineering import add_technical_indicators

st.write("### Enhanced AI Stock Predictions")

# Example usage
try:
    df = get_yahoo_data('AAPL')
    df = add_technical_indicators(df)

    # LSTM Example
    X, y, scaler = prepare_lstm_data(df)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    lstm_model = build_lstm_model((X.shape[1], 1))
    st.write("LSTM model ready for training (sample only)")

    # Prophet Example
    prophet_model = train_prophet(df)
    forecast = predict_prophet(prophet_model)
    st.write("Prophet forecast sample:")
    st.dataframe(forecast.tail())
except Exception as e:
    st.error(f"Model integration error: {e}")

import plotly.express as px
if 'forecast' in locals():
    fig = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], title='Prophet Forecast')
    st.plotly_chart(fig)

# --- Ensemble Models Integration ---
from models.ensemble_model import train_ensemble, predict_ensemble

try:
    df = get_yahoo_data('AAPL')
    df = add_advanced_indicators(df)
    X = df.drop(columns=['Close'])
    models = train_ensemble(df)
    ensemble_pred = predict_ensemble(models, X)
    st.write("Ensemble model sample predictions:")
    st.write(ensemble_pred[-5:])  # show last 5 predictions
except Exception as e:
    st.error(f"Ensemble model integration error: {e}")

import streamlit as st

st.sidebar.title("📊 Stock Recommender Dashboard")
page = st.sidebar.selectbox("Select Page", ["Home", "LSTM Model", "Prophet Model", "Ensemble Model"])

if page == "Home":
    st.write("## Welcome to the Enhanced Stock Recommender App")
    st.write("Use the sidebar to navigate between models and view predictions.")
elif page == "LSTM Model":
    st.write("## LSTM Model Predictions")
    # (Insert LSTM visualization here)
elif page == "Prophet Model":
    st.write("## Prophet Model Forecast")
    # (Insert Prophet visualization here)
elif page == "Ensemble Model":
    st.write("## Ensemble Model Predictions")
    # (Insert Ensemble visualization here)

import plotly.express as px

if page == "LSTM Model":
    st.write("## LSTM Model Predictions")
    # Generate dummy data for illustration
    actual = [i + (i * 0.1) for i in range(100)]
    predicted = [i + (i * 0.1) + (i * 0.05) for i in range(100)]
    fig = px.line(x=list(range(100)), y=[actual, predicted], labels={'x': 'Time', 'value': 'Price'}, title='LSTM Actual vs Predicted')
    fig.update_traces(name='Actual', selector=dict(name='wide_variable_0'))
    fig.update_traces(name='Predicted', selector=dict(name='wide_variable_1'))
    st.plotly_chart(fig)

elif page == "Prophet Model":
    st.write("## Prophet Model Forecast")
    actual = [i + (i * 0.2) for i in range(100)]
    forecast = [i + (i * 0.2) + (i * 0.1) for i in range(100)]
    fig = px.line(x=list(range(100)), y=[actual, forecast], labels={'x': 'Time', 'value': 'Price'}, title='Prophet Actual vs Forecast')
    fig.update_traces(name='Actual', selector=dict(name='wide_variable_0'))
    fig.update_traces(name='Forecast', selector=dict(name='wide_variable_1'))
    st.plotly_chart(fig)

elif page == "Ensemble Model":
    st.write("## Ensemble Model Predictions")
    actual = [i + (i * 0.15) for i in range(100)]
    ensemble_pred = [i + (i * 0.15) + (i * 0.07) for i in range(100)]
    fig = px.line(x=list(range(100)), y=[actual, ensemble_pred], labels={'x': 'Time', 'value': 'Price'}, title='Ensemble Actual vs Predicted')
    fig.update_traces(name='Actual', selector=dict(name='wide_variable_0'))
    fig.update_traces(name='Predicted', selector=dict(name='wide_variable_1'))
    st.plotly_chart(fig)
