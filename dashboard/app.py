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