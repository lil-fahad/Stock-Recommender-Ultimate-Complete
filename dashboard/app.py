import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import requests
from data.fetch_polygon import get_polygon_data

st.title("📈 Ultimate Stock Recommender")

symbol = st.text_input("Stock Symbol", "AAPL")
if st.button("Load Data"):
    df = get_polygon_data(symbol, '2023-01-01', '2023-12-31')
    st.line_chart(df['Close'])

if st.button("Get Recommendation"):
    df = get_polygon_data(symbol, '2023-01-01', '2023-12-31')
    last_20 = df['Close'][-20:].tolist()
    response = requests.post("https://my-fastapi-backend.onrender.com/predict", json={"features": last_20})
    if response.status_code == 200:
        pred = response.json()['prediction']
        st.success(f"Predicted Next Close Price: ${pred:.2f}")
    else:
        st.error("Prediction API error")