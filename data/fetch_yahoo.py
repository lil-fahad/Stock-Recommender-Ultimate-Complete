import yfinance as yf
import pandas as pd

def get_yahoo_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]