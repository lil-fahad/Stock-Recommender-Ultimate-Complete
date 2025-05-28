
import requests
import pandas as pd

API_KEY = 'vyje3A_Mvpeywm7ROMWQdrspgH2KIOBu'

def get_polygon_data(symbol, start, end):
    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    results = data.get('results', [])
    if not results:
        raise ValueError(f"No data for {symbol}")
    df = pd.DataFrame(results)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'c': 'Close', 'h': 'High', 'l': 'Low', 'o': 'Open', 'v': 'Volume'}, inplace=True)
    df.set_index('t', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
