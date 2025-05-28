
import pandas as pd
import ta  # Technical Analysis library

def add_advanced_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['bollinger_h'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['bollinger_l'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['stochastic'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['volume_osc'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    return df
