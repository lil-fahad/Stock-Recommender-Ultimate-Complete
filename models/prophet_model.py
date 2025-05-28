
import pandas as pd
from prophet import Prophet

def train_prophet(df, date_col='Date', target_col='Close'):
    df_prophet = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    return model

def predict_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
