
import pandas as pd
from prophet import Prophet
import joblib

def train_prophet(df, date_col='Date', target_col='Close'):
    df_prophet = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    joblib.dump(model, 'prophet_model.pkl')
    print('✅ Prophet model saved.')

if __name__ == '__main__':
    df = pd.read_csv('your_data.csv')  # replace with your data
    train_prophet(df)
