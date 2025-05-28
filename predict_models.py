
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def predict_lstm(df, look_back=60):
    scaler = joblib.load('lstm_scaler.pkl')
    model = load_model('lstm_model.h5')
    scaled_data = scaler.transform(df['Close'].values.reshape(-1,1))
    X = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

def predict_prophet(future_periods=30):
    model = joblib.load('prophet_model.pkl')
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def predict_ensemble(df):
    rf = joblib.load('random_forest_model.pkl')
    gb = joblib.load('gradient_boosting_model.pkl')
    X = df.drop(columns=['Close'])
    rf_pred = rf.predict(X)
    gb_pred = gb.predict(X)
    combined = (rf_pred + gb_pred) / 2
    return combined

if __name__ == '__main__':
    df = pd.read_csv('your_data.csv')  # replace with your data
    lstm_preds = predict_lstm(df)
    print('✅ LSTM Predictions:', lstm_preds[-5:])
    prophet_preds = predict_prophet()
    print('✅ Prophet Predictions:', prophet_preds.tail())
    ensemble_preds = predict_ensemble(df)
    print('✅ Ensemble Predictions:', ensemble_preds[-5:])
