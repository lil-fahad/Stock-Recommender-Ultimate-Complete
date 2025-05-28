from fastapi import FastAPI
import joblib
import torch
import numpy as np
from models.super_lstm import SuperLSTMModel
import xgboost as xgb

app = FastAPI()
lstm_model = SuperLSTMModel(1, 64, 3, 1)
lstm_model.load_state_dict(torch.load('best_lstm.pth', map_location=torch.device('cpu')))
lstm_model.eval()
xgb_model = joblib.load('best_xgb.pkl')
scaler = joblib.load('scaler.pkl')

@app.post("/predict")
def predict(features: list):
    scaled = scaler.transform(np.array(features).reshape(-1, 1))
    input_tensor = torch.tensor(scaled.reshape(1, len(features), 1)).float()
    with torch.no_grad():
        lstm_pred_scaled = lstm_model(input_tensor).item()
    lstm_pred = scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
    xgb_pred = xgb_model.predict(np.array(features).reshape(1, -1))[0]
    final_pred = (lstm_pred + xgb_pred) / 2
    return {"lstm_prediction": round(lstm_pred, 2), "xgb_prediction": round(xgb_pred, 2), "final_prediction": round(final_pred, 2)}