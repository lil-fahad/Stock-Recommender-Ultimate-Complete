
from fastapi import FastAPI
import joblib
import torch
import numpy as np
from models.lstm_model import LSTMModel

app = FastAPI()
model = LSTMModel(1, 32, 2, 1)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
scaler = joblib.load('scaler.pkl')

@app.post("/predict")
def predict(features: list):
    scaled = scaler.transform(np.array(features).reshape(-1, 1))
    input_tensor = torch.tensor(scaled.reshape(1, len(features), 1)).float()
    with torch.no_grad():
        pred_scaled = model(input_tensor).item()
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    return {"prediction": pred}
