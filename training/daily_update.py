
from data.fetch_polygon import get_polygon_data
from training.trainer import train_lstm
import torch
import joblib

def update_lstm_model(symbol, start, end):
    df = get_polygon_data(symbol, start, end)
    values = df['Close'].values.reshape(-1, 1)
    scaler = joblib.load('scaler.pkl')
    scaled_values = scaler.transform(values)

    X = []
    y = []
    seq_length = 20
    for i in range(len(scaled_values) - seq_length):
        X.append(scaled_values[i:i + seq_length])
        y.append(scaled_values[i + seq_length])

    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()

    model = train_lstm(X_tensor, y_tensor, 1, 32, 2, 1, epochs=5)
    torch.save(model.state_dict(), 'best_model.pth')
