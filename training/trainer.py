import torch
import torch.optim as optim
import torch.nn as nn
from models.super_lstm import SuperLSTMModel
import joblib
import numpy as np

def train_lstm(X_train, y_train, input_dim=1, hidden_dim=64, num_layers=3, output_dim=1, epochs=20, lr=0.001):
    model = SuperLSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), 'best_lstm.pth')

def train_xgb(X_train, y_train):
    import xgboost as xgb
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, 'best_xgb.pkl')