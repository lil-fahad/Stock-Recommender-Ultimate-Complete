
import torch
import torch.optim as optim
import torch.nn as nn
from models.lstm_model import LSTMModel

def train_lstm(X_train, y_train, input_dim, hidden_dim, num_layers, output_dim, epochs=10, lr=0.001):
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    return model
