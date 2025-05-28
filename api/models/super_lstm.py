import torch
import torch.nn as nn

class SuperLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(SuperLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(attn_output[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out)