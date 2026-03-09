import torch
import torch.nn as nn


class LSTMSepsisModel(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)

        logits = self.fc(last_hidden)

        return logits.squeeze(1)