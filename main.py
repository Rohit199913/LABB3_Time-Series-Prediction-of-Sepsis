import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_all_data
from src.data.windowing import create_windows
from src.models.lstm_model import LSTMSepsisModel
from src.training.train import train_one_epoch, evaluate


print("Loading datasets...")

datasets = load_all_data("data/raw")

df_train = pd.concat([datasets[0], datasets[1]], ignore_index=True)
df_val = datasets[2]
df_test = datasets[3]


print("Creating windows...")

X_train, y_train, feature_columns = create_windows(df_train, window_size=12, horizon_steps=4)
X_val, y_val, _ = create_windows(df_val, window_size=12, horizon_steps=4)
X_test, y_test, _ = create_windows(df_test, window_size=12, horizon_steps=4)


print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)


train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMSepsisModel(input_size=X_train.shape[2]).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5


for epoch in range(epochs):

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    val_loss, val_probs, val_targets = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")