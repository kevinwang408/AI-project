import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import predict

# Load and preprocess dataset
def load_data(path='SolarRadiationPrediction.csv', look_back=5):
    dataset = pd.read_csv(path, engine='python', nrows=576 * 15)
    dataset = dataset.drop(columns=["Data", "Time"])
    dataset = dataset.values.astype('float32')

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    target_col = dataset[:, 1].reshape(-1, 1)
    dataset_scaled = scaler_all.fit_transform(dataset)
    target_scaled = scaler_target.fit_transform(target_col)

    train_size = int(len(dataset_scaled) * 0.67)
    train = dataset_scaled[:train_size]
    test = dataset_scaled[train_size:]

    def create_dataset(data):
        dataX, dataY = [], []
        for i in range(len(data) - look_back - 1):
            dataX.append(data[i:i + look_back, :9])
            dataY.append(data[i + look_back, 1])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train)
    testX, testY = create_dataset(test)

    return (torch.tensor(trainX, dtype=torch.float32),
            torch.tensor(trainY, dtype=torch.float32).unsqueeze(1),  # fix target shape
            torch.tensor(testX, dtype=torch.float32),
            torch.tensor(testY, dtype=torch.float32).unsqueeze(1),   # fix target shape
            scaler_target)

# Positional encoding for time-series data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=1e-5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformer model for time-series regression
class TransformerModel(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=1, dropout=1e-5):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)                         # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)                     # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)             # (batch_size, seq_len, d_model)
        x = self.decoder(x[:, -1, :])               # (batch_size, 1)
        return x

# Training routine
def train_model(model, train_loader, test_loader, scaler_target, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    best_MAE = float('inf')
    best_predictions = None

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses, predictions = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                predictions.extend(outputs.squeeze().tolist())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # Evaluate on inverse transformed scale
        preds_inv = scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1))
        labels_inv = scaler_target.inverse_transform(testY.numpy())
        rmse = math.sqrt(mean_squared_error(labels_inv, preds_inv))
        mae = mean_absolute_error(labels_inv, preds_inv)

        if mae < best_MAE:
            best_MAE = mae
            best_predictions = preds_inv
            print("New best MAE")

        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.2f}")

    return best_predictions, labels_inv

# Load data and prepare loaders
look_back = 5
trainX, trainY, testX, testY, scaler_target = load_data(look_back=look_back)
train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=2, shuffle=True)
test_loader = DataLoader(TensorDataset(testX, testY), batch_size=2, shuffle=False)

# Model init and training
model = TransformerModel()
best_preds, true_vals = train_model(model, train_loader, test_loader, scaler_target)

# Visualization and metrics
predict.my_self(best_preds, true_vals, 'Transformer')
predict.score_calculation(best_preds, true_vals)
predict.plot_pred(best_preds, true_vals, 'Transformer')
predict.plot_residuals(best_preds, true_vals, 'Transformer')