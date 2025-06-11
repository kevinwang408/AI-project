import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import predict  # Custom module (assumed to exist)

# ===============================
# Data Preprocessing
# ===============================

def load_data(file_path, look_back=5):
    dataset = pd.read_csv(file_path, engine='python', nrows=576 * 15)
    dataset.drop(columns=["Data", "Time"], inplace=True)
    dataset = dataset.values

    # Normalize all features
    scaler_all = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler_all.fit_transform(dataset)

    # Normalize the prediction target (2nd column)
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    target_col = dataset[:, 1].reshape(-1, 1)
    scaler_target.fit(target_col)

    # Split into train/test
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[:train_size, :], dataset[train_size:, :]

    def create_dataset(data, look_back):
        dataX, dataY = [], []
        for i in range(len(data) - look_back - 1):
            dataX.append(data[i:i + look_back, 0:9])
            dataY.append(data[i + look_back, 1])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    return trainX, trainY, testX, testY, scaler_target

# ===============================
# Dataloader Creator
# ===============================

def create_dataloader(trainX, trainY, testX, testY, batch_size=2):
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainY = torch.tensor(trainY, dtype=torch.float32).view(-1, 1)  # Reshape to [batch_size, 1]
    testX = torch.tensor(testX, dtype=torch.float32)
    testY = torch.tensor(testY, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(trainX, trainY)
    test_dataset = TensorDataset(testX, testY)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, testY

# ===============================
# Positional Encoding Module
# ===============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=1e-5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ===============================
# Transformer Model
# ===============================

class TransformerModel(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=1, dropout=1e-5):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = self.encoder(x)  # shape: [batch, seq_len, d_model]
        x = x.transpose(0, 1)  # shape: [seq_len, batch, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[-1])  # Use output from last time step
        return x

# ===============================
# Training Loop
# ===============================

def train_model(model, train_loader, test_loader, testY, scaler_target, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    best_MAE = float('inf')
    best_predictions = []

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        predictions = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())
                predictions.extend(output.squeeze().tolist())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        pred_rescaled = scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1))
        true_rescaled = scaler_target.inverse_transform(testY.numpy())

        rmse = math.sqrt(mean_squared_error(true_rescaled, pred_rescaled))
        mae = mean_absolute_error(true_rescaled, pred_rescaled)

        if mae < best_MAE:
            best_MAE = mae
            best_predictions = pred_rescaled
            print("Best model updated.")

        print(f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | Best MAE: {best_MAE:.2f}")

    return best_predictions, scaler_target.inverse_transform(testY.numpy())

# ===============================
# Main Execution
# ===============================

if __name__ == '__main__':
    LOOK_BACK = 5
    FILE_PATH = 'SolarRadiationPrediction.csv'

    trainX, trainY, testX, testY, scaler_target = load_data(FILE_PATH, look_back=LOOK_BACK)
    train_loader, test_loader, testY_tensor = create_dataloader(trainX, trainY, testX, testY)

    model = TransformerModel()

    best_preds, true_vals = train_model(model, train_loader, test_loader, testY_tensor, scaler_target)

    # Post-processing
    predict.my_self(best_preds, true_vals, 'Transformer')
    predict.score_calculation(best_preds, true_vals)
    predict.plot_pred(best_preds, true_vals, 'Transformer')
    predict.plot_residuals(best_preds, true_vals, 'Transformer')
