import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN, tcn_full_summary
import predict

# ========== Parameters ==========
BATCH_SIZE = 32
TIME_STEPS = 5
INPUT_DIM = 9
EPOCHS = 150
TRAIN_RATIO = 0.67

# ========== Data Preprocessing ==========
def load_and_preprocess_data(filepath, nrows=None):
    """Load CSV and normalize dataset."""
    df = pd.read_csv(filepath, engine='python', nrows=nrows)
    df = df.drop(["Data", "Time"], axis=1)
    data = df.values.astype('float32')

    # Preserve original target (Radiation)
    target_column = data[:, 1].reshape(-1, 1)

    scaler_all = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler_all.fit_transform(data)
    target_scaled = scaler_target.fit_transform(target_column)

    return data_scaled, target_scaled, scaler_all, scaler_target

def split_data(data, train_ratio):
    """Split dataset into train and test sets."""
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def create_lookback_dataset(data, look_back=TIME_STEPS):
    """Create sequences of look_back time steps as model input."""
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:i + look_back, :INPUT_DIM])
        Y.append(data[i + look_back, 1])  # Radiation (target)
    return np.array(X), np.array(Y)

# ========== Model Definition ==========
def build_tcn_model(input_shape):
    """Build a TCN regression model using keras-tcn."""
    tcn_layer = TCN(
        nb_filters=110,
        kernel_size=10,
        dropout_rate=0.4331,
        activation='relu',
        padding='causal',
        nb_stacks=1,
        dilations=(1, 2, 4, 8, 16, 32),
        input_shape=input_shape
    )
    print('Receptive field size =', tcn_layer.receptive_field)
    model = Sequential([
        tcn_layer,
        Dense(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    tcn_full_summary(model, expand_residual_blocks=False)
    return model

# ========== Training Pipeline ==========
def train_and_evaluate_model(trainX, trainY, testX, testY, scaler_target):
    """Train the model and evaluate with RMSE and MAE."""
    model = build_tcn_model((TIME_STEPS, INPUT_DIM))
    model.fit(trainX, trainY, epochs=EPOCHS, validation_split=0.2)

    # Predict
    train_pred = model.predict(trainX)
    test_pred = model.predict(testX)

    # Inverse scaling
    train_pred = scaler_target.inverse_transform(train_pred)
    trainY = scaler_target.inverse_transform([trainY])
    test_pred = scaler_target.inverse_transform(test_pred)
    testY = scaler_target.inverse_transform([testY])

    # Metrics
    test_rmse = math.sqrt(mean_squared_error(testY[0], test_pred[:, 0]))
    test_mae = mean_absolute_error(testY[0], test_pred[:, 0])
    print(f"Test Score: {test_rmse:.2f} RMSE")
    print(f"Test Score: {test_mae:.2f} MAE")

    # Visualization & Metrics
    predict.my_self(testY[0], test_pred[:, 0], 'TCN_2')
    predict.score_calculation(testY[0], test_pred[:, 0])
    predict.plot_pred(testY[0], test_pred[:, 0], 'TCN_2')
    predict.plot_residuals(testY[0], test_pred[:, 0], 'TCN_2')

# ========== Main ==========
if __name__ == "__main__":
    filepath = 'SolarRadiationPrediction.csv'
    raw_data, target_scaled, scaler_all, scaler_target = load_and_preprocess_data(filepath, nrows=576 * 15)
    train, test = split_data(raw_data, TRAIN_RATIO)
    trainX, trainY = create_lookback_dataset(train)
    testX, testY = create_lookback_dataset(test)

    train_and_evaluate_model(trainX, trainY, testX, testY, scaler_target)
