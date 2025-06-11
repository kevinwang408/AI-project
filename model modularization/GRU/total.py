import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import predict  # Custom utility script for plotting and evaluation

# ========================
# Load and preprocess data
# ========================
dataset = pd.read_csv('SolarRadiationPrediction.csv', engine='python', nrows=576 * 15)
dataset = dataset.drop(["Data", "Time"], axis=1)

# Convert DataFrame to NumPy array
raw_values = dataset.values.astype('float32')

# Extract the target feature to normalize separately (e.g., Radiation)
target_column = raw_values[:, 1].reshape(-1, 1)

# Apply normalization (Min-Max Scaling)
scaler_all = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

scaled_dataset = scaler_all.fit_transform(raw_values)
scaled_target = scaler_target.fit_transform(target_column)

# ============================
# Split dataset into training and testing sets
# ============================
train_size = int(len(scaled_dataset) * 0.67)
test_size = len(scaled_dataset) - train_size
train_data, test_data = scaled_dataset[0:train_size, :], scaled_dataset[train_size:, :]

# =================================
# Prepare sequences for GRU training
# =================================
def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        sequence = dataset[i:(i + look_back), 0:9]  # Select first 9 features
        label = dataset[i + look_back, 1]           # Predict the second column (target)
        dataX.append(sequence)
        dataY.append(label)
    return np.array(dataX), np.array(dataY)

look_back = 5
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

# ==============
# Build GRU model
# ==============
model = Sequential()
model.add(GRU(units=193, activation='tanh', input_shape=(look_back, 9)))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(rate=0.082379383823238))
model.add(Dense(units=1, activation="linear"))  # Output layer for regression
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# ================
# Train the model
# ================
model.fit(trainX, trainY, epochs=20, batch_size=16, verbose=2)

# ========================
# Make predictions
# ========================
trainPre = model.predict(trainX)
testPre = model.predict(testX)

# Inverse transform predictions and ground truth
trainPre = scaler_target.inverse_transform(trainPre)
trainY = scaler_target.inverse_transform([trainY])
testPre = scaler_target.inverse_transform(testPre)
testY = scaler_target.inverse_transform([testY])

# ========================
# Evaluate performance
# ========================
train_rmse = math.sqrt(mean_squared_error(trainY[0, :], trainPre[:, 0]))
test_rmse = math.sqrt(mean_squared_error(testY[0, :], testPre[:, 0]))
test_mae = mean_absolute_error(testY[0, :], testPre[:, 0])

print(f'Train Score: {train_rmse:.2f} RMSE')
print(f'Test Score: {test_rmse:.2f} RMSE')
print(f'Test Score: {test_mae:.2f} MAE')

# ========================
# Plotting and Evaluation
# ========================
ground_truth = scaler_target.inverse_transform(scaled_target)
predict.my_self(testY[0, :], testPre[:, 0], 'GRU')
predict.score_calculation(testY[0, :], testPre[:, 0])
predict.plot_pred(testY[0, :], testPre[:, 0], 'GRU')
predict.plot_residuals(testY[0, :], testPre[:, 0], 'GRU')
