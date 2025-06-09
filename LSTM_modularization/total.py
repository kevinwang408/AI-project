import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import predict  # Custom module with evaluation and plotting functions

# Set seed for reproducibility
np.random.seed(7)

# ====================
# 1. Data Preparation
# ====================

# Load the dataset and remove unnecessary columns
dataset = pd.read_csv('SolarRadiationPrediction.csv', engine='python', nrows=576*15)
dataset = dataset.drop(["Data", "Time"], axis=1)

# Convert to NumPy array and float32
dataset = dataset.values.astype('float32')

# Normalize features and target separately
scalar1 = MinMaxScaler(feature_range=(0, 1))  # For all features
scalar2 = MinMaxScaler(feature_range=(0, 1))  # For the radiation target

# Keep a separate copy of the radiation column (target) for inverse transform
scalar_dim = dataset[:, 1].reshape(-1, 1)

# Fit-transform the entire dataset
dataset = scalar1.fit_transform(dataset)
scalar_dim = scalar2.fit_transform(scalar_dim)

# Split dataset into training and testing
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[:train_size, :], dataset[train_size:, :]

# =========================
# 2. Create Time Series Data
# =========================
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0:9]  # use first 9 features as input
        dataX.append(a)
        dataY.append(dataset[i + look_back, 1])  # target is column index 1 (radiation)
    return np.array(dataX), np.array(dataY)

look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# =====================
# 3. Build LSTM Network
# =====================
model = Sequential()
model.add(LSTM(105, activation='relu', input_shape=(look_back, 9)))
model.add(Dropout(0.3618106763168733))
model.add(Dense(1))  # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# =====================
# 4. Training Loop with Visualization
# =====================
loss_arr = []
epochs = 100
for i in range(epochs):
    history = model.fit(trainX, trainY, epochs=1, batch_size=3, verbose=2)
    loss_arr.append(history.history['loss'][0])

    # Prediction
    trainPre = model.predict(trainX)
    testPre = model.predict(testX)

    # Inverse transform predictions and labels
    trainPre = scalar2.inverse_transform(trainPre)
    trainY_ord = scalar2.inverse_transform([trainY])
    testPre = scalar2.inverse_transform(testPre)
    testY_ord = scalar2.inverse_transform([testY])

    # Evaluation Metrics
    trainScore = math.sqrt(mean_squared_error(trainY_ord[0, :], trainPre[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY_ord[0, :], testPre[:, 0]))
    MAE = mean_absolute_error(testY_ord[0, :], testPre[:, 0])

    print(f"Epoch {i+1}/{epochs}")
    print(f"Train RMSE: {trainScore:.2f}")
    print(f"Test RMSE: {testScore:.2f}")
    print(f"Test MAE : {MAE:.2f}")

    # Prepare plots
    trainPredictplot = np.full_like(scalar_dim, np.nan)
    trainPredictplot[look_back:len(trainPre) + look_back, :] = trainPre

    testPredictPlot = np.full_like(scalar_dim, np.nan)
    testPredictPlot[len(trainPre) + (look_back * 2) + 1:len(dataset) - 1, :] = testPre

    plt.figure(figsize=(10, 8))

    # Plot predictions vs ground truth
    plt.subplot(311)
    plt.plot(testY_ord[0, :], label='Ground Truth')
    plt.plot(testPre[:, 0], label='Prediction (testing)')
    plt.title(f'Epoch: {i+1}')
    plt.legend()

    # Plot only predictions
    plt.subplot(312)
    plt.plot(testY_ord[0, :])
    plt.plot(testPre[:, 0], label='Prediction')
    plt.legend()

    # Plot learning curve
    plt.subplot(313)
    plt.plot(loss_arr, label='Loss value')
    plt.title('Learning Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.pause(0.01)
    plt.close()

# =====================
# 5. Final Evaluation
# =====================
predict.my_self(testY_ord[0, :], testPre[:, 0], 'LSTM')
predict.score_calculation(testY_ord[0, :], testPre[:, 0])
predict.plot_pred(testY_ord[0, :], testPre[:, 0], 'LSTM')
predict.plot_residuals(testY_ord[0, :], testPre[:, 0], 'LSTM')
