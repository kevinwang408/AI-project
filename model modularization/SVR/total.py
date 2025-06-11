import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import predict

# -------------------------------
# 1. Data Preprocessing Function
# -------------------------------
def load_and_preprocess_data(file_path: str, num_rows: int):
    """
    Load and clean solar radiation dataset.
    - Drop irrelevant columns
    - Swap columns for specific model needs
    - Return feature matrix and target array
    """
    dataset = pd.read_csv(file_path, engine='python', nrows=num_rows)
    dataset = dataset.drop(["Data", "Time"], axis=1)
    
    target = dataset["Radiation"].values
    dataset = dataset.values
    
    # Overwrite column 1 with column 8, and set column 8 as target
    dataset[:, 1] = dataset[:, 8]
    dataset[:, 8] = target[:]
    
    return dataset.astype('float32'), target.astype('float32')


# ------------------------------------------
# 2. Time Series Dataset Construction
# ------------------------------------------
def create_dataset(dataset, look_back):
    """
    Convert the dataset into input-output pairs using a sliding window.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        window = dataset[i:(i + look_back), 0:9]
        dataX.append(window)
        dataY.append(dataset[i + look_back, 8])
    return np.array(dataX), np.array(dataY)


# ------------------------------------------
# 3. Data Normalization and Reshaping
# ------------------------------------------
def reshape_and_normalize(trainX, testX, trainY, testY):
    """
    Reshape input for SVR and normalize all inputs/outputs using MinMaxScaler.
    """
    # Reshape: (samples, features)
    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)
    trainY = trainY.reshape(-1, 1)
    testY = testY.reshape(-1, 1)

    # Normalize
    scaler_X_train = MinMaxScaler()
    scaler_X_test = MinMaxScaler()
    scaler_Y_train = MinMaxScaler()
    scaler_Y_test = MinMaxScaler()
    
    trainX = scaler_X_train.fit_transform(trainX)
    testX = scaler_X_test.fit_transform(testX)
    trainY = scaler_Y_train.fit_transform(trainY)
    testY = scaler_Y_test.fit_transform(testY)

    return trainX, testX, trainY, testY, scaler_Y_train, scaler_Y_test


# ------------------------------------------
# 4. SVR Model Training and Evaluation
# ------------------------------------------
def train_and_evaluate_svr(trainX, trainY, testX, testY, scaler_Y_train, scaler_Y_test):
    """
    Train SVR and evaluate it on test data.
    """
    # Flatten target array
    trainY = trainY.ravel()

    # Train SVR model
    model = svm.SVR(kernel='linear', C=0.10127678320148709, epsilon=0.029240378785064282, gamma='auto')
    model.fit(trainX, trainY)

    # Predict
    train_pred = model.predict(trainX).reshape(-1, 1)
    test_pred = model.predict(testX).reshape(-1, 1)

    # Inverse scale
    train_pred = scaler_Y_train.inverse_transform(train_pred)
    trainY = scaler_Y_train.inverse_transform(trainY.reshape(-1, 1))
    test_pred = scaler_Y_test.inverse_transform(test_pred)
    testY = scaler_Y_test.inverse_transform(testY)

    # Evaluation
    rmse_train = math.sqrt(mean_squared_error(trainY, train_pred))
    rmse_test = math.sqrt(mean_squared_error(testY, test_pred))
    mae = mean_absolute_error(testY, test_pred)

    print(f'Train Score: {rmse_train:.2f} RMSE')
    print(f'Test Score: {rmse_test:.2f} RMSE')
    print(f'Test Score: {mae:.2f} MAE')

    # Prediction analysis
    predict.my_self(testY, test_pred, 'SVR')
    predict.score_calculation(testY, test_pred)
    predict.plot_pred(testY, test_pred, 'SVR')
    predict.plot_residuals(testY, test_pred, 'SVR')


# ------------------------------------------
# 5. Main Function
# ------------------------------------------
def main():
    np.random.seed(7)

    file_path = 'SolarRadiationPrediction.csv'
    num_rows = 576 * 15
    look_back = 5

    dataset, target = load_and_preprocess_data(file_path, num_rows)

    # Train-test split
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[:train_size], dataset[train_size:]

    # Time window dataset
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Normalization
    trainX, testX, trainY, testY, scaler_Y_train, scaler_Y_test = reshape_and_normalize(trainX, testX, trainY, testY)

    # Train and evaluate SVR
    train_and_evaluate_svr(trainX, trainY, testX, testY, scaler_Y_train, scaler_Y_test)


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
