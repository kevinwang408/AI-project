import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import predict


def load_data(path, num_rows=576*15):
    """
    Load and preprocess the solar radiation dataset.
    """
    dataset = pd.read_csv(path, engine='python', nrows=num_rows)
    dataset = dataset.drop(columns=["Data", "Time"])  # Remove non-numeric columns

    target = dataset["Radiation"].values
    dataset = dataset.values.astype('float32')

    # Swap Radiation column to position 8 for convenience
    dataset[:, 1] = dataset[:, 8]
    dataset[:, 8] = target

    return dataset, target


def create_dataset(dataset, look_back):
    """
    Generate time-windowed input features (X) and targets (Y).
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        window = dataset[i:(i + look_back), 0:9]
        dataX.append(window)
        dataY.append(dataset[i + look_back, 8])  # Radiation target at future time step
    return np.array(dataX), np.array(dataY)


def preprocess_data(trainX, testX, trainY, testY):
    """
    Normalize features and targets separately using MinMaxScaler.
    """
    scaler_X_train = MinMaxScaler()
    scaler_X_test = MinMaxScaler()
    scaler_Y_train = MinMaxScaler()
    scaler_Y_test = MinMaxScaler()

    trainX = scaler_X_train.fit_transform(trainX)
    testX = scaler_X_test.fit_transform(testX)
    trainY = scaler_Y_train.fit_transform(trainY)
    testY = scaler_Y_test.fit_transform(testY)

    return trainX, testX, trainY, testY, scaler_Y_train, scaler_Y_test


def evaluate_model(true_Y, pred_Y, label='Random Forest'):
    """
    Evaluate model performance and plot results.
    """
    rmse = math.sqrt(mean_squared_error(true_Y, pred_Y))
    mae = mean_absolute_error(true_Y, pred_Y)

    print(f'{label} - Test RMSE: {rmse:.2f}')
    print(f'{label} - Test MAE : {mae:.2f}')

    predict.my_self(true_Y, pred_Y, label)
    predict.score_calculation(true_Y, pred_Y)
    predict.plot_pred(true_Y, pred_Y, label)
    predict.plot_residuals(true_Y, pred_Y, label)


def run_random_forest_pipeline(file_path):
    """
    Main pipeline to run RandomForestRegressor on solar radiation data.
    """
    LOOK_BACK = 5
    dataset, target = load_data(file_path)

    train_size = int(len(dataset) * 0.67)
    train, test = dataset[:train_size], dataset[train_size:]

    trainX, trainY = create_dataset(train, LOOK_BACK)
    testX, testY = create_dataset(test, LOOK_BACK)

    # Flatten input features to 2D for scikit-learn
    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)
    trainY = trainY.reshape(-1, 1)
    testY = testY.reshape(-1, 1)

    # Normalize data
    trainX, testX, trainY_norm, testY_norm, y_scaler_train, y_scaler_test = preprocess_data(
        trainX, testX, trainY, testY
    )

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=152,
        max_depth=4,
        max_features='sqrt',
        random_state=42
    )
    model.fit(trainX, trainY_norm.ravel())

    # Predict
    train_pred = model.predict(trainX).reshape(-1, 1)
    test_pred = model.predict(testX).reshape(-1, 1)

    # Inverse transform predictions
    train_pred_inv = y_scaler_train.inverse_transform(train_pred)
    trainY_inv = y_scaler_train.inverse_transform(trainY_norm)
    test_pred_inv = y_scaler_test.inverse_transform(test_pred)
    testY_inv = y_scaler_test.inverse_transform(testY_norm)

    # Evaluate
    print("===== Training Set Evaluation =====")
    evaluate_model(trainY_inv, train_pred_inv, label='Random Forest (Train)')
    print("===== Test Set Evaluation =====")
    evaluate_model(testY_inv, test_pred_inv, label='Random Forest (Test)')


# If this script is run directly, execute the pipeline:
if __name__ == "__main__":
    run_random_forest_pipeline("SolarRadiationPrediction.csv")
