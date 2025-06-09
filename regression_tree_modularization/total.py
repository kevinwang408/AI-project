import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import predict  # User-defined module for plotting and evaluation

# Load and preprocess the dataset
def load_and_preprocess_data(filepath, num_rows=576*15):
    dataset = pd.read_csv(filepath, engine='python', nrows=num_rows)
    dataset = dataset.drop(["Data", "Time"], axis=1)
    target = dataset["Radiation"].values

    dataset = dataset.values
    dataset[:, 1] = dataset[:, 8]  # Move radiation to the second column
    dataset[:, 8] = target         # Restore radiation at index 8

    return dataset

# Create the feature and target dataset with look_back window
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        # Get a window of `look_back` steps as input
        a = dataset[i:(i + look_back), 0:9]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 8])  # Radiation as target
    return np.array(dataX), np.array(dataY)

# Prepare training and testing datasets
def split_and_scale_data(dataset, look_back):
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[:train_size, :], dataset[train_size:, :]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape 3D input (samples, time_steps, features) to 2D for DecisionTree input
    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)
    trainY = trainY.reshape(-1, 1)
    testY = testY.reshape(-1, 1)

    # Scale inputs and targets separately
    scalers = {
        'trainX': MinMaxScaler((0, 1)),
        'testX': MinMaxScaler((0, 1)),
        'trainY': MinMaxScaler((0, 1)),
        'testY': MinMaxScaler((0, 1))
    }
    trainX = scalers['trainX'].fit_transform(trainX)
    testX = scalers['testX'].fit_transform(testX)
    trainY = scalers['trainY'].fit_transform(trainY)
    testY = scalers['testY'].fit_transform(testY)

    return trainX, trainY, testX, testY, scalers

# Train the Decision Tree model and evaluate

def train_and_evaluate(trainX, trainY, testX, testY, scalers):
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=18,
        min_samples_leaf=20,
        max_features='auto'  # For regression, this means sqrt(n_features)
    )
    model.fit(trainX, trainY)

    train_pred = model.predict(trainX).reshape(-1, 1)
    test_pred = model.predict(testX).reshape(-1, 1)

    # Inverse scale predictions and actual values
    train_pred = scalers['trainY'].inverse_transform(train_pred)
    trainY = scalers['trainY'].inverse_transform(trainY)
    test_pred = scalers['testY'].inverse_transform(test_pred)
    testY = scalers['testY'].inverse_transform(testY)

    # Calculate and print evaluation metrics
    train_rmse = math.sqrt(mean_squared_error(trainY, train_pred))
    test_rmse = math.sqrt(mean_squared_error(testY, test_pred))
    test_mae = mean_absolute_error(testY, test_pred)

    print(f'Train Score: {train_rmse:.2f} RMSE')
    print(f'Test Score: {test_rmse:.2f} RMSE')
    print(f'Test Score: {test_mae:.2f} MAE')

    # Use external plotting and evaluation functions
    predict.my_self(testY, test_pred, 'Decision Tree')
    predict.score_calculation(testY, test_pred)
    predict.plot_pred(testY, test_pred, 'Decision Tree')
    predict.plot_residuals(testY, test_pred, 'Decision Tree')
    

# Main pipeline
if __name__ == "__main__":
    data_path = 'SolarRadiationPrediction.csv'
    look_back = 5
    dataset = load_and_preprocess_data(data_path)
    trainX, trainY, testX, testY, scalers = split_and_scale_data(dataset, look_back)
    train_and_evaluate(trainX, trainY, testX, testY, scalers)
