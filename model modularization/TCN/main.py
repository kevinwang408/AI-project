from data_preprocessing import load_and_preprocess
from dataset_creator import split_dataset, create_dataset
from model_TCN import build_TCN
from utils import inverse_transform_and_evaluate
import predict  
import os
LOOK_BACK = 5
INPUT_DIM = 9
TARGET_INDEX = 1
current_dir = os.path.dirname(__file__)  
file_path = os.path.join(current_dir, 'SolarRadiationPrediction.csv')  

# data preprocessing
dataset_scaled, scalar_dim_scaled, scaler_dim, scaler_all = load_and_preprocess(file_path, nrows=576*15)

# divide data into time seqences
train, test = split_dataset(dataset_scaled)
trainX, trainY = create_dataset(train, look_back=LOOK_BACK, input_dim=INPUT_DIM, target_index=TARGET_INDEX)
testX, testY = create_dataset(test, look_back=LOOK_BACK, input_dim=INPUT_DIM, target_index=TARGET_INDEX)

# build and train the model
model = build_TCN(input_shape=(LOOK_BACK, INPUT_DIM))
model.fit(trainX, trainY, epochs=10, batch_size=21, validation_data=(testX, testY))

# prediction and evaluation
trainPred = model.predict(trainX)
testPred = model.predict(testX)

#trainY_inv, trainPred_inv, _, _ = inverse_transform_and_evaluate(scaler_dim, trainY, trainPred)
testY_inv, testPred_inv, rmse, mae = inverse_transform_and_evaluate(scaler_dim, testY, testPred)

# plot
predict.my_self(testY_inv, testPred_inv, 'LSTM')
predict.score_calculation(testY_inv, testPred_inv)
predict.plot_pred(testY_inv, testPred_inv, 'LSTM')
predict.plot_residuals(testY_inv, testPred_inv, 'LSTM')
