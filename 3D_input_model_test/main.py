from data_preprocessing import load_and_preprocess
from dataset_creator import split_dataset, create_dataset
import model_1dcnn 
from utils import inverse_transform_and_evaluate
import predict  

LOOK_BACK = 5
INPUT_DIM = 9
TARGET_INDEX = 1

# data preprocessing
dataset_scaled, scalar_dim_scaled, scaler_dim, scaler_all = load_and_preprocess('SolarRadiationPrediction.csv', nrows=576*15)

# divide data into time seqences
train, test = split_dataset(dataset_scaled)
trainX, trainY = create_dataset(train, look_back=LOOK_BACK, input_dim=INPUT_DIM, target_index=TARGET_INDEX)
testX, testY = create_dataset(test, look_back=LOOK_BACK, input_dim=INPUT_DIM, target_index=TARGET_INDEX)

# prediction and evaluation
trainPred, testPred= model_1dcnn.build_predict(LOOK_BACK,INPUT_DIM,trainX,trainY,testX,testY)
trainY_inv, trainPred_inv, _, _ = inverse_transform_and_evaluate(scaler_dim, trainY, trainPred)
testY_inv, testPred_inv, rmse, mae = inverse_transform_and_evaluate(scaler_dim, testY, testPred)

# plot
predict.my_self(testY_inv, testPred_inv, 'LSTM')
predict.score_calculation(testY_inv, testPred_inv)
predict.plot_pred(testY_inv, testPred_inv, 'LSTM')
predict.plot_residuals(testY_inv, testPred_inv, 'LSTM')
