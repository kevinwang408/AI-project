from data_preprocessing import load_and_preprocess
from dataset_creator import split_dataset, create_dataset
import model_SVR
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

trainPred, testPred = model_SVR.build_and_predict(trainX,testX,trainY)


#trainY_inv, trainPred_inv, _, _ = inverse_transform_and_evaluate(scaler_dim, trainY, trainPred)
testY_inv, testPred_inv, rmse, mae = inverse_transform_and_evaluate(scaler_dim, testY, testPred)

# plot
predict.my_self(testY_inv, testPred_inv, 'SVR')
predict.score_calculation(testY_inv, testPred_inv)
predict.plot_pred(testY_inv, testPred_inv, 'SVR')
predict.plot_residuals(testY_inv, testPred_inv, 'SVR')
