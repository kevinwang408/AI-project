import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tcn import TCN, tcn_full_summary
from optuna.samplers import TPESampler,CmaEsSampler
import optuna
batch_size, time_steps, input_dim = 20, 5, 9
dataset=pd.read_csv('SolarRadiationPrediction.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
dataset.head(5)
dataset=dataset.values


dataset=dataset.astype('float32')

scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar_dim=dataset[:,1]
dataset=scalar1.fit_transform(dataset)

scalar_dim=scalar_dim.reshape(len(dataset),1)
print(scalar_dim)
scalar_dim=scalar2.fit_transform(scalar_dim)
print(scalar_dim)

train_size=int(len(dataset)*0.67)
test_size=len(dataset)- train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]

def create_dataset(dataset,look_back):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        #a=dataset第0行i~i+look_back個
        a=dataset[i:(i+look_back),0:9]
        dataX.append(a)
        dataY.append(dataset[i+look_back,1])
    return np.array(dataX),np.array(dataY)

look_back=5
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)

def objective(trial):
    tcn_layer = TCN(
                    nb_filters=trial.suggest_int("nb_filters", 1, 128),
                    kernel_size=trial.suggest_int("kernel_size",5,10),
                    dropout_rate=trial.suggest_float("dropout_rate",0.1,0.5),
                    #activation=trial.suggest_categorical("activation",['relu','tanh','linear']),
                    nb_stacks=trial.suggest_int("nb_stacks", 1, 3),
                    padding='causal',
                    input_shape=(time_steps, input_dim))
    print('Receptive field size =', tcn_layer.receptive_field)

    model = Sequential([tcn_layer,Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    tcn_full_summary(model, expand_residual_blocks=False)
    model.fit(trainX, trainY, epochs=5, validation_split=0.2)
    # 進行預測
    y_pred = model.predict(testX)

    # 計算均方誤差
    mse = mean_squared_error(testY, y_pred)

    return mse

sampler = optuna.samplers.TPESampler()
study = optuna.create_study(direction="minimize",sampler=sampler)
study.optimize(objective, n_trials=20)

# 取得最佳超參數
best_params = study.best_params
print("Best Hyperparameters:", best_params)