import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import predict

dataset=pd.read_csv('SolarRadiationPrediction.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
target=dataset["Radiation"]

dataset=dataset.values
target=target.values

dataset[:,1]=dataset[:,8]
dataset[:,8]=target[:]

def create_dataset(dataset,look_back):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        #a=dataset第0行i~i+look_back個
        a=dataset[i:(i+look_back),0:9]
        dataX.append(a)
        dataY.append(dataset[i+look_back,8])
    return np.array(dataX),np.array(dataY)

train_size=int(len(dataset)*0.67)
test_size=len(dataset)- train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]

trainX,trainY=create_dataset(train,5)
testX,testY=create_dataset(test,5)
# print("train_x: ",trainX.shape)
# print("train y: ",trainY.shape)
# print("testX: ",testX.shape)
# print(trainX[0,:,:])
# print(trainY[0])

trainX=trainX.reshape(5782,45)
testX=testX.reshape(2846,45)
trainY=trainY.reshape(len(trainY),1)
testY=testY.reshape(len(testY),1)
# print(trainX.shape)
# print(trainX[0,:])

# dataset=dataset.astype('float32')
# target=target.astype('float32')

scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar3=MinMaxScaler(feature_range=(0,1))
scalar4=MinMaxScaler(feature_range=(0,1))
#scalar_dim=dataset[:,1]

#target=dataset[:,1]
trainX=scalar1.fit_transform(trainX)
testX=scalar2.fit_transform(testX)
trainY=scalar3.fit_transform(trainY)
testY=scalar4.fit_transform(testY)
#target=target.reshape(len(target),1)
# target=scalar2.fit_transform(target)

model = DecisionTreeRegressor(max_depth=5,min_samples_split=18,min_samples_leaf=20,max_features='auto')
# 訓練模型
model.fit(trainX,trainY)

y_pred = model.predict(testX)
train_pre=model.predict(trainX)

y_pred=np.reshape(y_pred,(len(y_pred),1))
train_pre=np.reshape(train_pre,(len(train_pre),1))
print(y_pred.shape)
# print(y_train.shape)
print(train_pre.shape)
# print(y_test.shape)

train_pre=scalar3.inverse_transform(train_pre)
trainY=scalar3.inverse_transform(trainY)
y_pred=scalar4.inverse_transform(y_pred)
testY=scalar4.inverse_transform(testY)

trainScore=math.sqrt(mean_squared_error(trainY,train_pre))
print('Train Score:%.2f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY,y_pred))
print('Test Score:%.2f RMSE'%(testScore))
MAE=mean_absolute_error(testY,y_pred)
print('Test Score:%.2f MAE'%(MAE))

predict.my_self(testY,y_pred,'Decision Tree')
predict.score_calculation(testY,y_pred)
predict.plot_pred(testY,y_pred,'Decision Tree')
predict.plot_residuals(testY,y_pred,'Decision Tree')
