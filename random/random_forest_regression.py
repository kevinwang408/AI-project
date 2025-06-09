import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import predict

dataset=pd.read_csv('SolarRadiationPrediction.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
#dataset=dataset.drop("Ra",axis=1)
# dataset=dataset.drop("TimeSunRise",axis=1)
# dataset=dataset.drop("TimeSunSet",axis=1)
target=dataset["Radiation"]
#dataset.head(5)

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
print("train_x: ",trainX.shape)
print("train y: ",trainY.shape)
print("testX: ",testX.shape)
print(trainX[0,:,:])
print(trainY[0])

trainX=trainX.reshape(5782,45)
testX=testX.reshape(2846,45)
trainY=trainY.reshape(len(trainY),1)
testY=testY.reshape(len(testY),1)
print(trainX.shape)
print(trainX[0,:])

np.random.seed(7)


dataset=dataset.astype('float32')
target=target.astype('float32')

scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar3=MinMaxScaler(feature_range=(0,1))
scalar4=MinMaxScaler(feature_range=(0,1))
#scalar_dim=dataset[:,1]
print("65")
#target=dataset[:,1]
trainX=scalar1.fit_transform(trainX)
testX=scalar2.fit_transform(testX)
trainY=scalar3.fit_transform(trainY)
testY=scalar4.fit_transform(testY)
#target=target.reshape(len(target),1)
# target=scalar2.fit_transform(target)
print("73")
trainY=trainY.ravel()
model = RandomForestRegressor(n_estimators=152,max_depth=4,max_features='sqrt',random_state=42)
model.fit(trainX, trainY)
print("76")
y_pred = model.predict(testX)
train_pre=model.predict(trainX)

y_pred=np.reshape(y_pred,(len(y_pred),1))
train_pre=np.reshape(train_pre,(len(train_pre),1))
trainY=np.reshape(trainY,(len(trainY),1))
#print(y_pred.shape)
# print(y_train.shape)
#print(train_pre.shape)
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

predict.my_self(testY,y_pred,'Random forest')
predict.score_calculation(testY,y_pred)
predict.plot_pred(testY,y_pred,'Random forest')
predict.plot_residuals(testY,y_pred,'Random forest')

    