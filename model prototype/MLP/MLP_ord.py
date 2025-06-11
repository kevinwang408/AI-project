import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense,Dropout
#from tensorflow.keras.optimizers import RMSprop

dataset=pd.read_csv('SolarRadiationPrediction.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
# dataset=dataset.drop("TimeSunRise",axis=1)
# dataset=dataset.drop("TimeSunSet",axis=1)
target=dataset["Radiation"]
dataset.head(5)

dataset=dataset.values
target=target.values

train_size=int(len(dataset)*0.67)
test_size=len(dataset)- train_size
trainX,testX=dataset[0:train_size,:],dataset[train_size:len(dataset),:]
trainY,testY=target[0:train_size],target[train_size:len(target)]

trainY=trainY.reshape(len(trainY),1)
testY=testY.reshape(len(testY),1)
scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar3=MinMaxScaler(feature_range=(0,1))
scalar4=MinMaxScaler(feature_range=(0,1))
trainX=scalar1.fit_transform(trainX)
testX=scalar2.fit_transform(testX)
trainY=scalar3.fit_transform(trainY)
testY=scalar4.fit_transform(testY)

model = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=1000, random_state=42)
model.fit(trainX,trainY)

y_pred = model.predict(testX)
train_pre=model.predict(trainX)

train_pre=train_pre.reshape(len(train_pre),1)
y_pred=y_pred.reshape(len(y_pred),1)
train_pre=scalar3.inverse_transform(train_pre)
trainY=scalar3.inverse_transform(trainY)
y_pred=scalar4.inverse_transform(y_pred)
testY=scalar4.inverse_transform(testY)

#print(X_test)
# print(y_pred)
# print(testY)

trainScore=math.sqrt(mean_squared_error(trainY,train_pre))
print('Train Score:%.2f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY,y_pred))
print('Test Score:%.2f RMSE'%(testScore))
MAE=mean_absolute_error(testY,y_pred)
print('Test Score:%.2f MAE'%(MAE))

plt.plot(testY,label='Ground Truth')
plt.plot(y_pred,label='Prediction')
#plt.plot(testPredictPlot[0:200],label='Prediction(testing)')
plt.legend()
plt.show()