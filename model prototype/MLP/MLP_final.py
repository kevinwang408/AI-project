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
import predict

dataset=pd.read_csv('SolarRadiationPrediction.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
# dataset=dataset.drop("TimeSunRise",axis=1)
# dataset=dataset.drop("TimeSunSet",axis=1)
target=dataset["Radiation"]
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

trainX=trainX.reshape(5782,45)
testX=testX.reshape(2846,45)
trainY=trainY.reshape(len(trainY),1)
testY=testY.reshape(len(testY),1)

model = MLPRegressor(hidden_layer_sizes=(4,210), max_iter=328,alpha=0.0201293522716579 , random_state=42)
model.fit(trainX, trainY)
#model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

y_pred = model.predict(testX)
train_pre=model.predict(trainX)


train_pre=train_pre.reshape(len(train_pre),1)
y_pred=y_pred.reshape(len(y_pred),1)
trainPre=scalar2.inverse_transform(train_pre)
trainY_ord=scalar2.inverse_transform(trainY)
testPre=scalar2.inverse_transform(y_pred)
testY_ord=scalar2.inverse_transform(testY)

#print(X_test)
print(y_pred)
print(testY)

trainScore=math.sqrt(mean_squared_error(trainY_ord,trainPre))
print('Train Score:%.2f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY_ord,testPre))
print('Test Score:%.2f RMSE'%(testScore))
MAE=mean_absolute_error(testY_ord,testPre)
print('Test Score:%.2f MAE'%(MAE))

predict.my_self(testY_ord,testPre,'MLPRegressor')
predict.score_calculation(testY_ord,testPre)
predict.plot_pred(testY_ord,testPre,'MLPRegressor')
predict.plot_residuals(testY_ord,testPre,'MLPRegressor')