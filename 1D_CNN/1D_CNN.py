import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten ,MaxPooling1D
from sklearn.metrics import mean_absolute_error ,mean_squared_error
import matplotlib.pyplot as plt
import predict

dataset=pd.read_csv('SolarRadiationPrediction.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
# dataset=dataset.drop("TimeSunRise",axis=1)
# dataset=dataset.drop("TimeSunSet",axis=1)
target=dataset["Radiation"]
#dataset.head(5)

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

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(trainX.shape[1], 9)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(trainX, trainY, epochs=100, batch_size=21, validation_data=(testX, testY))

trainPre=model.predict(trainX)
testPre=model.predict(testX)
trainPre.shape
testPre.shape

trainPre=scalar2.inverse_transform(trainPre)
trainY_ord=scalar2.inverse_transform([trainY])
testPre=scalar2.inverse_transform(testPre)
testY_ord=scalar2.inverse_transform([testY])

trainScore=math.sqrt(mean_squared_error(trainY_ord[0,:],trainPre[:,0]))
print('Train Score:%.2f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY_ord[0,:],testPre[:,0]))
print('Test Score:%.2f RMSE'%(testScore))
MAE=mean_absolute_error(testY_ord[0,:],testPre[:,0])
print('Test Score:%.2f MAE'%(MAE))

trainPredictplot=np.empty_like(scalar_dim)
trainPredictplot[:,:]=np.nan
trainPredictplot[look_back:len(trainPre)+look_back,:]=trainPre

testPredictPlot=np.empty_like(scalar_dim)
testPredictPlot[:,:]=np.NaN
testPredictPlot[len(trainPre)+(look_back*2)+1:len(dataset)-1,:]=testPre
    
predict.my_self(testY_ord[0,:],testPre[:,0],'1D_CNN')
predict.score_calculation(testY_ord[0,:],testPre[:,0])
predict.plot_pred(testY_ord[0,:],testPre[:,0],'1D_CNN')
predict.plot_residuals(testY_ord[0,:],testPre[:,0],'1D_CNN')

