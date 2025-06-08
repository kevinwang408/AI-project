import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import predict

dataset=pd.read_csv('SolarPrediction_aligned_sun.csv',engine='python',nrows=576*15)
dataset=dataset.drop("Data",axis=1)
dataset=dataset.drop("Time",axis=1)
# dataset=dataset.drop("TimeSunRise",axis=1)
# dataset=dataset.drop("TimeSunSet",axis=1)
dataset.head(5)

dataset=dataset.values

np.random.seed(7)


dataset=dataset.astype('float32')

scalar1=MinMaxScaler(feature_range=(0,1))
scalar2=MinMaxScaler(feature_range=(0,1))
scalar_dim=dataset[:,1]
dataset=scalar1.fit_transform(dataset)

scalar_dim=scalar_dim.reshape(len(dataset),1)
#print(scalar_dim)
scalar_dim=scalar2.fit_transform(scalar_dim)
#print(scalar_dim)

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

model=Sequential()

model.add(GRU(193,activation='tanh',input_shape=(look_back,9)))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(rate= 0.082379383823238))
#Dense全連接後輸出一層
model.add(Dense(units=1, activation="linear"))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model.fit(trainX,trainY,epochs=20,batch_size=16,verbose=2)


trainPre=model.predict(trainX)
testPre=model.predict(testX)

trainPre=scalar2.inverse_transform(trainPre)
trainY=scalar2.inverse_transform([trainY])
testPre=scalar2.inverse_transform(testPre)
testY=scalar2.inverse_transform([testY])

#MSE是均方根誤差 即平均誤差 會被平均除下去 所以數據多不代表誤差會增加
trainScore=math.sqrt(mean_squared_error(trainY[0,:],trainPre[:,0]))
print('Train Score:%.2f RMSE'%(trainScore))
testScore=math.sqrt(mean_squared_error(testY[0,:],testPre[:,0]))
print('Test Score:%.2f RMSE'%(testScore))
MAE=mean_absolute_error(testY[0,:],testPre[:,0])
print('Test Score:%.2f MAE'%(MAE))

trainPredictplot=np.empty_like(scalar_dim)
#print(trainPredictplot)
trainPredictplot[:,:]=np.nan
#print(trainPredictplot)
trainPredictplot[look_back:len(trainPre)+look_back,:]=trainPre
# print(trainPredictplot)
# print(trainPredictplot.shape)

testPredictPlot=np.empty_like(scalar_dim)
testPredictPlot[:,:]=np.NaN
testPredictPlot[len(trainPre)+(look_back*2)+1:len(dataset)-1,:]=testPre
#print(testPredictPlot)

ground_truth=scalar2.inverse_transform(scalar_dim)
predict.my_self(testY[0,:],testPre[:,0],'GRU')
predict.score_calculation(testY[0,:],testPre[:,0])
predict.plot_pred(testY[0,:],testPre[:,0],'GRU')
predict.plot_residuals(testY[0,:],testPre[:,0],'GRU')