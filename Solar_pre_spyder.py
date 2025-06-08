import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('SolarPrediction_aligned.csv',usecols=[3],engine='python',nrows=576)
dataset.head(5)

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(7)

dataset=dataset.values
dataset=dataset.astype('float32')

scalar=MinMaxScaler(feature_range=(0,1))

dataset=scalar.fit_transform(dataset)

train_size=int(len(dataset)*0.67)
test_size=len(dataset)- train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]


def create_dataset(dataset,look_back):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        #a=dataset第0行i~i+look_back個
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return numpy.array(dataX),numpy.array(dataY)

#根據前面幾個資料來看下一步 彙整成look_back行的資料
look_back=5
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)

#轉為三維
trainX=numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
testX=numpy.reshape(testX,(testX.shape[0],testX.shape[1],1))

model=Sequential()

model.add(LSTM(4,input_shape=(look_back,1)))
#Dense全連接後輸出一層
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

loss_arr=[]
epochs=500
for i in range(epochs):

    history=model.fit(trainX,trainY,epochs=1,batch_size=1,verbose=2)
    loss_arr.append(history.history['loss'][0])
    trainPre=model.predict(trainX)
    testPre=model.predict(testX)


    trainPre=scalar.inverse_transform(trainPre)
    trainY_ord=scalar.inverse_transform([trainY])
    testPre=scalar.inverse_transform(testPre)
    testY_ord=scalar.inverse_transform([testY])
    
    trainScore=math.sqrt(mean_squared_error(trainY_ord[0,:],trainPre[:,0]))
    print('Train Score:%.2f RMSE'%(trainScore))
    testScore=math.sqrt(mean_squared_error(testY_ord[0,:],testPre[:,0]))
    print('Train Score:%.2f RMSE'%(testScore))
    
    trainPredictplot=numpy.empty_like(dataset)
    #print(trainPredictplot)
    trainPredictplot[:,:]=numpy.nan
    #print(trainPredictplot)
    trainPredictplot[look_back:len(trainPre)+look_back,:]=trainPre
    #print(trainPredictplot)
    
    testPredictPlot=numpy.empty_like(dataset)
    testPredictPlot[:,:]=numpy.NaN
    testPredictPlot[len(trainPre)+(look_back*2)+1:len(dataset)-1,:]=testPre
    #print(testPredictPlot)
    
    plt.subplot(311)
    plt.cla()
    plt.plot(scalar.inverse_transform(dataset),label='Ground Truth')
    plt.plot(trainPredictplot,label='Prediction(training)')
    plt.plot(testPredictPlot,label='Prediction(testing)')
    plt.title('Epoch:'+str(i))
    plt.legend()
    plt.subplot(312)
    plt.cla()
    plt.plot(scalar.inverse_transform(dataset))
    plt.plot(trainPredictplot)
    plt.plot(testPredictPlot)
    plt.legend()
    plt.subplot(313)
    plt.cla()
    plt.plot(loss_arr,label='loss value')
    plt.title('learning curve')
    plt.legend()
    plt.show()
    plt.pause(0.01)
    plt.close()
    